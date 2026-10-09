import random
import warnings
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
import torch
import os
import triton

C_TUNE = "\033[96m"  # 青色
C_NAME = "\033[95m"  # 紫色
C_DESC = "\033[93m"  # 黄色
C_ALGO = "\033[92m"  # 绿色
C_TIME = "\033[91m"  # 红色
C_BEST = "\033[1;92m"  # 加粗绿色
C_RST = "\033[0m"  # 重置颜色
ENABLE_AUTOTUNE = True


class AutotunerPrecisionMismatch(RuntimeError):
    # 只捕获精度有问题的的 Error, 其他 Exception 仍然照抄原样抛出
    pass


class RuntimeAutotuner:
    """
    Runtime kernel auto-tuner for selecting optimal implementation per bucket.

    Usage:
        # In __init__:
        self.attn_tuner = RuntimeAutotuner(name=self.__class__, fallback="flash_attn")

        # In forward: candidates 接受 args 参数，benchmark 时自动 randn_like 保护输入
        candidates = {
            "algo_a": lambda x: algo_a_fn(x, ...),
            "algo_b": lambda x: algo_b_fn(x, ...),
        }
        output = tuner.dispatch(bucket, candidates, is_warmup, args=(input_tensor,))
    """

    # Class variable: {name: {bucket: algo_name}}
    _shared_tables: dict = {}
    _cache_path: str | None = None
    _cache_data: dict | None = None
    _cache_load_logged: bool = False
    enable = os.environ.get("ENABLE_AUTOTUNE") or ENABLE_AUTOTUNE
    enable = enable.lower() not in ["0", "f", "false"] if isinstance(enable, str) else enable

    def __init__(self, name, fallback: str, check_correctness: bool = True, strict: bool = False, atol: float = 1e-2, rtol: float = 1e-2):
        self.name = name
        self.fallback = fallback
        self.check_correctness = check_correctness
        self.strict = strict  # 严格模式：精度不通过直接 raise
        self.atol = atol
        self.rtol = rtol
        if name not in RuntimeAutotuner._shared_tables:
            RuntimeAutotuner._shared_tables[name] = {}

    @property
    def dispatch_table(self) -> dict[int, str]:
        return RuntimeAutotuner._shared_tables[self.name]

    @staticmethod
    def _clone_safe_args(args: tuple | None) -> tuple | None:
        """Clone args 中的 Tensor 并填充随机数据，用于精度检查"""
        if args is None:
            return None
        out = []
        for a in args:
            out.append(a.clone() if isinstance(a, torch.Tensor) else a)
        return tuple(out)

    @staticmethod
    def _call(fn: Callable, args: tuple | None):
        return fn(*args) if args is not None else fn()

    @staticmethod
    def _default_json_cache_path() -> str:
        return str(Path(__file__).resolve().parent / "runtime_autotune.json")

    @classmethod
    def _json_cache_path(cls) -> str:
        return os.environ.get("AUTOTUNE_CACHE_JSON", "").strip() or cls._default_json_cache_path()

    @staticmethod
    def _cache_log_enabled() -> bool:
        value = os.environ.get("AUTOTUNE_CACHE_LOG", "1")
        return value.lower() not in ["0", "f", "false", "off", "no"]

    @classmethod
    def _log(cls, message: str) -> None:
        if cls._cache_log_enabled():
            print(f"{C_TUNE}[AutotuneCache]{C_RST} {message}", flush=True)

    @classmethod
    def _load_json_cache(cls) -> dict | None:
        path = cls._json_cache_path()
        if cls._cache_data is not None and cls._cache_path == path:
            return cls._cache_data

        cls._cache_path = path
        try:
            with open(path) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("cache root is not a JSON object")
            entries = data.get("entries")
            if not isinstance(entries, dict):
                data["entries"] = {}
            data["version"] = int(data.get("version", 1))
        except FileNotFoundError:
            data = {"version": 1, "entries": {}}
        except Exception as exc:
            warnings.warn(f"RuntimeAutotuner: failed to load AUTOTUNE_CACHE_JSON={path!r}: {exc}; using empty cache")
            data = {"version": 1, "entries": {}}

        cls._cache_data = data
        if not cls._cache_load_logged:
            cls._log(f"load path={path} entries={len(data.get('entries', {}))}")
            cls._cache_load_logged = True
        return cls._cache_data

    @classmethod
    def _save_json_cache(cls) -> None:
        path = cls._cache_path or cls._json_cache_path()
        if not path or cls._cache_data is None:
            return
        try:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            tmp_path = f"{path}.tmp.{os.getpid()}"
            with open(tmp_path, "w") as f:
                json.dump(cls._cache_data, f, indent=2, sort_keys=True)
                f.write("\n")
            os.replace(tmp_path, path)
            cls._log(f"save path={path} entries={len(cls._cache_data.get('entries', {}))}")
        except Exception as exc:
            warnings.warn(f"RuntimeAutotuner: failed to save AUTOTUNE_CACHE_JSON={path!r}: {exc}")

    def _name_key(self) -> str:
        return self.name if isinstance(self.name, str) else getattr(self.name, "__name__", str(self.name))

    @staticmethod
    def _candidate_sig(candidates: dict[str, Callable]) -> str:
        return "|".join(sorted(candidates.keys()))

    def _json_cache_key(self, dispatch_key: str, candidates: dict[str, Callable]) -> str:
        return f"{self._name_key()}::{dispatch_key}::{self._candidate_sig(candidates)}"

    def _lookup_json_best(self, dispatch_key: str, candidates: dict[str, Callable]) -> str | None:
        cache = RuntimeAutotuner._load_json_cache()
        if cache is None:
            return None
        cache_key = self._json_cache_key(dispatch_key, candidates)
        entry = cache.get("entries", {}).get(cache_key)
        if not isinstance(entry, dict):
            RuntimeAutotuner._log(f"miss tuner={self._name_key()} key={dispatch_key} candidates={self._candidate_sig(candidates)}")
            return None
        best = entry.get("best")
        if isinstance(best, str) and best in candidates:
            RuntimeAutotuner._log(f"hit tuner={self._name_key()} key={dispatch_key} best={best}")
            return best
        RuntimeAutotuner._log(f"stale tuner={self._name_key()} key={dispatch_key} cached_best={best!r}")
        return None

    def _store_json_best(self, dispatch_key: str, candidates: dict[str, Callable], best_name: str) -> None:
        cache = RuntimeAutotuner._load_json_cache()
        if cache is None:
            return
        cache.setdefault("entries", {})[self._json_cache_key(dispatch_key, candidates)] = {
            "best": best_name,
            "candidates": sorted(candidates.keys()),
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        RuntimeAutotuner._save_json_cache()

    def _check_correctness(self, candidates: dict[str, Callable], fallback: str, args: tuple | None) -> dict[str, str]:

        failed = {}
        if not self.check_correctness:
            return failed

        rand_args = self._clone_safe_args(args)
        baseline_out = self._call(candidates[fallback], rand_args)

        _base = baseline_out if isinstance(baseline_out, tuple) else (baseline_out,)
        _base = tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in _base)

        for name, fn in candidates.items():
            if name == fallback:
                continue
            try:
                cand_args = self._clone_safe_args(args)
                out = self._call(fn, cand_args)

                _out = out if isinstance(out, tuple) else (out,)
                _out = tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in _out)

                tensor_pairs = [(o, b) for o, b in zip(_out, _base) if isinstance(o, torch.Tensor)]

                diffs = [(o - b).abs().max().item() for o, b in tensor_pairs]
                is_close = all(torch.allclose(o, b, atol=self.atol, rtol=self.rtol) for o, b in tensor_pairs)

                if not is_close:
                    msg = f"PRECISION MISMATCH (max_diff={max(diffs):.6f})"
                    failed[name] = msg
                    raise AutotunerPrecisionMismatch(f"RuntimeAutotuner: candidate '{name}' {msg}")

            except AutotunerPrecisionMismatch:
                # 严格模式下直接 raise，不继续测其他候选者了
                if self.strict:
                    raise

        return failed

    def _benchmark(self, candidates: dict[str, Callable], fallback: str, args: tuple | None, description: str, check_correctness: bool = False, use_cudagraph: bool = True) -> str:

        RuntimeAutotuner._log(f"start tuner={self._name_key()} key={description} candidates={self._candidate_sig(candidates)}")
        precision_failed: dict[str, str] = {}
        if check_correctness:
            precision_failed = self._check_correctness(candidates, fallback, args)

        results = {}
        best_name, best_time = None, float("inf")
        bench_fn_maker = triton.testing.do_bench_cudagraph if use_cudagraph else triton.testing.do_bench
        bench_kwargs = {"rep": 200} if use_cudagraph else {"warmup": 25, "rep": 200}

        for name, fn in candidates.items():
            if name in precision_failed:
                results[name] = precision_failed[name]
                continue

            # =====================================================================
            # 【闭包延迟绑定 (Late Binding) 陷阱】
            # 如果你这样写：bench_fn_maker(lambda: self._call(fn, args), **bench_kwargs)
            # 会导致一个致命 BUG：所有候选者测出来的全是字典里的“最后一个函数”。
            #
            # 根本原因：
            # 1. 延迟绑定：Python 的闭包（lambda）在引用外部变量 `fn` 时，记住的是它的
            #    “引用”，而不是循环到这一步时的“值”。
            # 2. 运行时求值：只有当 lambda 真正被 bench_fn_maker 执行时，它才会去外面看
            #    `fn` 是什么。但此时 for 循环往往已经跑完了，`fn` 已经变成了最后一个元素。
            #
            # 为什么 `f=fn` 能解决：
            # Python 函数的【默认参数】是在【函数定义时】立即求值的。
            # 通过 `f=fn`，我们在每次循环创建 lambda 的瞬间，强行把当时的 `fn` 拍了个
            # “快照”，死死绑定在了局部变量 `f` 上，从而完美固化了上下文。
            # =====================================================================
            t = bench_fn_maker(lambda f=fn: self._call(f, args), **bench_kwargs)
            results[name] = t
            if t < best_time:
                best_name, best_time = name, t

        if best_name is None:
            best_name = fallback

        times_str = ", ".join(f"{k}={v:.3f}ms" if isinstance(v, float) else f"{k}={v}" for k, v in results.items())
        print(
            f"{C_TUNE}[RuntimeAutotuner]{C_RST}"
            f"{C_NAME}[{self.name if isinstance(self.name, str) else self.name.__name__}]{C_RST} "
            f"{C_DESC}{description}{C_RST}: "
            f"{times_str} -> "
            f"best={C_BEST}'{best_name}'{C_RST}"
        )
        RuntimeAutotuner._log(f"done tuner={self._name_key()} key={description} best={best_name}")

        return best_name

    def dispatch(self, bucket: int, candidates: dict[str, Callable], is_warmup: bool, fallback: str | None = None, args: tuple | None = None):
        fallback = fallback or self.fallback
        memory_key = bucket
        dispatch_key = f"bucket:{bucket}"

        if memory_key in self.dispatch_table:
            return self._call(candidates[self.dispatch_table[memory_key]], args)

        cached_best = self._lookup_json_best(dispatch_key, candidates)
        if cached_best is not None:
            self.dispatch_table[memory_key] = cached_best
            return self._call(candidates[cached_best], args)

        if not self.enable or len(candidates) == 1:
            RuntimeAutotuner._log(f"fallback tuner={self._name_key()} key={dispatch_key} reason={'disabled' if not self.enable else 'single_candidate'} fallback={fallback}")
            runner = candidates[fallback] if fallback else random.choice(list(candidates.values()))
            return self._call(runner, args)

        if is_warmup:
            best_name = self._benchmark(
                candidates=candidates, fallback=fallback, args=args, description=f"bucket:{bucket}", check_correctness=self.check_correctness, use_cudagraph=True
            )
            self.dispatch_table[memory_key] = best_name
            self._store_json_best(dispatch_key, candidates, best_name)
            return self._call(candidates[self.dispatch_table[memory_key]], args)

        warnings.warn(f"RuntimeAutotuner: bucket {bucket} not tuned, using fallback '{fallback}'")
        return self._call(candidates[fallback], args)

    def dispatch_once(self, candidates: dict[str, Callable], name: str = "default", fallback: str | None = None, args: tuple | None = None):
        # 与 self.dispatch 的区别就是，不用根据 bucket 值来区分，只需要一次
        fallback = fallback or self.fallback
        key = f"once:{name}"

        if key in self.dispatch_table:
            return self._call(candidates[self.dispatch_table[key]], args)

        cached_best = self._lookup_json_best(key, candidates)
        if cached_best is not None:
            self.dispatch_table[key] = cached_best
            return self._call(candidates[cached_best], args)

        if not self.enable or len(candidates) == 1:
            RuntimeAutotuner._log(f"fallback tuner={self._name_key()} key={key} reason={'disabled' if not self.enable else 'single_candidate'} fallback={fallback}")
            runner = candidates[fallback] if fallback else random.choice(list(candidates.values()))
            return self._call(runner, args)

        best_name = self._benchmark(candidates=candidates, fallback=fallback, args=args, description=key, check_correctness=self.check_correctness, use_cudagraph=True)
        self.dispatch_table[key] = best_name
        self._store_json_best(key, candidates, best_name)
        return self._call(candidates[self.dispatch_table[key]], args)
