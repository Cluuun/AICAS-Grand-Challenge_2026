import torch
from functools import wraps
from collections import deque
import functools


def cuda_time_profiler(max_history=100):
    """
    专门用于统计 PyTorch GPU 算子耗时的装饰器
    """
    # first in first out
    history = deque(maxlen=max_history)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            result = func(*args, **kwargs)
            end_event.record()

            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)

            history.append(elapsed_time)

            avg_time = sum(history) / len(history)
            print(f"[Profiler - {func.__name__}] 当前: {elapsed_time:.4f} ms | 最近 {len(history)} 次均值: {avg_time:.4f} ms")

            return result

        return wrapper

    return decorator


def shape_cache_decorator(func):

    cache = {}
    initialized = False

    # 预设的高频 Shape 列表
    # fmt: off
    PREDEFINED_SHAPES = [
        [1, 48, 64], [1, 42, 64], [1, 64, 48], [1, 64, 64], [1, 64, 42], 
        [1, 36, 64], [1, 44, 64], [1, 40, 64], [1, 64, 46], [1, 46, 64],
        [1, 64, 44], [1, 64, 40], [1, 48, 48], [1, 64, 52], [1, 52, 64],
        [1, 38, 64], [1, 56, 64], [1, 56, 56], [1, 54, 56],
        [1, 50, 62], [1, 62, 50], [1, 52, 58], [1, 58, 52], [1, 52, 60], [1, 60, 52],
        [1, 62, 64], [1, 64, 50], [1, 64, 38],
        [1, 34, 64], [1, 50, 64], [1, 64, 54], [1, 64, 36], [1, 54, 64],
        [1, 64, 60], [1, 64, 58], [1, 64, 56], [1, 58, 64], [1, 32, 64],
        [1, 64, 28], [1, 64, 62], [1, 28, 64], [1, 64, 26], [1, 64, 30],
        [1, 64, 24], [1, 60, 64], [1, 26, 64], [1, 64, 22], [1, 30, 64],
        [1, 48, 62], [1, 56, 48], [1, 60, 48], [1, 64, 32], [1, 22, 64],
        [1, 48, 60], [1, 24, 64], [1, 20, 64], [1, 48, 52], [1, 54, 48],
        [1, 50, 48], [1, 58, 48], [1, 48, 54], [1, 52, 48], [1, 48, 50],
        [1, 64, 20], [1, 64, 16]
    ]
    # fmt: on

    @functools.wraps(func)
    def wrapper(self, grid_thw_list, *args, **kwargs):
        nonlocal initialized

        # ==========================================
        # 阶段一：首次执行时的串行预热 (带进度日志)
        # ==========================================
        if not initialized:
            print(f"\n🚀 [{func.__name__}] 首次调用触发！开始串行预热 {len(PREDEFINED_SHAPES)} 个高频 Shape...")

            for i, shape in enumerate(PREDEFINED_SHAPES, 1):
                key = tuple(shape)
                print(f"   ⏳ [{i}/{len(PREDEFINED_SHAPES)}] 正在计算: {shape} ...", end="", flush=True)

                # 传入纯 list 计算，强制生成缓存
                res = func(self, [shape], *args, **kwargs)

                if isinstance(res, tuple):
                    cache[key] = tuple(r.detach().clone() for r in res)
                else:
                    cache[key] = res.detach().clone()

                print(" 完成!")  # 这一行算完了才会打印

            print(f"✅ [{func.__name__}] 预热完毕！当前缓存池大小: {len(cache)}\n")
            initialized = True

        # ==========================================
        # 阶段二：常规的特征提取与动态更新
        # ==========================================

        # 1. 提取可哈希的 Key
        if isinstance(grid_thw_list, torch.Tensor):
            key = tuple(grid_thw_list.view(-1).tolist())
        elif isinstance(grid_thw_list, list):
            flat_list = grid_thw_list[0] if isinstance(grid_thw_list[0], list) else grid_thw_list
            key = tuple(flat_list)
        else:
            key = tuple(grid_thw_list)

        # 2. 查表或计算新 Shape (动态拓展)
        if key not in cache:
            print(f"⚠️ [{func.__name__}] 遇到未预热的边缘 Shape {key}，正在动态计算并追加缓存...")
            result = func(self, grid_thw_list, *args, **kwargs)

            if isinstance(result, tuple):
                cache[key] = tuple(r.detach().clone() for r in result)
            else:
                cache[key] = result.detach().clone()

        # 3. 返回缓存
        return cache[key]

    wrapper.clear_cache = lambda: cache.clear()
    wrapper.cache_size = lambda: len(cache)

    return wrapper
