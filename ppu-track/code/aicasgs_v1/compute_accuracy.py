#!/usr/bin/env python3
"""
适配 /opt/AICASGC 的类别命中率评估脚本。

用途：
- 读取 benchmark 生成的 result.json
- 从 /opt/AICASGC/data 中查找对应 question_id 的 image_classes
- 统计 prediction 是否命中任一类别词

说明：
- 当前仓库通常按前 20 / 100 个样本顺序评测，但这里仍按 question_id 建立映射，
  避免后续结果顺序变化时统计失真。
- 这是一个弱监督指标，只表示预测文本是否包含图像类别词，不代表真实 VQA 准确率。
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk


DEFAULT_RESULT_PATH = "/opt/AICASGC/result.json"
DEFAULT_DATASET_PATH = "/opt/AICASGC/data"


def _load_dataset(dataset_path: str) -> Dataset:
    dataset_obj = load_from_disk(dataset_path)
    if isinstance(dataset_obj, DatasetDict):
        if "validation" in dataset_obj:
            return dataset_obj["validation"]
        if "val" in dataset_obj:
            return dataset_obj["val"]
        first_split = next(iter(dataset_obj.keys()))
        return dataset_obj[first_split]
    return dataset_obj


def _build_id_to_classes(dataset: Dataset, question_ids: set[int]) -> dict[int, list[str]]:
    id_to_classes: dict[int, list[str]] = {}
    for item in dataset:
        qid = item.get("question_id")
        if qid in question_ids:
            id_to_classes[qid] = item.get("image_classes", []) or []
            if len(id_to_classes) == len(question_ids):
                break
    return id_to_classes


def compute_class_hit_rate(result_path: str, dataset_path: str) -> dict:
    result_file = Path(result_path)
    dataset_dir = Path(dataset_path)

    print(f"正在加载结果: {result_file}")
    with result_file.open("r", encoding="utf-8") as f:
        results = json.load(f)

    answers = results.get("answers", [])
    if not answers:
        print("错误: 结果中没有 answers 字段")
        return {}

    question_ids = {
        answer.get("question_id")
        for answer in answers
        if answer.get("question_id") is not None
    }
    print(f"共有 {len(answers)} 个答案，涉及 {len(question_ids)} 个 question_id")

    print(f"正在加载数据集: {dataset_dir}")
    dataset = _load_dataset(str(dataset_dir))
    print(f"数据集样本数: {len(dataset)}")

    id_to_classes = _build_id_to_classes(dataset, question_ids)
    missing_qids = sorted(question_ids - set(id_to_classes.keys()))

    hits = 0
    total = 0
    skipped = 0
    details = []

    for answer_item in answers:
        qid = answer_item.get("question_id")
        prediction = answer_item.get("prediction") or ""
        predicted_lower = prediction.lower()

        classes = id_to_classes.get(qid, [])
        if not classes:
            skipped += 1
            details.append(
                {
                    "question_id": qid,
                    "predicted": prediction[:100],
                    "classes": [],
                    "matched": None,
                    "hit": False,
                    "status": "missing_classes",
                }
            )
            continue

        hit = False
        matched_class = None
        for cls in classes:
            if cls and cls.lower() in predicted_lower:
                hit = True
                matched_class = cls
                break

        if hit:
            hits += 1
        total += 1
        details.append(
            {
                "question_id": qid,
                "predicted": prediction[:100],
                "classes": classes,
                "matched": matched_class,
                "hit": hit,
                "status": "evaluated",
            }
        )

    hit_rate = hits / total if total > 0 else 0.0

    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"结果文件: {result_file}")
    print(f"数据集路径: {dataset_dir}")
    print(f"答案总数: {len(answers)}")
    print(f"有效评估数: {total}")
    print(f"跳过数: {skipped}")
    print(f"命中数: {hits}")
    print(f"类别命中率: {hit_rate * 100:.2f}%")
    if missing_qids:
        preview = missing_qids[:10]
        suffix = " ..." if len(missing_qids) > 10 else ""
        print(f"未在数据集中找到的 question_id: {preview}{suffix}")
    print("=" * 60)

    print("\n命中的示例:")
    for item in [d for d in details if d["hit"]][:5]:
        print(f"✓ QID={item['question_id']}")
        print(f"   预测: {item['predicted'][:60]}...")
        print(f"   命中: {item['matched']}")
        print()

    print("未命中的示例:")
    for item in [d for d in details if d["status"] == "evaluated" and not d["hit"]][:5]:
        print(f"✗ QID={item['question_id']}")
        print(f"   预测: {item['predicted'][:60]}...")
        print(f"   类别: {item['classes'][:3]}")
        print()

    return {
        "result_path": str(result_file),
        "dataset_path": str(dataset_dir),
        "answers": len(answers),
        "total": total,
        "skipped": skipped,
        "hits": hits,
        "hit_rate": hit_rate,
        "missing_question_ids": missing_qids,
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(description="AICASGC 类别命中率评估")
    parser.add_argument(
        "--result",
        type=str,
        default=DEFAULT_RESULT_PATH,
        help=f"结果文件路径，默认: {DEFAULT_RESULT_PATH}",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help=f"数据集路径，默认: {DEFAULT_DATASET_PATH}",
    )
    args = parser.parse_args()

    compute_class_hit_rate(args.result, args.dataset_path)


if __name__ == "__main__":
    main()
