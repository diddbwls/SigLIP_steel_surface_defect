import json
from pathlib import Path
import argparse
from collections import defaultdict

# 클래스별 선택된 파일 인덱스
class_source_map = {
    "pitted_surface": 3,
    "patches": 0,
    "rolled-in_scale": 1,
    "scratches": 2,
    "crazing": 2,
}

def load_results(base_dir):
    result_data = {}
    for idx in set(class_source_map.values()):
        file_path = Path(base_dir) / f"result_generated_prompts_{idx}.json"
        with open(file_path, 'r', encoding='utf-8') as f:
            result_data[idx] = json.load(f)
    return result_data

def ensemble_results(result_data):
    ensemble_summary = {}
    total_correct = total_images = total_conf_sum = 0
    misclassified = []

    for cls, src_idx in class_source_map.items():
        cls_data = result_data[src_idx][cls]
        ensemble_summary[cls] = cls_data

        total_correct += cls_data["correct"]
        total_images += cls_data["total"]
        total_conf_sum += cls_data["avg_confidence"] * cls_data["total"] / 100

        for m in result_data[src_idx].get("misclassified", []):
            if m["true_class"] == cls:
                misclassified.append(m)

    # 전체 요약
    ensemble_summary["overall_accuracy"] = round((total_correct / total_images) * 100, 2)
    ensemble_summary["overall_avg_confidence"] = round((total_conf_sum / total_images) * 100, 2)
    ensemble_summary["misclassified"] = misclassified
    return ensemble_summary

def save_ensemble(summary, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✅ 앙상블 결과 저장 완료: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="기존 result_*.json 파일들이 있는 디렉토리")
    parser.add_argument("--output", type=str, default="ensemble_result.json", help="저장할 앙상블 결과 파일 경로")
    args = parser.parse_args()

    result_data = load_results(args.base_dir)
    ensemble_summary = ensemble_results(result_data)
    save_ensemble(ensemble_summary, Path(args.output))