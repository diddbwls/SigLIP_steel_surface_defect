import json
import argparse
from collections import defaultdict
import numpy as np
from pathlib import Path

def load_result(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def summarize_results(result):
    print(f"\n📊 클래스별 정확도 및 평균 신뢰도")
    for cls, val in result.items():
        if cls in {"misclassified", "overall_accuracy", "overall_avg_confidence"}:
            continue
        acc = val["accuracy"]
        conf = val["avg_confidence"]
        print(f"  - {cls}: {acc:.2f}% (avg conf: {conf:.2f}%, {val['correct']}/{val['total']})")

    print(f"\n📈 전체 정확도: {result['overall_accuracy']:.2f}%")
    print(f"📈 전체 평균 신뢰도: {result['overall_avg_confidence']:.2f}%")

def analyze_misclassifications(result):
    misclassified = result.get("misclassified", [])
    if not misclassified:
        print("\n✅ 오인식 없음")
        return

    print(f"\n❌ 총 오인식 수: {len(misclassified)}")

    # 클래스별 오인식 분석
    mis_map = defaultdict(lambda: defaultdict(list))
    for entry in misclassified:
        true_cls = entry["true_class"]
        pred_cls = entry["predicted_class"]
        conf = entry["confidence"]
        mis_map[true_cls][pred_cls].append(conf)

    print("\n📉 오인식 상세:")
    for true_cls, preds in mis_map.items():
        print(f"\n🔸 {true_cls} →")
        for pred_cls, confs in preds.items():
            avg_conf = np.mean(confs)
            print(f"   → {pred_cls}: {len(confs)}개 (avg conf: {avg_conf:.2f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, help="result_*.json 파일 경로")
    args = parser.parse_args()

    result = load_result(args.result)
    summarize_results(result)
    analyze_misclassifications(result)