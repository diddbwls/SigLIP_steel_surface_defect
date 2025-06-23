import subprocess, atexit, torch, json
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from collections import defaultdict
import heapq
from copy import deepcopy

# caffeinate 실행 (절전 방지)
caffeinate_proc = subprocess.Popen(["caffeinate"])
atexit.register(caffeinate_proc.terminate)

def setup_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_prompt_file(prompt_path):
    with open(prompt_path, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_siglip(image_root: str, prompt_dict: dict, device: str):
    image_root = Path(image_root)
    summary = {}
    misclassified = []
    total, correct = 0, 0
    total_confidence_sum = 0.0

    model_name = "google/siglip-base-patch16-224"
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModelForZeroShotImageClassification.from_pretrained(model_name).to(device).eval()

    prompt_list, prompt_class_map = [], []
    for cls, prompts in prompt_dict.items():
        for p in prompts:
            prompt_list.append(p)
            prompt_class_map.append(cls)

    for cls in prompt_dict:
        folder = image_root / cls
        if not folder.exists():
            continue

        images = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png")) + \
                 list(folder.glob("*.JPG")) + list(folder.glob("*.JPEG")) + list(folder.glob("*.PNG"))

        class_total, class_correct, class_score_sum = len(images), 0, 0.0

        for img_path in images:
            try:
                image = Image.open(img_path).convert("RGB")
            except:
                continue

            inputs = processor(images=image, text=prompt_list, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits_per_image, dim=-1)
                pred_idx = probs.argmax().item()
                pred_class = prompt_class_map[pred_idx]
                confidence = probs[0][pred_idx].item()

            if pred_class == cls:
                class_correct += 1
            else:
                misclassified.append({
                    "file": str(img_path),
                    "predicted_class": pred_class,
                    "confidence": round(confidence, 4),
                    "true_class": cls
                })
            class_score_sum += confidence

        acc = class_correct / class_total if class_total else 0
        avg_score = class_score_sum / class_total if class_total else 0
        summary[cls] = {
            "accuracy": round(acc * 100, 2),
            "avg_confidence": round(avg_score * 100, 2),
            "correct": class_correct,
            "total": class_total
        }
        total += class_total
        correct += class_correct
        total_confidence_sum += class_score_sum

    summary["overall_accuracy"] = round((correct / total) * 100 if total else 0, 2)
    summary["overall_avg_confidence"] = round((total_confidence_sum / total) * 100 if total else 0, 2)
    summary["misclassified"] = misclassified
    return summary

def evaluate_single_prompt_combo(image_root: str, prompt_combo: dict, device: str):
    summary = evaluate_siglip(image_root, prompt_combo, device)
    return summary["overall_accuracy"], summary.get("overall_avg_confidence", 0)

def beam_search_prompt_selection(prompt_dict, image_root, device, beam_width=3):
    classes = list(prompt_dict.keys())
    beams = [({}, 0, 0)]  # (prompt_combo_dict, acc, avg_conf)

    for cls in classes:
        new_beams = []
        for combo, _, _ in beams:
            for prompt in prompt_dict[cls]:
                new_combo = deepcopy(combo)
                new_combo[cls] = [prompt]
                acc, avg_conf = evaluate_single_prompt_combo(image_root, new_combo, device)
                new_beams.append((new_combo, acc, avg_conf))
        beams = heapq.nlargest(beam_width, new_beams, key=lambda x: (x[1], x[2]))

    best_combo, best_acc, _ = max(beams, key=lambda x: (x[1], x[2]))
    return best_combo, best_acc

def save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument("--prompt", required=True, help="쉼표로 구분된 JSON 프롬프트 파일들")
    parser.add_argument("--prompt", default="prompt/generated_prompts.json", help="JSON 프롬프트 파일 경로")
    #parser.add_argument("--image_root", required=True, help="이미지 루트 디렉토리")
    parser.add_argument("--image_root", default="data/NEU-DET/train/images", help="이미지 루트 디렉토리")
    parser.add_argument("--output", default="siglip_result", help="결과 저장 디렉토리")
    parser.add_argument("--beam_width", type=int, default=3)
    args = parser.parse_args()

    device = setup_device()
    prompt_files = [p.strip() for p in args.prompt.split(",")]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for prompt_file in prompt_files:
        prompt_dict = load_prompt_file(prompt_file)
        best_prompt_combo, _ = beam_search_prompt_selection(prompt_dict, args.image_root, device, beam_width=args.beam_width)
        final_summary = evaluate_siglip(args.image_root, best_prompt_combo, device)

        # 저장
        stem = Path(prompt_file).stem
        save_json(best_prompt_combo, output_dir / f"best_prompt_{stem}.json")
        save_json(final_summary, output_dir / f"result_{stem}.json")