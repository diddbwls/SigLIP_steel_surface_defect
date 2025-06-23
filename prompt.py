import openai
import json
from typing import List

# 최신 openai 라이브러리 사용
from openai import OpenAI

# OpenAI API 키 설정
client = OpenAI(
    api_key=""  # 여기에 본인의 API 키 입력
 # 여기에 본인의 API 키 입력
)


def generate_prompts(class_name: str, description: str = "", n: int = 5) -> List[str]:
    system_msg = (
        "You are a visual prompt generator for a vision-language model like CLIP. "
        "Given a class name and optionally a brief defect description, "
        "generate detailed, vivid, visual descriptions suitable as text prompts for classification."
    )

    user_msg = (
        f"Class: {class_name}\n"
        f"Description: {description}\n\n"
        f"Generate {n} distinct, high-quality visual prompts for this class, "
        f"written in natural English, each starting with 'A photo of ...'."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.7
    )

    output = response.choices[0].message.content
    prompts = [line.strip("- ").strip() for line in output.strip().split("\n") if line.strip()]
    return prompts[:n]


import os

def batch_generate_prompts(class_descriptions: dict, save_path: str = "/Users/yujin/Desktop/hm/prompt/generated_prompts.json"):
    # 중복 파일명 처리 로직
    base, ext = os.path.splitext(save_path)
    counter = 1
    final_save_path = save_path
    while os.path.exists(final_save_path):
        final_save_path = f"{base}_{counter}{ext}"
        counter += 1

    all_prompts = {}
    for cls, desc in class_descriptions.items():
        print(f"▶ Generating prompts for: {cls}")
        prompts = generate_prompts(cls, desc)
        all_prompts[cls] = prompts
        for p in prompts:
            print(f"  - {p}")
        print()

    with open(final_save_path, "w", encoding="utf-8") as f:
        json.dump(all_prompts, f, indent=2, ensure_ascii=False)
    print(f"✅ 저장 완료: {final_save_path}")


if __name__ == "__main__":
    class_descriptions = {
        "crazing": "Fine cracks appearing on the surface of steel, usually due to thermal stress.",
        "patches": "Irregular surface defects on steel that appear as discolored or rough patches.",
        "pitted_surface": "Corrosion-induced small holes or pits on the metal sheet surface.",
        "rolled-in_scale": "Oxide scale defects pressed into hot-rolled steel surfaces.",
        "scratches": "Linear abrasions or marks on the surface of steel sheets."
    }

    batch_generate_prompts(class_descriptions)