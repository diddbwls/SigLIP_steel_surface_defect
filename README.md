# **Zero-shot 프롬프트 엔지니어링 기반 철강 표면 결함 분류**  
**Zero-shot Prompt Engineering based Steel Surface Defect Classification**  
*(SigLIP 기반 모델 + Beam Search + Ensemble을 통한 성능 향상)*

---

## ✅ 기존 Supervised Learning 기반 모델의 한계
- 새로운 결함 발생 시 라벨링 및 재학습이 필요  
- 빠른 대응이 어려움 → 실무 적용 시 확장성과 유연성 부족

---

## 🎯 본 프로젝트의 목적
- **사전학습 멀티모달 모델(SigLIP)** 기반  
- **도메인 지식을 활용한 프롬프트 조정만으로 분류 가능성 검증**  
- 라벨링/재학습 없이 **빠른 적용**이 가능한 분류 파이프라인 구축  
![prompt_goal](https://github.com/user-attachments/assets/2ed844b6-ed62-4e72-bb09-adc2e75ad8a6)

---

## 🔍 사용 모델
- **SigLIP (CLIP 계열 사전학습 멀티모달 모델)**  
- 이미지-텍스트 유사도 기반 zero-shot 분류  
![siglip](https://github.com/user-attachments/assets/4b3f0f31-b9e1-40a6-950c-3cf98e512b87)

---

## 🏗️ 아키텍처 구성
1. 프롬프트 후보군 작성  
2. **Beam Search (width=3)** 기반 프롬프트 최적화  
3. **Hard Voting Ensemble** 적용  
4. Best Prompt 선별  
5. 최종 SigLIP 추론 및 결과 도출  
![architecture](https://github.com/user-attachments/assets/3c281ae7-ef99-41f4-99d5-735c55988f56)

---

## 📂 사용 데이터
- **NEU 데이터셋**  
  - ISO 7788:2021 기준 5개 클래스 사용
    - *rolled-in scale*, *pitted surface*, *scratches*, *patches*, *crazing*  
- 출처: [Kaggle NEU Dataset](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)  
![neu1](https://github.com/user-attachments/assets/ac9141fa-5fff-4f25-859c-f77099f4f08b)  
![neu2](https://github.com/user-attachments/assets/8c400bd2-099e-46a0-a546-6d79db345b11)

---

## ✍️ 프롬프트 작성 예시
(도메인 지식을 기반으로 다양한 시도)  
![p1](https://github.com/user-attachments/assets/3541df2a-6c84-4bf7-ad03-2d1621ec657b)  
![p2](https://github.com/user-attachments/assets/b4dc6b16-6f61-4219-b763-abbf24c45418)  
![p3](https://github.com/user-attachments/assets/d1212746-62b3-4a22-81f0-ca820d3da341)  
![p4](https://github.com/user-attachments/assets/509f7130-a810-4c93-9478-6e7b46937a87)  
![p5](https://github.com/user-attachments/assets/e759491d-6e4f-4d2b-9f4f-ce35b66d97f9)  
![p6](https://github.com/user-attachments/assets/24470f60-7a1f-46fe-a01b-a4604099f40b)

---
 
## 📊 실험 결과

### ✅ Train Set (5 클래스, 총 1,500장)
![train_result](https://github.com/user-attachments/assets/ea7e801b-4dd8-4b71-a3a3-7e9fcb326f15)

### ✅ Test Set (5 클래스, 총 300장)
- **Zero-shot + Prompt Engineering** 만으로 도메인 특화 분류 성능 확보  
- Beam Search + Ensemble 설계만으로도 기존 supervised 모델과 경쟁 가능  
![test_result](https://github.com/user-attachments/assets/0e61f0dd-8b8f-4e85-ad95-b7840f4a6d2c)

---

## ⚠️ 한계점
- 일부 클래스 (*patches*)에서 분류 성능 저조
- test set에서 일부 결과가 잘 나오지 않았지만 추후 greedy search 등 최적 프롬프트를 찾는다면 성능 개선 가능성 존재
- 단일 프롬프트보단 **프롬프트 파일 집합** 기반 접근 필요  
- **계산량 증가**: 각 프롬프트 조합마다 추론 필요  
- **설명 가능성 부족**: 특정 프롬프트가 왜 효과적인지 설명 어려움

---

## 🔬 향후 연구 방향
- **Beam Search Width**, 프롬프트 개수 등 **하이퍼파라미터 최적화**  
- Prompt Set의 구조화 및 Greedy Search 기반 실험 추가
