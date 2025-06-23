zero-shot 프롬프트 엔지니어링 기반 철강 표면 결함 분류
zero-shop prompt engineering based steel surface defect classification
(SigLIP 기반 모델과 Beam Search, Ensemble을 통한 성능 향상 접근)

기존 라벨링, 학습 기반의 supervised-learning기반 모델의 한계
- 새로운 결함 등 이슈 발견시 재학습, 라벨링 데이터 확보 등의 이슈로 빠른 대응이 어려움

본 프로젝트의 목적
- 도메인 지식을 가지고 있는 상태로 빠르게 프롬프팅하여 결함 분류를 테스트
- ![image](https://github.com/user-attachments/assets/2ed844b6-ed62-4e72-bb09-adc2e75ad8a6)

사용 모델
CLIP기반의 사전학습 멀티모달 모델 SigLIP
![image](https://github.com/user-attachments/assets/4b3f0f31-b9e1-40a6-950c-3cf98e512b87)

본 프로젝트의 아키텍쳐
- 프롬프트 작성
- beam search (width=3) 및 SigLIP 수행
- ensemble(hard voting)
- best prompt 도출
- Final SigLIP 및 최종 분류 결과 도출
![image](https://github.com/user-attachments/assets/3c281ae7-ef99-41f4-99d5-735c55988f56)

사용 데이터
- NEU data set중, ISO 7788-2021 결함 분류에 속한 5가지 클래스(rolled-in scale, pitted_surface, scratches, patches, crazing)를 사용
- https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database
![image](https://github.com/user-attachments/assets/ac9141fa-5fff-4f25-859c-f77099f4f08b)
![image](https://github.com/user-attachments/assets/8c400bd2-099e-46a0-a546-6d79db345b11)

프롬프트 작성 예시
![image](https://github.com/user-attachments/assets/3541df2a-6c84-4bf7-ad03-2d1621ec657b)
![image](https://github.com/user-attachments/assets/b4dc6b16-6f61-4219-b763-abbf24c45418)
![image](https://github.com/user-attachments/assets/d1212746-62b3-4a22-81f0-ca820d3da341)
![image](https://github.com/user-attachments/assets/509f7130-a810-4c93-9478-6e7b46937a87)
![image](https://github.com/user-attachments/assets/e759491d-6e4f-4d2b-9f4f-ce35b66d97f9)
![image](https://github.com/user-attachments/assets/24470f60-7a1f-46fe-a01b-a4604099f40b)

결과 - train set(5개 클래스, 1,500장)
![image](https://github.com/user-attachments/assets/ea7e801b-4dd8-4b71-a3a3-7e9fcb326f15)

결과 - test set(5개 클래스, 300장)
- zero-shot 모델 + 프롬프트 엔지니어링만으로 도메인 특화 분류를 수행
- 학습 없이도 beam search, ensemble 설계를 통해 기존 supervised 방식 대비 경쟁력 확보
![image](https://github.com/user-attachments/assets/0e61f0dd-8b8f-4e85-ad95-b7840f4a6d2c)

한계
- test set에서 일부 결과가 잘 나오지 않았지만 추후 greedy search 등 최적 프롬프트를 찾는다면 성능 개선 가능성 존재
- 일부 class (patches)에서는 성능이 잘 나오지 않음 : 단일 프롬프트가 아닌 프롬프트 파일로 존재해야 의미가 있다는 한계
- 계산 비용(연산량) 증가 : 빔서치 과정에서 프롬프트 조합별로 SigLIP을 실행해야 함
- 설명 가능성 부족 : 왜 특정 프롬프트가 더 잘 작동하는지에 대한 설명력이 부족함


향후 연구
- beam search의 width, 프롬프트 개수 등 파라미터에 따라 결과가 달라질 수 있으므로 최적 하이퍼 파라미터를 찾는 실험이 필요


