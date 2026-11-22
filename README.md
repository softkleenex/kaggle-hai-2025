# 🕵️‍♂️ HAI 2025 Deepfake Detection Challenge

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)

> **Competition:** [HAI 2025 Deepfake Detection Challenge](https://dacon.io/competitions/official/236628/overview/description) (Dacon)  
> **Team:** softkleenex (개인)

## ⚠️ 먼저 밝혀 둘 것 — 최종 순위는 규정 위반 방법으로 얻은 점수입니다

대회 규칙(`docs/constraints.md`)은 **평가 데이터(Test Set)를 학습에 쓰는 것**과 **여러 모델의 결과를 평균하는 앙상블**을 금지합니다.
그런데 이 저장소의 최종 제출(Private AUC 0.70403, 121위/1,356팀)은 바로 그 두 가지 — Test Set pseudo-labeling과 결과 앙상블 — 로 얻은 점수입니다.
규칙을 문서로 정리해 두고도 실험을 바꿀 때마다 그 기준에 비춰 보지 않은 것이 제 잘못입니다. 그래서 이 순위는 **성과로 내세우지 않습니다.**

이 저장소는 그 과정과 실패를 그대로 남겨 둔 기록입니다. 규정 안에서 확인된 사실은 아래뿐입니다.

| 규정 안에서 확인된 것 | 값 |
|---|---|
| 학습 데이터 | 공개 데이터 **140k Real and Fake Faces**(실제 사진 + StyleGAN 생성 얼굴, `scripts/download_data.py`) · 일부 실험에 Celeb-DF 추가 |
| 단일 모델의 대회 테스트 AUC | 약 **0.49 ~ 0.64** (학습 데이터 검증셋에서는 높았지만 대회 데이터에서는 무작위 수준에 가까움) |
| Celeb-DF를 더해 학습한 ConvNeXt 단일 모델 | 0.593 (`docs/SUBMISSION_HISTORY.csv`) |
| 배운 것 | 학습 데이터와 대회 데이터의 생성 방식·촬영 환경 차이는 데이터를 늘리는 것만으로 넘지 못했다 |

> 아래 표와 설명 가운데 **pseudo-labeling·TTA·앙상블·영상(LSTM) 모델**이 들어간 항목은 모두 규정 위반이거나 규정상 확인되지 않은 방법입니다. 재현하거나 참고할 때 이 점을 먼저 확인하세요.

---

## 📌 Overview
생성형 AI로 만든 딥페이크와 실제 영상을 구분하는 분류 모델을 만드는 대회입니다. 얼굴을 잘라 낸 이미지를 DINOv2(ViT)·EfficientNet·ConvNeXt로 학습했습니다.

## 🛠️ 시도한 방법 (규정 위반 여부 표시)

| 방법 | 규정 | 메모 |
|---|---|---|
| DINOv2 / EfficientNet / ConvNeXt 단일 모델 학습 | ✅ 허용 | 공개 데이터(140k Real and Fake Faces, Celeb-DF)로 학습 |
| 압축·모션블러 증강 (Blur-Breaker) | ✅ 허용 | 저화질 가짜가 진짜로 오분류되는 문제 대응. 단일 모델 점수는 크게 오르지 않음 |
| Test Set pseudo-labeling (상위/하위 5% 추출 → 10배 oversampling) | ❌ **위반** (Test Set 학습 사용) | Public 0.67 → 0.72 상승의 원인. 성과로 쓰지 않음 |
| 결과 앙상블 (DINO + EffB5 + ViT 가중 평균) | ❌ **위반** (결과 앙상블) | 최종 제출이 이 방식 |
| TTA (5가지 증강 결과 평균) | ⚠️ 확인 필요 | 대회 Q&A에서 모델 앙상블과 다르다는 언급이 있었으나 재확인하지 못함 |
| 영상 모델 (CNN+LSTM) | ❌ **위반** (시계열 학습 금지) | 앙상블에 결합한 제출도 점수 하락 |

---

## 📂 Project Structure
```text
/
├── checkpoints/       # 학습된 모델 가중치 (gitignore)
├── configs/           # 모델 및 실험 설정파일 (.yaml)
├── data/
│   ├── raw/           # 원본 데이터셋
│   ├── processed/     # 전처리된 얼굴 크롭 이미지
│   └── submission/    # 추론 결과 CSV
├── scripts/           # 자동화 및 유틸리티 스크립트
│   ├── run_kaggle.py  # Kaggle 학습 자동화
│   ├── ensemble.py    # 앙상블 스크립트
│   └── fix_submission.py # 제출 파일 형식 검증
├── src/
│   ├── data/          # Dataset & DataLoader (Custom Logic)
│   ├── models/        # DINO, EffNet, ConvNeXt 모델 정의
│   └── trainer/       # 학습 및 추론 루프
└── README.md          # 프로젝트 문서
```

---

## 📝 Experiments & Retrospective

| Ver | Method | Score (Public) | Insight |
|:---:|:---|:---:|:---|
| **V2** | Baseline Ensemble (DINO+ViT+EffB4) ❌ 결과 앙상블 | 0.6781 | 결과 앙상블은 규정상 금지. |
| **V7** | Pseudo Labeling (3% Injection) ❌ 위반 | 0.7038 | Test Set을 학습에 쓴 결과라 성과로 보지 않음. |
| **V10** | Pseudo (10x) + FewShot ❌ 위반 | 0.7215 | 위와 같음. |
| **V15** | Ensemble (DINO + ConvNeXt + EffB5) ❌ 결과 앙상블 | 0.7193 | ConvNeXt(Blur 증강)를 앙상블에 넣은 버전. 단일 모델로는 크게 오르지 않음. |
| **V17** | Ensemble + Video Model (CNN+LSTM) ❌ 결과 앙상블 | 0.7124 | 시계열 모델을 더해도 점수가 내려감. |

### 💡 What Worked / What I Learned
*   **(반성)** 점수를 가장 크게 올린 것은 Test Set pseudo-labeling이었지만, 이는 규정 위반이라 결과로 인정할 수 없다. 규정을 실험 설계의 첫 조건으로 두어야 했다.
*   **Self-Supervised Learning:** 적은 데이터 환경에서는 DINOv2 같은 SSL 모델이 CNN보다 강력한 Feature Extractor 역할을 수행함.

### ⚠️ Challenges & Future Work
*   **Video Context:** 프레임 단위 분류에 집중하느라 영상 전체의 시계열적(Temporal) 부자연스러움을 완벽히 활용하지 못함. (`VideoMAE` 등 3D-CNN 도입 필요)
*   **External Data:** Celeb-DF 등 외부 데이터를 추가했으나, 도메인 갭으로 인해 성능 향상에는 기여하지 못함.

---

### 💻 Installation & Usage

**1. Environment Setup**
```bash
git clone https://github.com/softkleenex/HAI-2025.git
cd HAI-2025
pip install -r requirements.txt
```

**2. Inference**
```bash
# Run inference using the best checkpoint
python src/trainer/inference.py --config configs/dino_large.yaml --checkpoint checkpoints/dino_best.pt --output submission.csv --tta
```

**3. Ensemble**
```bash
# Combine predictions
python scripts/ensemble.py --inputs sub1.csv sub2.csv --weights 0.5 0.5 --output final.csv
```

<!-- BLOG-URL:START -->

## Blog

- Blog note: [DACON HAI 2025 Deepfake Detection 회고](https://softkleenex.github.io/posts/dacon-hai-2025/)

<!-- BLOG-URL:END -->
