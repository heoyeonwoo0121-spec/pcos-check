# 🩺 PCOS Check

비침습 문진(24문항)만으로 다낭성난소증후군(PCOS) 위험도를 사전 확인하는 스크리닝 웹앱입니다.

- **베이스 논문**: Zigarelli et al., *Machine-Aided Self-diagnostic Prediction Models for Polycystic Ovary Syndrome*, JMIR Formative Research 2022
- **모델**: CatBoost (Patient / 비침습 24변수), 계층화 5-fold 교차검증 기준 정확도 약 81.5%
- **설명가능성**: SHAP으로 각 요인의 위험 기여도 표시

> ⚠️ 본 서비스는 의료 진단이 아니며, 참고용 위험도 정보만 제공합니다.

## 파일 구성
- `app.py` — Streamlit 웹앱
- `pcos_patient_model.cbm` — 학습된 CatBoost 모델
- `feature_meta.json` — 입력 변수 메타(범위·중앙값)
- `requirements.txt` — 의존성

## 로컬 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud 배포
1. 이 폴더의 파일을 GitHub 저장소에 업로드
2. https://share.streamlit.io 접속 → GitHub 연결
3. 저장소 선택, main 파일 경로 `app.py` 지정 → **Deploy**
