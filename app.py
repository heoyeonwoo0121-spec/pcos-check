# -*- coding: utf-8 -*-
"""
PCOS Check — 비침습 문진 기반 PCOS 사전 스크리닝 웹앱
논문(조사-연구-구현) 기반 솔루션 · 베이스: Zigarelli et al., JMIR Formative Research 2022

⚠ 본 서비스는 의료 진단이 아니며, 참고용 위험도 정보만 제공합니다.
"""
import json
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier, Pool
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import os

# ------------------------------------------------------------------
# 페이지 설정
# ------------------------------------------------------------------
st.set_page_config(page_title="PCOS Check", page_icon="🩺", layout="centered")

BURGUNDY = "#6B1F3B"

# 한글 폰트 (배포 환경에 있으면 사용)
for fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]:
    if os.path.exists(fp):
        try:
            fm.fontManager.addfont(fp)
            plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
        except Exception:
            pass
        break
plt.rcParams["axes.unicode_minus"] = False

# ------------------------------------------------------------------
# 모델 · 메타 로드 (캐시)
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    m = CatBoostClassifier()
    m.load_model("pcos_patient_model.cbm")
    return m

@st.cache_data
def load_meta():
    return json.load(open("feature_meta.json", encoding="utf-8"))

model = load_model()
meta = load_meta()
FEATURES = meta["features"]

# 사용자 친화 한글 라벨
KO = {
 "Age (yrs)":"나이 (세)", "Weight (Kg)":"체중 (kg)", "Height(Cm)":"신장 (cm)",
 "BMI":"BMI", "Blood Group":"혈액형 코드", "Pulse rate(bpm)":"맥박 (bpm)",
 "RR (breaths/min)":"호흡수 (회/분)", "Cycle(R/I)":"생리주기 규칙성",
 "Cycle length(days)":"생리주기 길이 (일)", "Marraige Status (Yrs)":"결혼 기간 (년)",
 "Pregnant(Y/N)":"임신 중이신가요?", "No. of aborptions":"유산 횟수",
 "Hip(inch)":"엉덩이둘레 (inch)", "Waist(inch)":"허리둘레 (inch)",
 "Waist:Hip Ratio":"허리:엉덩이 비율", "Weight gain(Y/N)":"최근 체중이 증가했나요?",
 "hair growth(Y/N)":"과도한 체모(다모증)가 있나요?", "Skin darkening (Y/N)":"피부가 검게 변한 부위(흑색가시세포증)가 있나요?",
 "Hair loss(Y/N)":"탈모가 있나요?", "Pimples(Y/N)":"여드름이 잦나요?",
 "Fast food (Y/N)":"패스트푸드를 자주 드시나요?", "Reg.Exercise(Y/N)":"규칙적으로 운동하나요?",
 "BP _Systolic (mmHg)":"수축기 혈압 (mmHg)", "BP _Diastolic (mmHg)":"이완기 혈압 (mmHg)",
}
BINARY = {m["col"] for m in meta["meta"] if m["is_binary_like"]}
MED = {m["col"]: m["median"] for m in meta["meta"]}
MINV = {m["col"]: m["min"] for m in meta["meta"]}
MAXV = {m["col"]: m["max"] for m in meta["meta"]}

# ------------------------------------------------------------------
# 헤더
# ------------------------------------------------------------------
st.markdown(f"<h1 style='color:{BURGUNDY};margin-bottom:0'>🩺 PCOS Check</h1>",
            unsafe_allow_html=True)
st.markdown("**초음파·혈액검사 없이, 비침습 문진만으로 다낭성난소증후군(PCOS) 위험도를 사전 확인**")
st.caption("베이스 논문: Zigarelli et al., JMIR Formative Research 2022 · CatBoost Patient 모델")

st.warning("⚠️ 본 서비스는 **의료 진단이 아닙니다.** 참고용 위험도 정보이며, "
           "실제 진단은 반드시 의료기관에서 받으세요.")

st.divider()

# ------------------------------------------------------------------
# 문진 폼
# ------------------------------------------------------------------
st.subheader("문진 입력")
st.caption("아래 항목을 입력하면 PCOS 위험도(%)와 그 근거를 계산합니다.")

inputs = {}

with st.form("survey"):
    st.markdown("##### 1. 증상 (해당하면 체크)")
    symptom_cols = ["Skin darkening (Y/N)", "hair growth(Y/N)", "Pimples(Y/N)",
                    "Hair loss(Y/N)", "Weight gain(Y/N)"]
    c1, c2 = st.columns(2)
    for i, col in enumerate(symptom_cols):
        with (c1 if i % 2 == 0 else c2):
            inputs[col] = 1 if st.checkbox(KO[col], key=col) else 0

    st.markdown("##### 2. 생리 관련")
    c1, c2 = st.columns(2)
    with c1:
        cyc = st.radio("생리주기가 규칙적인가요?", ["규칙적", "불규칙"], key="cyc")
        # 데이터 코드: 2=규칙 계열, 4/5=불규칙 계열 → 규칙2, 불규칙4로 매핑
        inputs["Cycle(R/I)"] = 2 if cyc == "규칙적" else 4
    with c2:
        inputs["Cycle length(days)"] = st.number_input(
            "평균 생리주기 길이 (일)", min_value=0, max_value=30,
            value=int(MED["Cycle length(days)"]), key="cyclen")

    st.markdown("##### 3. 신체 계측")
    c1, c2, c3 = st.columns(3)
    with c1:
        inputs["Age (yrs)"] = st.number_input("나이 (세)", 15, 60,
            int(MED["Age (yrs)"]))
        inputs["Weight (Kg)"] = st.number_input("체중 (kg)", 30, 150,
            int(MED["Weight (Kg)"]))
    with c2:
        inputs["Height(Cm)"] = st.number_input("신장 (cm)", 130, 200,
            int(MED["Height(Cm)"]))
        inputs["Waist(inch)"] = st.number_input("허리둘레 (inch)", 20, 55,
            int(MED["Waist(inch)"]))
    with c3:
        inputs["Hip(inch)"] = st.number_input("엉덩이둘레 (inch)", 25, 60,
            int(MED["Hip(inch)"]))
        inputs["Pulse rate(bpm)"] = st.number_input("맥박 (bpm)", 40, 120,
            int(MED["Pulse rate(bpm)"]))

    st.markdown("##### 4. 생활습관")
    c1, c2 = st.columns(2)
    with c1:
        inputs["Fast food (Y/N)"] = 1 if st.checkbox(
            "패스트푸드를 자주 먹는다", key="ff") else 0
    with c2:
        inputs["Reg.Exercise(Y/N)"] = 1 if st.checkbox(
            "규칙적으로 운동한다", key="ex") else 0

    st.markdown("##### 5. 기타")
    c1, c2 = st.columns(2)
    with c1:
        inputs["Pregnant(Y/N)"] = 1 if st.checkbox("현재 임신 중", key="pg") else 0
        inputs["Marraige Status (Yrs)"] = st.number_input("결혼 기간 (년, 없으면 0)",
            0, 40, int(MED["Marraige Status (Yrs)"]))
    with c2:
        inputs["No. of aborptions"] = st.number_input("유산 횟수", 0, 10,
            int(MED["No. of aborptions"]))
        bp = st.text_input("혈압 (예: 120/80, 모르면 비워두세요)", "")

    submitted = st.form_submit_button("위험도 계산하기", type="primary",
                                      use_container_width=True)

# ------------------------------------------------------------------
# 예측
# ------------------------------------------------------------------
if submitted:
    # 혈압 파싱
    sys_v, dia_v = MED["BP _Systolic (mmHg)"], MED["BP _Diastolic (mmHg)"]
    if bp.strip():
        try:
            s, d = bp.split("/")
            sys_v, dia_v = float(s), float(d)
        except Exception:
            st.info("혈압 형식을 인식하지 못해 평균값으로 처리했습니다.")
    inputs["BP _Systolic (mmHg)"] = sys_v
    inputs["BP _Diastolic (mmHg)"] = dia_v

    # 파생/미입력 변수 자동 채움
    h_m = inputs["Height(Cm)"] / 100.0
    inputs["BMI"] = round(inputs["Weight (Kg)"] / (h_m * h_m), 1)
    whr = inputs["Waist(inch)"] / max(inputs["Hip(inch)"], 1)
    inputs["Waist:Hip Ratio"] = round(whr, 2)
    # 문진에 없는 나머지는 데이터 중앙값으로
    for col in FEATURES:
        if col not in inputs:
            inputs[col] = MED[col]

    # 예측
    row = pd.DataFrame([[inputs[c] for c in FEATURES]], columns=FEATURES)
    proba = float(model.predict_proba(row)[0, 1])
    pct = round(proba * 100, 1)

    st.divider()
    st.subheader("결과")

    # 위험도 표시
    if proba >= 0.5:
        band, color = "높음", "#C0392B"
    elif proba >= 0.3:
        band, color = "중간", "#E67E22"
    else:
        band, color = "낮음", "#27AE60"

    st.markdown(
        f"<div style='padding:1.2rem;border-radius:12px;background:{color}18;'>"
        f"<span style='font-size:1.1rem;'>PCOS 위험도</span><br>"
        f"<span style='font-size:2.6rem;font-weight:800;color:{color};'>{pct}%</span> "
        f"<span style='font-size:1.2rem;color:{color};'>(위험 수준: {band})</span>"
        f"</div>", unsafe_allow_html=True)
    st.progress(min(proba, 1.0))

    # SHAP 기여도
    st.markdown("##### 이 결과에 영향을 준 요인 (SHAP)")
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Pool(row))
        contrib = sorted(zip(FEATURES, sv[0]), key=lambda x: abs(x[1]), reverse=True)[:8]
        labels = [KO.get(c, c) for c, _ in contrib]
        vals = [v for _, v in contrib]
        colors = [BURGUNDY if v > 0 else "#2E86DE" for v in vals]

        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        ax.barh(range(len(vals))[::-1], vals, color=colors)
        ax.set_yticks(range(len(vals))[::-1])
        ax.set_yticklabels(labels, fontsize=10)
        ax.axvline(0, color="#888", lw=0.8)
        ax.set_xlabel("← 위험 낮춤    |    위험 높임 →", fontsize=9)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0)
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("빨강 = PCOS 위험을 높이는 방향, 파랑 = 낮추는 방향으로 작용한 요인")
    except Exception as e:
        st.info(f"SHAP 시각화를 표시할 수 없습니다: {e}")

    st.divider()
    st.error("🏥 이 결과는 참고용입니다. 위험도가 낮게 나와도 증상이 있으면, "
             "그리고 높게 나오면 **반드시 산부인과·내분비내과 진료**를 받으세요.")

st.divider()
st.caption("PCOS Check · 논문(조사-연구-구현) 기반 솔루션 · 비진단 참고용 도구")
