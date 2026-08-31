# -*- coding: utf-8 -*-
"""
PCOS Check — 비침습 문진 기반 PCOS 사전 스크리닝 웹앱
논문(조사-연구-구현) 기반 솔루션 · 베이스: Zigarelli et al., JMIR Formative Research 2022
"""
import json, os, glob, datetime
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier, Pool
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

st.set_page_config(page_title="PCOS Check", page_icon="🩺", layout="centered")
BURGUNDY = "#6B1F3B"

def _setup_korean_font():
    cands = glob.glob("/usr/share/fonts/truetype/nanum/*.ttf")
    cands += glob.glob("/usr/share/fonts/**/NanumGothic*.ttf", recursive=True)
    cands += ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
    for fp in cands:
        if os.path.exists(fp):
            try:
                fm.fontManager.addfont(fp)
                plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
                return
            except Exception:
                continue
_setup_korean_font()
plt.rcParams["axes.unicode_minus"] = False

@st.cache_resource
def load_model():
    m = CatBoostClassifier(); m.load_model("pcos_patient_model.cbm"); return m
@st.cache_data
def load_meta():
    return json.load(open("feature_meta.json", encoding="utf-8"))
@st.cache_data
def load_compare():
    try:
        return json.load(open("compare_stats.json", encoding="utf-8"))
    except Exception:
        return {}

model = load_model()
meta = load_meta()
CMP = load_compare()
FEATURES = meta["features"]
MED = {m["col"]: m["median"] for m in meta["meta"]}

KO = {
 "Skin darkening (Y/N)":"피부가 검게 변한 부위(흑색가시세포증)", "hair growth(Y/N)":"과도한 체모(다모증)",
 "Pimples(Y/N)":"여드름", "Hair loss(Y/N)":"탈모", "Weight gain(Y/N)":"체중 증가",
 "Cycle(R/I)":"생리주기 규칙성", "Cycle length(days)":"생리주기 길이",
 "Fast food (Y/N)":"패스트푸드 잦음", "Reg.Exercise(Y/N)":"규칙적 운동",
 "Age (yrs)":"나이", "BMI":"BMI", "Waist:Hip Ratio":"허리엉덩이비", "Waist(inch)":"허리둘레",
}

# ── 헤더 ──
st.markdown(f"<h1 style='color:{BURGUNDY};margin-bottom:0'>🩺 PCOS Check</h1>", unsafe_allow_html=True)
st.markdown("**초음파·혈액검사 없이, 간단한 문진만으로 다낭성난소증후군(PCOS) 위험도를 사전 확인**")
st.caption("베이스 논문: Zigarelli et al., JMIR Formative Research 2022 · CatBoost Patient 모델 (5-fold 정확도 81.5%)")
st.warning("⚠️ 본 서비스는 **의료 진단이 아닙니다.** 참고용 위험도 정보이며, 실제 진단은 반드시 의료기관에서 받으세요.")

with st.expander("ℹ️ 이 서비스는 어떻게 작동하나요?"):
    st.markdown(
        "- **데이터**: 인도 케랄라 10개 병원 541명의 공개 데이터로 학습\n"
        "- **모델**: CatBoost 분류기 (비침습 24개 문진 변수)\n"
        "- **설명**: 결과마다 SHAP으로 '어떤 항목이 위험도에 얼마나 영향을 줬는지' 표시\n"
        "- **성능**: 5-fold 교차검증 정확도 81.5%, AUC 0.87")

st.divider()
st.subheader("문진 입력")
st.caption("아는 항목만 입력하세요. 모르는 항목은 비워두거나 '모름'을 선택하면 평균값으로 처리됩니다.")

inputs = {}
tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ 증상", "2️⃣ 생리", "3️⃣ 신체·생활", "4️⃣ 추가(선택)"])

with tab1:
    st.caption("PCOS의 대표적인 겉으로 드러나는 증상입니다. 위험도 예측에 가장 큰 영향을 줍니다.")
    symptom_cols = ["Skin darkening (Y/N)", "hair growth(Y/N)", "Pimples(Y/N)", "Hair loss(Y/N)", "Weight gain(Y/N)"]
    hints = {
        "Skin darkening (Y/N)":"목·겨드랑이·사타구니 피부가 두껍고 검게 변함",
        "hair growth(Y/N)":"입가·턱·가슴 등에 남성형 털이 많이 남",
        "Pimples(Y/N)":"성인 여드름이 잦음",
        "Hair loss(Y/N)":"정수리 쪽 머리숱이 줄어듦",
        "Weight gain(Y/N)":"최근 특별한 이유 없이 체중이 늘어남",
    }
    c1, c2 = st.columns(2)
    for i, col in enumerate(symptom_cols):
        with (c1 if i % 2 == 0 else c2):
            inputs[col] = 1 if st.checkbox(KO[col], key=col, help=hints[col]) else 0

with tab2:
    cyc = st.radio("생리주기가 규칙적인가요?", ["규칙적", "불규칙", "잘 모름"], horizontal=True, key="cyc")
    inputs["Cycle(R/I)"] = 2 if cyc=="규칙적" else (4 if cyc=="불규칙" else MED["Cycle(R/I)"])
    know_cyclen = st.checkbox("생리주기 길이를 알고 있어요", key="kcl")
    if know_cyclen:
        inputs["Cycle length(days)"] = st.slider("평균 생리주기 길이 (일)", 1, 30, int(MED["Cycle length(days)"]), key="cyclen")
    else:
        inputs["Cycle length(days)"] = MED["Cycle length(days)"]

with tab3:
    c1, c2, c3 = st.columns(3)
    with c1:
        inputs["Age (yrs)"] = st.number_input("나이 (세)", 15, 60, 25)
    with c2:
        inputs["Weight (Kg)"] = st.number_input("체중 (kg)", 30, 150, int(MED["Weight (Kg)"]))
    with c3:
        inputs["Height(Cm)"] = st.number_input("신장 (cm)", 130, 200, int(MED["Height(Cm)"]))
    c1, c2 = st.columns(2)
    with c1:
        inputs["Fast food (Y/N)"] = 1 if st.checkbox("패스트푸드를 자주 먹는다", key="ff") else 0
    with c2:
        inputs["Reg.Exercise(Y/N)"] = 1 if st.checkbox("규칙적으로 운동한다", key="ex") else 0

with tab4:
    st.caption("허리·엉덩이 둘레는 줄자로 잰 값(inch). 모르면 비워두세요.")
    c1, c2 = st.columns(2)
    with c1:
        waist = st.number_input("허리둘레 (inch, 모르면 0)", 0, 60, 0, key="waist")
    with c2:
        hip = st.number_input("엉덩이둘레 (inch, 모르면 0)", 0, 60, 0, key="hip")
    preg = st.checkbox("현재 임신 중", key="pg")
    inputs["Pregnant(Y/N)"] = 1 if preg else 0

submitted = st.button("위험도 계산하기", type="primary", use_container_width=True)

if submitted:
    h_m = inputs["Height(Cm)"] / 100.0
    inputs["BMI"] = round(inputs["Weight (Kg)"] / (h_m*h_m), 1)
    inputs["Waist(inch)"] = waist if waist>0 else MED["Waist(inch)"]
    inputs["Hip(inch)"] = hip if hip>0 else MED["Hip(inch)"]
    inputs["Waist:Hip Ratio"] = round(inputs["Waist(inch)"]/max(inputs["Hip(inch)"],1), 2)
    for col in FEATURES:
        if col not in inputs:
            inputs[col] = MED[col]

    row = pd.DataFrame([[inputs[c] for c in FEATURES]], columns=FEATURES)
    proba = float(model.predict_proba(row)[0,1]); pct = round(proba*100,1)

    st.divider(); st.subheader("결과")
    if proba>=0.5: band,color,emoji="높음","#C0392B","🔴"
    elif proba>=0.3: band,color,emoji="중간","#E67E22","🟠"
    else: band,color,emoji="낮음","#27AE60","🟢"
    st.markdown(
        f"<div style='padding:1.2rem;border-radius:12px;background:{color}18;'>"
        f"<span style='font-size:1.1rem;'>PCOS 위험도</span><br>"
        f"<span style='font-size:2.6rem;font-weight:800;color:{color};'>{pct}%</span> "
        f"<span style='font-size:1.2rem;color:{color};'>{emoji} 위험 수준: {band}</span></div>",
        unsafe_allow_html=True)
    st.progress(min(proba,1.0))

    # ── 위험도별 맞춤 안내 ──
    st.markdown("##### 맞춤 안내")
    if band=="높음":
        st.error("증상과 지표가 PCOS 위험이 높은 편으로 나타났습니다. **산부인과 또는 내분비내과 진료를 권장**합니다. "
                 "초음파·혈액검사로 정확한 진단을 받아보세요.")
    elif band=="중간":
        st.warning("일부 지표에서 PCOS 가능성이 관찰됩니다. 생리 불순·체중 변화 등이 지속되면 **진료 상담을 고려**하세요. "
                   "규칙적 운동과 식습관 관리도 도움이 됩니다.")
    else:
        st.success("현재 입력 기준으로는 위험도가 낮게 나왔습니다. 다만 증상이 새로 생기거나 지속되면 언제든 진료를 받으세요. "
                   "이 결과는 진단이 아닌 참고용입니다.")

    # ── SHAP ──
    st.markdown("##### 이 결과에 영향을 준 요인 (SHAP)")
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Pool(row))
        contrib = sorted(zip(FEATURES, sv[0]), key=lambda x: abs(x[1]), reverse=True)[:8]
        labels = [KO.get(c,c) for c,_ in contrib]; vals=[v for _,v in contrib]
        colors = [BURGUNDY if v>0 else "#2E86DE" for v in vals]
        fig,ax = plt.subplots(figsize=(6.5,3.6))
        ax.barh(range(len(vals))[::-1], vals, color=colors)
        ax.set_yticks(range(len(vals))[::-1]); ax.set_yticklabels(labels, fontsize=10)
        ax.axvline(0, color="#888", lw=0.8)
        ax.set_xlabel("← 위험 낮춤    |    위험 높임 →", fontsize=9)
        for s in ["top","right"]: ax.spines[s].set_visible(False)
        ax.tick_params(length=0); plt.tight_layout()
        st.pyplot(fig)
        st.caption("빨강 = PCOS 위험을 높이는 방향, 파랑 = 낮추는 방향으로 작용한 요인")
    except Exception as e:
        st.info(f"SHAP 시각화를 표시할 수 없습니다: {e}")

    # ── 내 지표 vs 데이터셋 평균 ──
    if CMP:
        st.markdown("##### 내 지표 vs 참여자 평균")
        st.caption("학습 데이터(541명)의 PCOS군·비PCOS군 평균과 내 값을 비교합니다.")
        rows_html = "<table style='width:100%;border-collapse:collapse;font-size:0.9rem;'>"
        rows_html += "<tr style='background:#6B1F3B;color:white;'><th style='padding:6px;text-align:left;'>지표</th><th>내 값</th><th>PCOS군 평균</th><th>비PCOS군 평균</th></tr>"
        mymap = {"BMI":inputs.get("BMI"), "Age (yrs)":inputs.get("Age (yrs)"),
                 "Weight (Kg)":inputs.get("Weight (Kg)"), "Cycle length(days)":inputs.get("Cycle length(days)")}
        for key, info in CMP.items():
            myval = mymap.get(key, "-")
            bg = "#FBF7F1"
            rows_html += (f"<tr style='background:{bg};'><td style='padding:6px;'>{info['ko']}</td>"
                          f"<td style='text-align:center;font-weight:700;'>{myval}</td>"
                          f"<td style='text-align:center;'>{info['pcos']}</td>"
                          f"<td style='text-align:center;'>{info['nonpcos']}</td></tr>")
        rows_html += "</table>"
        st.markdown(rows_html, unsafe_allow_html=True)

    # ── 결과 요약 다운로드 ──
    st.markdown("##### 결과 저장")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    top3 = ", ".join([f"{KO.get(c,c)}({'+' if v>0 else '-'})" for c,v in contrib[:3]])
    summary = (f"PCOS Check 결과 요약\n생성일시: {now}\n"
               f"----------------------------------------\n"
               f"PCOS 위험도: {pct}% (위험 수준: {band})\n"
               f"주요 영향 요인 Top3: {top3}\n"
               f"BMI: {inputs.get('BMI')}\n"
               f"----------------------------------------\n"
               f"※ 본 결과는 의료 진단이 아니며 참고용입니다.\n"
               f"  위험도가 높거나 증상이 지속되면 의료기관 진료를 받으세요.\n"
               f"기반 논문: Zigarelli et al., JMIR Form Res 2022;6(3):e29967")
    st.download_button("📄 결과 요약 텍스트 저장", summary,
                       file_name=f"PCOS_Check_결과_{now[:10]}.txt", mime="text/plain",
                       use_container_width=True)

    st.divider()
    st.error("🏥 이 결과는 참고용입니다. 위험도가 높게 나오면, 그리고 낮아도 증상이 있으면 **반드시 산부인과·내분비내과 진료**를 받으세요.")

st.divider()
st.caption("PCOS Check · 논문(조사-연구-구현) 기반 솔루션 · 비진단 참고용 도구")
