# -*- coding: utf-8 -*-
"""
PCOS Check — 비침습 문진 기반 PCOS 사전 스크리닝 웹 서비스
논문(조사-연구-구현) 기반 솔루션 · 베이스: Zigarelli et al., JMIR Formative Research 2022
의학 정보 출처: WHO PCOS Fact Sheet, Mayo Clinic
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

st.set_page_config(page_title="PCOS Check", page_icon="🩺", layout="centered",
                   initial_sidebar_state="expanded")
BURGUNDY = "#6B1F3B"; GREEN = "#4A7C59"; BLUE = "#2E86DE"

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
    try: return json.load(open("compare_stats.json", encoding="utf-8"))
    except Exception: return {}

model = load_model(); meta = load_meta(); CMP = load_compare()
FEATURES = meta["features"]
MED = {m["col"]: m["median"] for m in meta["meta"]}

KO = {
 "Skin darkening (Y/N)":"피부가 검게 변한 부위(흑색가시세포증)", "hair growth(Y/N)":"과도한 체모(다모증)",
 "Pimples(Y/N)":"여드름", "Hair loss(Y/N)":"탈모", "Weight gain(Y/N)":"체중 증가",
 "Cycle(R/I)":"생리주기 규칙성", "Cycle length(days)":"생리주기 길이",
 "Fast food (Y/N)":"패스트푸드 잦음", "Reg.Exercise(Y/N)":"규칙적 운동",
 "Age (yrs)":"나이", "BMI":"BMI", "Waist:Hip Ratio":"허리엉덩이비", "Waist(inch)":"허리둘레",
}

# ── 사이드바: 큼직한 버튼형 메뉴 ──
st.markdown("""<style>
section[data-testid="stSidebar"] button {
  width:100%; text-align:left; font-size:1.05rem !important;
  padding:0.7rem 1rem !important; margin-bottom:0.3rem; border-radius:10px;
}
div.block-container{padding-top:2rem;}
</style>""", unsafe_allow_html=True)

if "page" not in st.session_state: st.session_state.page = "home"
if "step" not in st.session_state: st.session_state.step = 1
if "answers" not in st.session_state: st.session_state.answers = {}

with st.sidebar:
    st.markdown(f"<h2 style='color:{BURGUNDY};margin-bottom:0;'>🩺 PCOS Check</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#666;font-size:0.95rem;'>비침습 문진 기반<br>PCOS 사전 스크리닝</p>", unsafe_allow_html=True)
    st.write("")
    menu = [("home","🏠  홈"), ("diag","📋  자가진단"), ("learn","📚  PCOS 알아보기"), ("model","🔬  이 모델은?")]
    for key,label in menu:
        typ = "primary" if st.session_state.page==key else "secondary"
        if st.button(label, key=f"nav_{key}", type=typ, use_container_width=True):
            st.session_state.page = key
            if key=="diag": st.session_state.step = 1
            st.rerun()
    st.divider()
    st.markdown("<p style='font-size:0.85rem;color:#999;'>⚠️ 의료 진단이 아닌 참고용 도구입니다.<br>출처: WHO · Zigarelli et al. 2022</p>", unsafe_allow_html=True)

page = st.session_state.page

# ============================================================
# 🏠 홈
# ============================================================
if page == "home":
    st.markdown(f"<h1 style='color:{BURGUNDY};margin-bottom:0'>🩺 PCOS Check</h1>", unsafe_allow_html=True)
    st.markdown("#### 검사 없이, 문진만으로 다낭성난소증후군(PCOS) 위험도를 미리 확인하세요")
    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.metric("모델 정확도", "81.5%", "5-fold")
    c2.metric("AUC", "0.87", "판별력")
    c3.metric("학습 데이터", "541명", "10개 병원")
    st.write("")
    st.markdown("#### 왜 필요한가요?")
    st.markdown(f"<div style='background:#FBF7F1;padding:1rem;border-radius:10px;border-left:4px solid {BURGUNDY};margin-bottom:0.6rem;'>"
                f"<b style='color:{BURGUNDY};'>① 흔하지만 놓치기 쉬움</b><br>"
                "가임기 여성의 10~13%가 PCOS를 가지며, 최대 70%가 자신이 PCOS인지 모릅니다. <span style='font-size:0.8rem;color:#888;'>(WHO)</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='background:#FBF7F1;padding:1rem;border-radius:10px;border-left:4px solid {BURGUNDY};margin-bottom:0.6rem;'>"
                f"<b style='color:{BURGUNDY};'>② 검사 문턱이 높음</b><br>"
                "진단에는 초음파와 혈액검사가 필요해 병원 방문·비용·시간 부담이 큽니다.</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='background:#FBF7F1;padding:1rem;border-radius:10px;border-left:4px solid {GREEN};margin-bottom:0.6rem;'>"
                f"<b style='color:{GREEN};'>③ 집에서 미리 확인</b><br>"
                "문진 24문항만으로 위험도를 계산하고, 어떤 요인이 영향을 줬는지까지 설명해 드립니다.</div>", unsafe_allow_html=True)
    st.write("")
    if st.button("📋  자가진단 시작하기", type="primary", use_container_width=True):
        st.session_state.page = "diag"; st.session_state.step = 1; st.rerun()

# ============================================================
# 📚 PCOS 알아보기
# ============================================================
elif page == "learn":
    st.markdown(f"<h1 style='color:{BURGUNDY};'>📚 PCOS 알아보기</h1>", unsafe_allow_html=True)
    st.caption("각 항목을 눌러 자세한 설명을 확인하세요. (출처: WHO, Mayo Clinic)")
    with st.expander("❓ PCOS(다낭성난소증후군)란?"):
        st.write("안드로겐(남성호르몬)이 정상보다 높아져 생기는 흔한 호르몬 질환입니다. 불규칙한 생리, 배란 이상, "
                 "불임, 과도한 털, 여드름 등을 유발할 수 있습니다. 가임기 여성의 약 10~13%가 해당하며, "
                 "전 세계적으로 무배란성 불임의 가장 흔한 원인입니다.")
        st.caption("출처: WHO Fact Sheet")
    with st.expander("🔍 주요 증상"):
        st.markdown("- **불규칙한 생리주기**: 생리를 건너뛰거나 주기가 들쭉날쭉함\n"
                    "- **다모증**: 입가·턱·가슴 등에 남성형 털이 과도하게 남\n"
                    "- **여드름·지성 피부**: 성인기에도 잦은 여드름\n"
                    "- **흑색가시세포증**: 목·겨드랑이 피부가 두껍고 검게 변함(인슐린 저항성 신호)\n"
                    "- **체중 증가·탈모**: 특히 복부 비만, 정수리 탈모")
        st.caption("출처: Mayo Clinic, WHO")
    with st.expander("🧬 원인"):
        st.write("정확한 원인은 완전히 밝혀지지 않았으나, **인슐린 저항성**과 고인슐린혈증이 안드로겐 증가를 "
                 "유발하는 핵심 기전입니다. 유전(가족력)과 생활습관(비만·운동 부족·식습관)도 영향을 줍니다.")
        st.caption("출처: 의학 리뷰 PMC6489978")
    with st.expander("⚠️ 방치 시 위험"):
        st.write("가임기 이후에도 지속되는 만성 대사질환입니다. 관리하지 않으면 인슐린 저항성, 2형 당뇨병, "
                 "비만, 심혈관 질환, 자궁내막 증식증 위험이 높아지고, 우울·불안과의 연관성도 보고됩니다.")
        st.caption("출처: WHO, PMC6489978")
    with st.expander("💪 관리 방법"):
        st.markdown("- **생활습관 개선이 1순위**: 규칙적 운동 + 균형 잡힌 식사가 인슐린 감수성을 높임\n"
                    "- **체중 관리**: 과체중이면 체중 감량이 호르몬·생리주기 개선에 도움\n"
                    "- **의료적 치료**: 필요 시 경구피임약·메트포르민 등 — 반드시 의사와 상담\n"
                    "- 생활습관+약물 병행으로 85~90% 환자가 개선을 보인 연구도 있습니다.")
        st.caption("출처: PMC10042521, PMC9440853")
    st.warning("이 정보는 교육용이며 개인 진단을 대체하지 않습니다.")

# ============================================================
# 🔬 이 모델은?
# ============================================================
elif page == "model":
    st.markdown(f"<h1 style='color:{BURGUNDY};'>🔬 이 모델은 어떻게 만들어졌나요?</h1>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("정확도","81.5%"); c2.metric("AUC","0.87"); c3.metric("F1","0.72"); c4.metric("데이터","541명")
    with st.expander("📂 데이터", expanded=True):
        st.write("인도 케랄라 10개 병원 541명의 공개 데이터(Kaggle)를 사용했습니다. 음성 364 : 양성 177명이며, "
                 "44개 변수 중 검사가 필요 없는 비침습 24개 변수만 사용했습니다.")
    with st.expander("🤖 모델 — CatBoost"):
        st.write("범주형 변수 처리에 강한 그래디언트 부스팅 분류기입니다. 베이스 논문이 채택한 모델과 동일하며, "
                 "8개 모델 비교에서 비침습 조건 상위권 성능을 보였습니다.")
    with st.expander("📏 검증 — 계층화 5-fold 교차검증"):
        st.write("양성:음성 비율을 유지한 채 5조각으로 교차검증했습니다. 정확도 81.5%, AUC 0.87, F1 0.72, "
                 "fold별 표준편차가 작아 안정적입니다.")
    with st.expander("💡 설명가능 AI — SHAP"):
        st.write("모든 예측에 SHAP으로 변수 기여도를 계산합니다. 상위 변수는 흑색가시세포증·다모증·불규칙 생리·"
                 "체중 증가 등으로 원논문 주요 변수와 일치합니다.")
    with st.expander("⚖️ 불균형 처리 & 한계"):
        st.write("불균형은 class_weight로 보정(SMOTE와 비교 후 채택)했습니다. 데이터가 인도 케랄라 단일 지역이라 "
                 "일반화에 한계가 있으며, 진단이 아닌 사전 스크리닝 용도입니다.")
    st.info("기반 논문: Zigarelli, A., Jia, Z., & Lee, H. (2022). JMIR Formative Research, 6(3), e29967.")

# ============================================================
# 📋 자가진단 — 단계별 진행 방식
# ============================================================
elif page == "diag":
    st.markdown(f"<h1 style='color:{BURGUNDY};'>📋 자가진단</h1>", unsafe_allow_html=True)
    A = st.session_state.answers
    TOTAL = 4
    step = st.session_state.step

    # 진행 표시
    st.progress((step-1)/TOTAL if step<=TOTAL else 1.0)
    st.caption(f"{min(step,TOTAL)} / {TOTAL} 단계")

    why = {
        "Skin darkening (Y/N)":"목·겨드랑이 피부가 검고 두꺼워지는 흑색가시세포증은 인슐린 저항성의 신호로 PCOS와 강하게 연관됩니다.",
        "hair growth(Y/N)":"안드로겐 과다로 남성형 털이 늘어나는 다모증은 PCOS의 대표 징후입니다.",
        "Pimples(Y/N)":"안드로겐 증가는 피지 분비를 늘려 성인 여드름을 유발할 수 있습니다.",
        "Hair loss(Y/N)":"안드로겐성 탈모(정수리 중심)도 호르몬 불균형의 신호일 수 있습니다.",
        "Weight gain(Y/N)":"인슐린 저항성은 특히 복부 체중 증가와 관련되며 PCOS 위험을 높입니다.",
    }

    # ── STEP 1: 증상 (세로 1열) ──
    if step == 1:
        st.subheader("1단계 · 증상")
        st.caption("해당하는 증상을 모두 선택하세요. 위험도 예측에 가장 큰 영향을 줍니다.")
        symptom_cols = ["Skin darkening (Y/N)", "hair growth(Y/N)", "Pimples(Y/N)", "Hair loss(Y/N)", "Weight gain(Y/N)"]
        for col in symptom_cols:
            A[col] = 1 if st.checkbox(KO[col], value=bool(A.get(col,0)), key=f"s_{col}", help=why[col]) else 0
        st.write("")
        if st.button("다음 →", type="primary", use_container_width=True):
            st.session_state.step = 2; st.rerun()

    # ── STEP 2: 생리 ──
    elif step == 2:
        st.subheader("2단계 · 생리 관련")
        cyc = st.radio("생리주기가 규칙적인가요?", ["규칙적","불규칙","잘 모름"],
                       index=["규칙적","불규칙","잘 모름"].index(A.get("_cyc","규칙적")),
                       key="r_cyc", help="불규칙한 생리는 배란 장애를 반영하며 PCOS의 핵심 축입니다.")
        A["_cyc"]=cyc
        A["Cycle(R/I)"] = 2 if cyc=="규칙적" else (4 if cyc=="불규칙" else MED["Cycle(R/I)"])
        know = st.checkbox("생리주기 길이를 알고 있어요", value=bool(A.get("_know_len",False)), key="c_len")
        A["_know_len"]=know
        if know:
            A["Cycle length(days)"] = st.slider("평균 생리주기 길이 (일)", 1, 30, int(A.get("Cycle length(days)",MED["Cycle length(days)"])), key="sl_len")
        else:
            A["Cycle length(days)"] = MED["Cycle length(days)"]
        st.write("")
        b1,b2 = st.columns([1,2])
        if b1.button("← 이전", use_container_width=True):
            st.session_state.step = 1; st.rerun()
        if b2.button("다음 →", type="primary", use_container_width=True):
            st.session_state.step = 3; st.rerun()

    # ── STEP 3: 신체·생활 ──
    elif step == 3:
        st.subheader("3단계 · 신체·생활")
        A["Age (yrs)"] = st.number_input("나이 (세)", 15, 60, int(A.get("Age (yrs)",25)))
        A["Weight (Kg)"] = st.number_input("체중 (kg)", 30, 150, int(A.get("Weight (Kg)",MED["Weight (Kg)"])))
        A["Height(Cm)"] = st.number_input("신장 (cm)", 130, 200, int(A.get("Height(Cm)",MED["Height(Cm)"])))
        A["Fast food (Y/N)"] = 1 if st.checkbox("패스트푸드를 자주 먹는다", value=bool(A.get("Fast food (Y/N)",0)), key="c_ff",
            help="고열량·정제탄수 식이는 인슐린 저항성을 악화시킬 수 있습니다.") else 0
        A["Reg.Exercise(Y/N)"] = 1 if st.checkbox("규칙적으로 운동한다", value=bool(A.get("Reg.Exercise(Y/N)",0)), key="c_ex",
            help="규칙적 운동은 인슐린 감수성을 높여 PCOS 관리에 도움이 됩니다.") else 0
        st.write("")
        b1,b2 = st.columns([1,2])
        if b1.button("← 이전", use_container_width=True):
            st.session_state.step = 2; st.rerun()
        if b2.button("다음 →", type="primary", use_container_width=True):
            st.session_state.step = 4; st.rerun()

    # ── STEP 4: 추가(선택) + 계산 ──
    elif step == 4:
        st.subheader("4단계 · 추가 정보 (선택)")
        st.caption("허리·엉덩이 둘레는 줄자로 잰 값(inch). 모르면 비워두세요.")
        A["_waist"] = st.number_input("허리둘레 (inch, 모르면 0)", 0, 60, int(A.get("_waist",0)), key="n_w")
        A["_hip"] = st.number_input("엉덩이둘레 (inch, 모르면 0)", 0, 60, int(A.get("_hip",0)), key="n_h")
        A["Pregnant(Y/N)"] = 1 if st.checkbox("현재 임신 중", value=bool(A.get("Pregnant(Y/N)",0)), key="c_pg") else 0
        st.write("")
        b1,b2 = st.columns([1,2])
        if b1.button("← 이전", use_container_width=True):
            st.session_state.step = 3; st.rerun()
        if b2.button("✓ 위험도 계산하기", type="primary", use_container_width=True):
            st.session_state.step = 5; st.rerun()

    # ── STEP 5: 결과 ──
    elif step == 5:
        inputs = dict(A)
        h_m = inputs["Height(Cm)"]/100.0
        inputs["BMI"] = round(inputs["Weight (Kg)"]/(h_m*h_m),1)
        inputs["Waist(inch)"] = A["_waist"] if A.get("_waist",0)>0 else MED["Waist(inch)"]
        inputs["Hip(inch)"] = A["_hip"] if A.get("_hip",0)>0 else MED["Hip(inch)"]
        inputs["Waist:Hip Ratio"] = round(inputs["Waist(inch)"]/max(inputs["Hip(inch)"],1),2)
        for col in FEATURES:
            if col not in inputs: inputs[col] = MED[col]
        row = pd.DataFrame([[inputs[c] for c in FEATURES]], columns=FEATURES)
        proba = float(model.predict_proba(row)[0,1]); pct = round(proba*100,1)

        st.progress(1.0); st.caption("완료")
        st.subheader("📊 결과")
        if proba>=0.5: band,color,emoji="높음","#C0392B","🔴"
        elif proba>=0.3: band,color,emoji="중간","#E67E22","🟠"
        else: band,color,emoji="낮음","#27AE60","🟢"

        st.markdown(
            f"<div style='padding:1.3rem;border-radius:12px;background:{color}18;text-align:center;'>"
            f"<span style='font-size:1rem;'>PCOS 위험도</span><br>"
            f"<span style='font-size:3rem;font-weight:800;color:{color};'>{pct}%</span><br>"
            f"<span style='font-size:1.1rem;color:{color};'>{emoji} 위험 수준: {band}</span></div>",
            unsafe_allow_html=True)
        st.progress(min(proba,1.0))

        bmi=inputs.get("BMI"); bmi_cat = "저체중" if bmi<18.5 else "정상" if bmi<23 else "과체중" if bmi<25 else "비만"
        st.metric("BMI", f"{bmi}", bmi_cat)
        st.caption("BMI 23 이상 과체중, 25 이상 비만(아시아 기준). 하위군 분석에서 비만군 PCOS 유병률이 마름군의 약 1.6배였습니다.")

        st.markdown("##### 💡 맞춤 안내")
        if band=="높음":
            st.error("PCOS 위험이 높은 편입니다. **산부인과 또는 내분비내과 진료를 권장**합니다. 초음파·혈액검사로 정확한 진단을 받아보세요.")
        elif band=="중간":
            st.warning("일부 지표에서 PCOS 가능성이 관찰됩니다. 생리 불순·체중 변화가 지속되면 **진료 상담을 고려**하고, 규칙적 운동·식습관 관리도 도움이 됩니다.")
        else:
            st.success("현재 입력 기준 위험도가 낮게 나왔습니다. 증상이 새로 생기거나 지속되면 언제든 진료를 받으세요.")

        st.markdown("##### 🔍 이 결과에 영향을 준 요인 (SHAP)")
        contrib=[]
        try:
            sv = shap.TreeExplainer(model).shap_values(Pool(row))
            contrib = sorted(zip(FEATURES, sv[0]), key=lambda x: abs(x[1]), reverse=True)[:8]
            labels=[KO.get(c,c) for c,_ in contrib]; vals=[v for _,v in contrib]
            colors=[BURGUNDY if v>0 else BLUE for v in vals]
            fig,ax=plt.subplots(figsize=(6.5,3.6))
            ax.barh(range(len(vals))[::-1], vals, color=colors)
            ax.set_yticks(range(len(vals))[::-1]); ax.set_yticklabels(labels, fontsize=10)
            ax.axvline(0,color="#888",lw=0.8); ax.set_xlabel("← 위험 낮춤   |   위험 높임 →", fontsize=9)
            for s in ["top","right"]: ax.spines[s].set_visible(False)
            ax.tick_params(length=0); plt.tight_layout(); st.pyplot(fig)
            st.caption("빨강 = 위험 높이는 방향, 파랑 = 낮추는 방향")
        except Exception as e:
            st.info(f"SHAP 표시 불가: {e}")

        if CMP:
            st.markdown("##### 📈 내 지표 vs 참여자 평균")
            html="<table style='width:100%;border-collapse:collapse;font-size:0.9rem;'>"
            html+="<tr style='background:#6B1F3B;color:white;'><th style='padding:6px;text-align:left;'>지표</th><th>내 값</th><th>PCOS군</th><th>비PCOS군</th></tr>"
            mymap={"BMI":inputs.get("BMI"),"Age (yrs)":inputs.get("Age (yrs)"),"Weight (Kg)":inputs.get("Weight (Kg)"),"Cycle length(days)":inputs.get("Cycle length(days)")}
            for key,info in CMP.items():
                html+=(f"<tr style='background:#FBF7F1;'><td style='padding:6px;'>{info['ko']}</td>"
                       f"<td style='text-align:center;font-weight:700;'>{mymap.get(key,'-')}</td>"
                       f"<td style='text-align:center;'>{info['pcos']}</td><td style='text-align:center;'>{info['nonpcos']}</td></tr>")
            html+="</table>"; st.markdown(html, unsafe_allow_html=True)

        now=datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        top3=", ".join([f"{KO.get(c,c)}({'+' if v>0 else '-'})" for c,v in contrib[:3]]) if contrib else "-"
        summary=(f"PCOS Check 결과 요약\n생성일시: {now}\n{'-'*40}\nPCOS 위험도: {pct}% ({band})\n"
                 f"주요 요인 Top3: {top3}\nBMI: {bmi} ({bmi_cat})\n{'-'*40}\n"
                 f"※ 의료 진단이 아닌 참고용입니다.\n기반: Zigarelli et al., JMIR Form Res 2022;6(3):e29967")
        st.download_button("📄 결과 요약 저장", summary, file_name=f"PCOS_결과_{now[:10]}.txt", mime="text/plain", use_container_width=True)

        st.divider()
        st.error("🏥 참고용입니다. 위험도가 높거나 증상이 지속되면 반드시 산부인과·내분비내과 진료를 받으세요.")
        if st.button("🔄 다시 진단하기", use_container_width=True):
            st.session_state.step = 1; st.session_state.answers = {}; st.rerun()
