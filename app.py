import glob
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import streamlit as st

# 1. 페이지 설정
st.set_page_config(
    page_title="대장암 MSI 및 돌연변이 축적 분석",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 대장암 미세부수체 불안정성(MSI) 및 돌연변이 축적 분석")
st.markdown(
    """
TCGA 대장선암 데이터를 기반으로 **미세부수체 불안정성(MSI) 지표**와 
**체세포 돌연변이 수(Mutation Count)**의 상관관계 및 종양 발생 부위별 분포를 분석합니다.
"""
)


# 2. 데이터 로드 함수
@st.cache_data
def load_data():
    candidate_files = [
        "processed_colorectal_msi.csv",
        "processed_colorectal_msi_4.csv",
        "processed_colorectal_msi_no_sample_id.csv",
        "processed_colorectal_msi",
    ]

    all_csvs = glob.glob("*.csv")
    search_order = candidate_files + [
        f for f in all_csvs if f not in candidate_files
    ]

    for fpath in search_order:
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath)
                required_cols = [
                    "MSIsensor Score",
                    "Mutation Count",
                    "MSI MANTIS Score",
                ]
                if all(col in df.columns for col in required_cols):
                    return df, fpath
            except Exception:
                continue

    st.error(
        "분석용 CSV 파일을 찾을 수 없습니다. 프로젝트 폴더의 데이터 파일을 확인해주세요."
    )
    st.stop()


df, loaded_filename = load_data()

# ----------------- 3. 사이드바 컨트롤러 (필터 조건) -----------------
st.sidebar.header("⚙️ 분석 필터 조건")

# 3-1. MSI 지표 선택
msi_metric = st.sidebar.selectbox(
    "1. MSI 지표 선택",
    options=["MSIsensor Score", "MSI MANTIS Score"],
    help="분석 기준이 될 미세부수체 불안정성(MSI) 산출 알고리즘을 선택합니다.",
)

# 선택된 지표에 따른 상태 열 및 임계값 설정
if msi_metric == "MSIsensor Score":
    status_col = "MSIsensor 기반 MSI 상태"
    threshold = 3.5
    threshold_label = "기준선 (3.5)"
else:
    status_col = "MANTIS 기반 MSI 상태"
    threshold = 0.4
    threshold_label = "기준선 (0.4)"

# 3-2. Mutation Count 로그 스케일 여부
use_log_scale = st.sidebar.checkbox(
    "2. Mutation Count 로그 스케일(Log10) 적용",
    value=False,
    help="초변이(Hypermutation) 샘플로 인한 왜곡을 완화하여 저변이 구간의 차이를 확인합니다.",
)

# 3-3. 사이드바에 종양 발생 위치 선택(전체, 대장, 직장)
selected_site_kr = st.sidebar.selectbox(
    "3. 종양 발생 위치 선택",
    options=["전체", "대장", "직장"],
    help="분석할 종양의 해부학적 발생 부위를 선택합니다.",
)

# 선택된 옵션을 원본 영문 컬럼값과 매핑하여 필터링
site_mapping = {"대장": "Colon", "직장": "Rectum"}
if selected_site_kr in site_mapping:
    filtered_df = df[
        df["Tumor Disease Anatomic Site"] == site_mapping[selected_site_kr]
    ].copy()
else:
    filtered_df = df.copy()

# 데이터 요약 정보 출력
st.sidebar.markdown("---")
st.sidebar.subheader("📊 데이터 요약")
st.sidebar.caption(f"로드된 파일: `{loaded_filename}`")
st.sidebar.write(f"• 전체 샘플 수: **{len(df):,}** 건")
st.sidebar.write(
    f"• 선택 부위({selected_site_kr}) 샘플 수: **{len(filtered_df):,}** 건"
)

# 기본 색상 매핑 및 축 레이블
color_discrete_map = {"MSS": "#1f77b4", "MSI-H": "#d62728"}
site_colors = {"Colon": "#2ca02c", "Rectum": "#9467bd"}
y_label = "Mutation Count (Log10 Scale)" if use_log_scale else "Mutation Count"


# statsmodels 없이 scipy.stats로 선형/로그 회귀선을 안전하게 계산하는 함수
def calculate_regression_line(
    x_vals, y_vals, is_log_scale=False, n_points=100
):
    mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
    x_clean = np.array(x_vals)[mask]
    y_clean = np.array(y_vals)[mask]

    if len(x_clean) < 2 or (x_clean.max() == x_clean.min()):
        return None, None

    x_line = np.linspace(x_clean.min(), x_clean.max(), n_points)

    if is_log_scale:
        y_positive = np.where(y_clean > 0, y_clean, 1.0)
        log_y = np.log10(y_positive)
        slope, intercept, _, _, _ = stats.linregress(x_clean, log_y)
        y_line = 10 ** (slope * x_line + intercept)
    else:
        slope, intercept, _, _, _ = stats.linregress(x_clean, y_clean)
        y_line = slope * x_line + intercept

    return x_line, y_line


# ----------------- 4. 분석 결과 탭 구성 -----------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "1. MSI 점수 vs 돌연변이 수 산점도",
        "2. MSI 상태별 돌연변이 수 상자그림",
        "3. 종양 위치별 그룹 산점도",
        "4. 원본 데이터 테이블",
    ]
)

# --- 탭 1: MSI 점수와 돌연변이 수 산점도 ---
with tab1:
    st.subheader(f"1. {msi_metric}와 돌연변이 수(Mutation Count) 산점도")

    cols1 = [
        msi_metric,
        "Mutation Count",
        status_col,
        "Tumor Disease Anatomic Site",
    ]
    plot1_df = filtered_df[cols1].dropna()

    if plot1_df.empty:
        st.warning("선택된 조건에 해당하는 유효 데이터가 없습니다.")
    else:
        fig1 = px.scatter(
            plot1_df,
            x=msi_metric,
            y="Mutation Count",
            color=status_col,
            color_discrete_map=color_discrete_map,
            hover_data=["Tumor Disease Anatomic Site"],
            log_y=use_log_scale,
            labels={
                msi_metric: f"{msi_metric}",
                "Mutation Count": y_label,
                status_col: "MSI 판정",
                "Tumor Disease Anatomic Site
