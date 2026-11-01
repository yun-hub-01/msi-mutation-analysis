import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats
import streamlit as st

# 페이지 환경 설정
st.set_page_config(
    page_title="대장암 MSI 및 돌연변이 축적 분석",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 대장암 MSI 지표 및 돌연변이 축적 분석 대시보드")
st.markdown(
    """
TCGA PanCancer Atlas 대장암(COAD/READ) 데이터를 기반으로 
**미세부수체 불안정성(Microsatellite Instability, MSI)** 상태와 
**체세포 돌연변이 수(Mutation Count)** 간의 상호작용 및 종양 해부학적 발생 위치별 차이를 탐색합니다.
"""
)


# 데이터 로드 캐싱
@st.cache_data
def load_data():
    file_candidates = [
        "processed_colorectal_msi_no_sample_id.csv",
        "processed_colorectal_msi.csv",
        "Colorectal Adenocarcinoma (TCGA, PanCancer Atlas).csv",
    ]

    loaded_file = None
    for file_name in file_candidates:
        try:
            raw_df = pd.read_csv(file_name)
            loaded_file = file_name
            break
        except FileNotFoundError:
            continue

    if loaded_file is None:
        st.error(
            "분석용 CSV 파일을 찾을 수 없습니다. 프로젝트 디렉토리에 데이터 파일을 배치해 주세요."
        )
        st.stop()

    # 원본 파일이 직접 읽혔을 경우 필요한 파생 변수 및 열 구성
    if "MSIsensor 기반 MSI 상태" not in raw_df.columns:
        selected_cols = [
            "MSIsensor Score",
            "Mutation Count",
            "MSI MANTIS Score",
            "Tumor Disease Anatomic Site",
        ]
        if "Sample ID" in raw_df.columns:
            selected_cols.append("Sample ID")

        df = raw_df[
            [col for col in selected_cols if col in raw_df.columns]
        ].copy()

        df["MSIsensor 기반 MSI 상태"] = df["MSIsensor Score"].apply(
            lambda x: (
                "MSI-H" if x >= 3.5 else ("MSS" if pd.notna(x) else np.nan)
            )
        )
        df["MANTIS 기반 MSI 상태"] = df["MSI MANTIS Score"].apply(
            lambda x: (
                "MSI-H" if x >= 0.4 else ("MSS" if pd.notna(x) else np.nan)
            )
        )
    else:
        df = raw_df.copy()

    return df


df = load_data()

# Sample ID 열 존재 여부 확인
has_sample_id = "Sample ID" in df.columns

# ----------------- 사이드바 설정 -----------------
st.sidebar.header("🔍 분석 필터 설정")

# 1. MSI 지표 선택
msi_metric = st.sidebar.selectbox(
    "MSI 지표 선택",
    options=["MSIsensor Score", "MSI MANTIS Score"],
    help="분석에 사용할 미세부수체 불안정성 산출 알고리즘을 선택합니다.",
)

if msi_metric == "MSIsensor Score":
    status_col = "MSIsensor 기반 MSI 상태"
    threshold = 3.5
    desc = "임계값: 3.5 (MSS < 3.5 ≤ MSI-H)"
else:
    status_col = "MANTIS 기반 MSI 상태"
    threshold = 0.4
    desc = "임계값: 0.4 (MSS < 0.4 ≤ MSI-H)"

st.sidebar.info(f"**선택된 기준**: {desc}")

# 2. Y축 로그 스케일 적용 여부
use_log_scale = st.sidebar.checkbox(
    "Mutation Count 로그 스케일(Log10) 적용",
    value=False,
    help="초변이(Hypermutation) 샘플로 인한 왜곡을 완화하여 저변이 구간의 분포를 명확히 관찰합니다.",
)

# 3. 해부학적 위치 필터
available_sites = df["Tumor Disease Anatomic Site"].dropna().unique().tolist()
selected_sites = st.sidebar.multiselect(
    "종양 발생 위치 필터링",
    options=available_sites,
    default=available_sites,
    help="분석에 포함할 종양 부위를 선택합니다.",
)

# 필터링 적용
filtered_df = df[
    df["Tumor Disease Anatomic Site"].isin(selected_sites)
].copy()

# 요약 지표 출력
st.sidebar.markdown("---")
st.sidebar.subheader("📊 데이터 개요")
total_samples = len(df)
valid_metric_samples = filtered_df[
    [msi_metric, "Mutation Count", status_col]
].dropna()

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.metric("전체 샘플 수", f"{total_samples} 건")
with col_m2:
    st.metric("유효 분석 샘플 수", f"{len(valid_metric_samples)} 건")

msi_counts = valid_metric_samples[status_col].value_counts()
st.sidebar.write(f"**{status_col} 분포**:")
for status, count in msi_counts.items():
    st.sidebar.write(
        f"- {status}: {count}건 ({count/len(valid_metric_samples)*100:.1f}%)"
    )

# ----------------- 탭 구성 및 메인 시각화 -----------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "1. MSI 점수 vs 돌연변이 수 산점도",
        "2. MSI 상태별 돌연변이 수 상자그림",
        "3. 종양 위치별 그룹 산점도",
        "4. 원본 데이터 테이블",
    ]
)

color_discrete_map = {"MSS": "#3366CC", "MSI-H": "#DC3912"}

# --- TAB 1: MSI 점수 vs 돌연변이 수 산점도 ---
with tab1:
    st.subheader(f"{msi_metric}와 Mutation Count의 상관관계 산점도")

    cols1 = [
        msi_metric,
        "Mutation Count",
        status_col,
        "Tumor Disease Anatomic Site",
    ]
    hover_cols1 = ["Tumor Disease Anatomic Site"]
    if has_sample_id:
        cols1.append("Sample ID")
        hover_cols1.insert(0, "Sample ID")

    plot1_df = filtered_df[cols1].dropna()

    fig1 = px.scatter(
        plot1_df,
        x=msi_metric,
        y="Mutation Count",
        color=status_col,
        color_discrete_map=color_discrete_map,
        hover_data=hover_cols1,
        log_y=use_log_scale,
        labels={
            msi_metric: f"{msi_metric} 값",
            "Mutation Count": (
                "Mutation Count (Log Scale)"
                if use_log_scale
                else "Mutation Count"
            ),
            status_col: "MSI 판정 상태",
        },
        title=f"{msi_metric} 대비 체세포 돌연변이 축적도",
        template="plotly_white",
    )

    fig1.add_vline(
        x=threshold,
        line_width=2,
        line_dash="dash",
        line_color="black",
        annotation_text=f"기준선 ({threshold})",
        annotation_position="top right",
    )

    fig1.update_layout(
        height=550, legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02)
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Spearman 상관계수 산출
    corr, pval = stats.spearmanr(
        plot1_df[msi_metric], plot1_df["Mutation Count"]
    )
    st.caption(
        f"📌 **Spearman 상관계수**: {corr:.3f} (p-value: {pval:.2e}) — 미세부수체 불안정성이 높을수록 돌연변이가 유의미하게 많이 축적되는 양의 상관성을 나타냅니다."
    )
