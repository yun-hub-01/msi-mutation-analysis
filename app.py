import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    file_path = "processed_colorectal_msi.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        # 파일이 없을 경우 원본 파일에서 온더플라이 전처리
        raw_df = pd.read_csv(
            "Colorectal Adenocarcinoma (TCGA, PanCancer Atlas).csv"
        )
        selected_cols = [
            "MSIsensor Score",
            "Mutation Count",
            "MSI MANTIS Score",
            "Tumor Disease Anatomic Site",
        ]
        df = raw_df[selected_cols].copy()
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
        final_order = [
            "MSIsensor Score",
            "Mutation Count",
            "MSI MANTIS Score",
            "MSIsensor 기반 MSI 상태",
            "MANTIS 기반 MSI 상태",
            "Tumor Disease Anatomic Site",
        ]
        df = df[final_order]
    return df


df = load_data()

# ----------------- 사이드바 설정 -----------------
st.sidebar.header("🔍 분석 필터 설정")

# 1. MSI 지표 선택
msi_metric = st.sidebar.selectbox(
    "MSI 지표 선택",
    options=["MSIsensor Score", "MSI MANTIS Score"],
    help="분석에 사용할 미세부수체 불안정성 산출 알고리즘을 선택합니다.",
)

# 선택된 지표에 따른 상태 컬럼 및 임계값 맵핑
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

# 3. 해부학적 위치 필터 (선택 사항)
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

# 공통 색상 팔레트
color_discrete_map = {"MSS": "#3366CC", "MSI-H": "#DC3912"}

# --- TAB 1: MSI 점수 vs 돌연변이 수 산점도 ---
with tab1:
    st.subheader(f"{msi_metric}와 Mutation Count의 상관관계 산점도")

    # 결측치 제거
    plot1_df = filtered_df[
        [msi_metric, "Mutation Count", status_col, "Tumor Disease Anatomic Site"]
    ].dropna()

    fig1 = px.scatter(
        plot1_df,
        x=msi_metric,
        y="Mutation Count",
        color=status_col,
        color_discrete_map=color_discrete_map,
        hover_data=["Tumor Disease Anatomic Site"],
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

    # 임계값 수직 기준선 추가
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

    # 상관계수 계산 (Spearman rank correlation)
    corr, pval = stats.spearmanr(plot1_df[msi_metric], plot1_df["Mutation Count"])
    st.caption(
        f"📌 **Spearman 상관계수**: {corr:.3f} (p-value: {pval:.2e}) — 미세부수체 불안정성이 높을수록 유의미하게 많은 돌연변이가 축적되는 경향을 보입니다."
    )

# --- TAB 2: MSI-H vs MSS 상자그림 ---
with tab2:
    st.subheader(f"{status_col}에 따른 돌연변이 수 분포 (Box Plot)")

    plot2_df = filtered_df[
        [status_col, "Mutation Count", "Tumor Disease Anatomic Site"]
    ].dropna()

    fig2 = px.box(
        plot2_df,
        x=status_col,
        y="Mutation Count",
        color=status_col,
        color_discrete_map=color_discrete_map,
        points="all",
        hover_data=["Tumor Disease Anatomic Site"],
        log_y=use_log_scale,
        category_orders={status_col: ["MSS", "MSI-H"]},
        labels={
            status_col: "MSI 상태",
            "Mutation Count": (
                "Mutation Count (Log Scale)"
                if use_log_scale
                else "Mutation Count"
            ),
        },
        title="MSI 상태(MSS vs MSI-H)별 돌연변이 수 분포 비교",
        template="plotly_white",
    )

    fig2.update_layout(height=550, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    # 비모수 통계 검정 (Mann-Whitney U Test)
    mss_mut = plot2_df[plot2_df[status_col] == "MSS"]["Mutation Count"]
    msih_mut = plot2_df[plot2_df[status_col] == "MSI-H"]["Mutation Count"]

    if len(mss_mut) > 0 and len(msih_mut) > 0:
        stat, p_val = stats.mannwhitneyu(msih_mut, mss_mut, alternative="greater")
        st.markdown(
            f"""
            - **MSS 중앙값(Median)**: {mss_mut.median():.1f} 개 (평균: {mss_mut.mean():.1f})
            - **MSI-H 중앙값(Median)**: {msih_mut.median():.1f} 개 (평균: {msih_mut.mean():.1f})
            - **Mann-Whitney U 검정**: p-value = `{p_val:.2e}` (MSI-H 군의 돌연변이 축적이 통계적으로 유의하게 높음)
            """
        )

# --- TAB 3: 종양 발생 위치별 산점도 ---
with tab3:
    st.subheader(
        f"종양 발생 위치(Anatomic Site)별 {msi_metric} 및 Mutation Count 분석"
    )

    plot3_df = filtered_df[
        [msi_metric, "Mutation Count", status_col, "Tumor Disease Anatomic Site"]
    ].dropna()

    fig3 = px.scatter(
        plot3_df,
        x=msi_metric,
        y="Mutation Count",
        color="Tumor Disease Anatomic Site",
        symbol=status_col,
        hover_data=[status_col],
        log_y=use_log_scale,
        labels={
            msi_metric: f"{msi_metric} 값",
            "Mutation Count": (
                "Mutation Count (Log Scale)"
                if use_log_scale
