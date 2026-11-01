import glob
import os
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats
import streamlit as st

# 1. 페이지 레이아웃 기본 설정
st.set_page_config(
    page_title="대장암 MSI 및 돌연변이 축적 분석",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 대장암 MSI 지표 및 돌연변이 축적 분석 시스템")
st.markdown(
    """
TCGA PanCancer Atlas 대장암(COAD/READ) 데이터를 기반으로 
**미세부수체 불안정성(Microsatellite Instability, MSI)** 상태와 
**체세포 돌연변이 수(Mutation Count)** 간의 상호작용 및 종양 해부학적 발생 위치별 차이를 탐색합니다.
"""
)


# 2. 데이터 자동 로드 및 파생변수 생성 함수
@st.cache_data
def load_and_preprocess_data():
    candidate_files = [
        "processed_colorectal_msi_no_sample_id.csv",
        "processed_colorectal_msi_3.csv",
        "processed_colorectal_msi.csv",
        "Colorectal Adenocarcinoma (TCGA, PanCancer Atlas).csv",
    ]

    # 현재 디렉터리의 모든 CSV 파일 탐색
    all_csvs = glob.glob("*.csv")
    search_list = candidate_files + [
        f for f in all_csvs if f not in candidate_files
    ]

    loaded_df = None
    loaded_name = None

    for fname in search_list:
        if os.path.exists(fname):
            try:
                temp_df = pd.read_csv(fname)
                # 필수 핵심 컬럼 포함 여부 확인
                if (
                    "MSIsensor Score" in temp_df.columns
                    and "Mutation Count" in temp_df.columns
                ):
                    loaded_df = temp_df
                    loaded_name = fname
                    break
            except Exception:
                continue

    if loaded_df is None:
        st.error(
            "⚠️ 분석용 CSV 파일을 찾을 수 없습니다. 프로젝트 폴더에 데이터 파일을 업로드해 주세요."
        )
        st.stop()

    # 파생 변수('MSIsensor 기반 MSI 상태', 'MANTIS 기반 MSI 상태') 자동 보정 생성
    if "MSIsensor 기반 MSI 상태" not in loaded_df.columns:
        loaded_df["MSIsensor 기반 MSI 상태"] = loaded_df[
            "MSIsensor Score"
        ].apply(
            lambda x: (
                "MSI-H" if x >= 3.5 else ("MSS" if pd.notna(x) else np.nan)
            )
        )

    if "MANTIS 기반 MSI 상태" not in loaded_df.columns:
        if "MSI MANTIS Score" in loaded_df.columns:
            loaded_df["MANTIS 기반 MSI 상태"] = loaded_df[
                "MSI MANTIS Score"
            ].apply(
                lambda x: (
                    "MSI-H" if x >= 0.4 else ("MSS" if pd.notna(x) else np.nan)
                )
            )

    return loaded_df, loaded_name


df, data_filename = load_and_preprocess_data()
has_sample_id = "Sample ID" in df.columns

# 3. 사이드바 컨트롤러
st.sidebar.header("📑 네비게이션 & 필터")

# 페이지 전환 메뉴 (공백 방지를 위한 라디오 네비게이션)
menu_selection = st.sidebar.radio(
    "이동할 페이지 선택",
    options=[
        "1. MSI vs Mutation Count 산점도",
        "2. MSI 상태별 Box Plot",
        "3. 종양 발생 위치별 산점도",
        "4. 원본 데이터 테이블 및 통계",
        "📌 전체 한눈에 보기 (통합 뷰)",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 분석 지표 설정")

# MSI 지표 선택
msi_metric = st.sidebar.selectbox(
    "MSI 지표 선택",
    options=["MSIsensor Score", "MSI MANTIS Score"],
    help="분석에 사용할 미세부수체 불안정성 점수를 선택합니다.",
)

if msi_metric == "MSIsensor Score":
    status_col = "MSIsensor 기반 MSI 상태"
    threshold = 3.5
    desc = "임계값 3.5 (MSS < 3.5 ≤ MSI-H)"
else:
    status_col = "MANTIS 기반 MSI 상태"
    threshold = 0.4
    desc = "임계값 0.4 (MSS < 0.4 ≤ MSI-H)"

st.sidebar.info(f"**판정 기준**: {desc}")

# 로그 스케일 옵션
use_log_scale = st.sidebar.checkbox(
    "Mutation Count 로그 스케일(Log10)",
    value=False,
    help="Hypermutated 샘플로 인한 왜곡을 완화합니다.",
)

# 해부학적 위치 필터
if "Tumor Disease Anatomic Site" in df.columns:
    available_sites = (
        df["Tumor Disease Anatomic Site"].dropna().unique().tolist()
    )
    selected_sites = st.sidebar.multiselect(
        "종양 발생 위치 필터",
        options=available_sites,
        default=available_sites,
    )
    filtered_df = df[
        df["Tumor Disease Anatomic Site"].isin(selected_sites)
    ].copy()
else:
    filtered_df = df.copy()

# 데이터 현황 메트릭 표시
st.sidebar.markdown("---")
st.sidebar.caption(f"📁 로드된 파일: `{data_filename}`")
valid_df = filtered_df[[msi_metric, "Mutation Count", status_col]].dropna()
st.sidebar.write(f"• 전체 데이터: **{len(df):,}** 건")
st.sidebar.write(f"• 필터 적용 데이터: **{len(valid_df):,}** 건")

# 공통 색상 매핑
color_map = {"MSS": "#1f77b4", "MSI-H": "#d62728"}
plotly_config = {"responsive": True, "displayModeBar": True}


# 4. 시각화 모듈 정의
def render_page_1():
    st.subheader(f"1. {msi_metric} 대비 Mutation Count 산점도")

    cols = [msi_metric, "Mutation Count", status_col]
    hover_cols = []
    if "Tumor Disease Anatomic Site" in filtered_df.columns:
        cols.append("Tumor Disease Anatomic Site")
        hover_cols.append("Tumor Disease Anatomic Site")
    if has_sample_id:
        cols.append("Sample ID")
        hover_cols.append("Sample ID")

    plot_df = filtered_df[cols].dropna()

    if plot_df.empty:
        st.warning("선택된 조건에 해당하는 유효 데이터가 없습니다.")
        return

    fig = px.scatter(
        plot_df,
        x=msi_metric,
        y="Mutation Count",
        color=status_col,
        color_discrete_map=color_map,
        hover_data=hover_cols if hover_cols else None,
        log_y=use_log_scale,
        labels={
            msi_metric: f"{msi_metric}",
            "Mutation Count": (
                "Mutation Count (Log10)" if use_log_scale else "Mutation Count"
            ),
            status_col: "MSI 판정 상태",
        },
        template="plotly_white",
        height=550,
    )

    fig.add_vline(
        x=threshold,
        line_width=2,
        line_dash="dash",
        line_color="#333333",
        annotation_text=f"기준치 ({threshold})",
        annotation_position="top right",
    )
    fig.update_layout(
        autosize=True,
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
    )
    st.plotly_chart(fig, use_container_width=True, config=plotly_config)

    corr, pval = stats.spearmanr(plot_df[msi_metric], plot_df["Mutation Count"])
    st.info(
        f"📊 **Spearman 상관계수**: {corr:.3f} (p-value: {pval:.2e}) — "
        f"{msi_metric} 점수가 높을수록 돌연변이 축적량이 통계적으로 유의미하게 증가합니다."
    )


def render_page_2():
    st.subheader(f"2. {status_col}에 따른 Mutation Count 분포 (Box Plot)")

    cols = [status_col, "Mutation Count"]
    hover_cols = []
    if "Tumor Disease Anatomic Site" in filtered_df.columns:
        cols.append("Tumor Disease Anatomic Site")
        hover_cols.append("Tumor Disease Anatomic Site")
    if has_sample_id:
        cols.append("Sample ID")
        hover_cols.append("Sample ID")

    plot_df = filtered_df[cols].dropna()

    if plot_df.empty:
        st.warning("선택된 조건에 해당하는 유효 데이터가 없습니다.")
        return

    fig = px.box(
        plot_df,
        x=status_col,
        y="Mutation Count",
        color=status_col,
        color_discrete_map=color_map,
        points="all",
        hover_data=hover_cols if hover_cols else None,
        log_y=use_log_scale,
        category_orders={status_col: ["MSS", "MSI-H"]},
        labels={
            status_col: "MSI 분류",
            "Mutation Count": (
                "Mutation Count (Log10)" if use_log_scale else "Mutation Count"
            ),
        },
        template="plotly_white",
        height=550,
    )
    fig.update_layout(autosize=True, showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config=plotly_config)

    # 비모수 Mann-Whitney U 검정
    mss_vals = plot_df[plot_df[status_col] == "MSS"]["Mutation Count"]
    msih_vals = plot_df[plot_df[status_col] == "MSI-H"]["Mutation Count"]

    if len(mss_vals) > 0 and len(msih_vals) > 0:
        stat, pval = stats.mannwhitneyu(
            msih_vals, mss_vals, alternative="greater"
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("MSS 중앙값", f"{mss_vals.median():.0f} 개")
        c2.metric("MSI-H 중앙값", f"{msih_vals.median():.0f} 개")
        c3.metric(
            "Mann-Whitney U p-value",
            f"{pval:.2e}",
            delta="유의미한 차이" if pval < 0.05 else "유의차 없음",
        )


def render_page_3():
    st.subheader(
        f"3. 종양 발생 위치별 {msi_metric} & Mutation Count 그룹 산점도"
    )

    if "Tumor Disease Anatomic Site" not in filtered_df.columns:
        st.warning("데이터에 'Tumor Disease Anatomic Site' 열이 없습니다.")
        return

    cols = [
        msi_metric,
        "Mutation Count",
        status_col,
        "Tumor Disease Anatomic Site",
    ]
    hover_cols = [status_col]
    if has_sample_id:
        cols.append("Sample ID")
        hover_cols.append("Sample ID")

    plot_df = filtered_df[cols].dropna()

    if plot_df.empty:
        st.warning("선택된 조건에 해당하는 유효 데이터가 없습니다.")
        return

    fig = px.scatter(
        plot_df,
        x=msi_metric,
        y="Mutation Count",
        color="Tumor Disease Anatomic Site",
        symbol=status_col,
        hover_data=hover_cols,
        log_y=use_log_scale,
        labels={
            msi_metric: f"{msi_metric}",
            "Mutation Count": (
                "Mutation Count (Log10)" if use_log_scale else "Mutation Count"
            ),
            "Tumor Disease Anatomic Site": "발생 부위",
        },
        template="plotly_white",
        height=550,
    )
    fig.add_vline(
        x=threshold,
        line_width=1.5,
        line_dash="dot",
        line_color="gray",
        annotation_text=f"기준 ({threshold})",
    )
    fig.update_layout(autosize=True)
    st.plotly_chart(fig, use_container_width=True, config=plotly_config)

    # 위치별 상태 집계표 출력
    st.markdown("##### 📍 발생 부위별 MSI 상태 분포표")
    summary = (
        plot_df.groupby(["Tumor Disease Anatomic Site", status_col])
        .size()
        .unstack(fill_value=0)
    )
    if "MSI-H" in summary.columns and "MSS" in summary.columns:
        summary["총계"] = summary["MSS"] + summary["MSI-H"]
        summary["MSI-H 비율(%)"] = (
            (summary["MSI-H"] / summary["총계"]) * 100
        ).round(1)
    st.dataframe(summary, use_container_width=True)


def render_page_4():
    st.subheader("4. 정제 데이터셋 테이블 및 다운로드")

    col_t1, col_t
