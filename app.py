import os
import glob
import numpy as np
import pandas as pd
import plotly.express as px
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
    search_order = candidate_files + [f for f in all_csvs if f not in candidate_files]

    for fpath in search_order:
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath)
                required_cols = ["MSIsensor Score", "Mutation Count", "MSI MANTIS Score"]
                if all(col in df.columns for col in required_cols):
                    return df, fpath
            except Exception:
                continue

    st.error("분석용 CSV 파일을 찾을 수 없습니다. 프로젝트 폴더의 데이터 파일을 확인해주세요.")
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

# [새로 추가] 3-3. 사이드바에 종양 발생 위치 선택(전체, 대장, 직장) UI 추가
selected_site_kr = st.sidebar.selectbox(
    "3. 종양 발생 위치 선택",
    options=["전체", "대장", "직장"],
    help="분석할 종양의 해부학적 발생 부위를 선택합니다.",
)

# [새로 추가] 선택된 한국어 옵션('대장', '직장')을 원본 데이터의 영문 명칭('Colon', 'Rectum')과 매핑하여 필터링
site_mapping = {"대장": "Colon", "직장": "Rectum"}
if selected_site_kr in site_mapping:
    filtered_df = df[df["Tumor Disease Anatomic Site"] == site_mapping[selected_site_kr]].copy()
else:
    # '전체' 선택 시 결측치를 제외한 전체 데이터 유지
    filtered_df = df.copy()

# 데이터 요약 정보 출력
st.sidebar.markdown("---")
st.sidebar.subheader("📊 데이터 요약")
st.sidebar.caption(f"로드된 파일: `{loaded_filename}`")
st.sidebar.write(f"• 전체 샘플 수: **{len(df):,}** 건")
st.sidebar.write(f"• 선택 부위({selected_site_kr}) 샘플 수: **{len(filtered_df):,}** 건")

# 기본 색상 매핑 및 축 레이블
color_discrete_map = {"MSS": "#1f77b4", "MSI-H": "#d62728"}
y_label = "Mutation Count (Log10 Scale)" if use_log_scale else "Mutation Count"

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

    cols1 = [msi_metric, "Mutation Count", status_col, "Tumor Disease Anatomic Site"]
    plot1_df = filtered_df[cols1].dropna()

    if plot1_df.empty:
        st.warning("선택된 조건에 해당하는 유효 데이터가 없습니다.")
    else:
        # [새로 추가] 산점도에 전체 데이터 경향성을 나타내는 OLS 선형 회귀선(trendline) 추가
        fig1 = px.scatter(
            plot1_df,
            x=msi_metric,
            y="Mutation Count",
            color=status_col,
            color_discrete_map=color_discrete_map,
            hover_data=["Tumor Disease Anatomic Site"],
            log_y=use_log_scale,
            trendline="ols",  # [새로 추가] 최소자승법(OLS) 기반 회귀선 생성
            trendline_scope="overall",  # [새로 추가] 집단별로 나뉘지 않고 전체 데이터를 포괄하는 단일 회귀선 설정
            trendline_color_override="#2ca02c",  # [새로 추가] 회귀선 색상을 녹색으로 지정하여 가시성 확보
            labels={
                msi_metric: f"{msi_metric}",
                "Mutation Count": y_label,
                status_col: "MSI 판정",
                "Tumor Disease Anatomic Site": "종양 위치",
            },
            template="plotly_white",
            height=550,
        )

        # 임계 기준선 추가
        fig1.add_vline(
            x=threshold,
            line_width=2,
            line_dash="dash",
            line_color="#333333",
            annotation_text=threshold_label,
            annotation_position="top right",
        )
        fig1.update_layout(
            autosize=True,
            legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
        )
        st.plotly_chart(fig1, use_container_width=True)

        # 상관계수 계산
        corr, pval = stats.spearmanr(plot1_df[msi_metric], plot1_df["Mutation Count"])
        st.info(
            f"📌 **Spearman 상관계수**: `{corr:.3f}` (p-value: `{pval:.2e}`) | "
            f"녹색 실선은 전체 데이터의 **선형 회귀선**입니다."
        )

# --- 탭 2: MSI 상태별 돌연변이 수 상자그림 ---
with tab2:
    st.subheader(f"2. {status_col}에 따른 돌연변이 수 분포 (Box Plot)")

    cols2 = [status_col, "Mutation Count", "Tumor Disease Anatomic Site"]
    plot2_df = filtered_df[cols2].dropna()

    if plot2_df.empty:
        st.warning("선택된 조건에 해당하는 유효 데이터가 없습니다.")
    else:
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
                "Mutation Count": y_label,
                "Tumor Disease Anatomic Site": "종양 위치",
            },
            template="plotly_white",
            height=550,
        )
        fig2.update_layout(autosize=True, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

        # 비모수 검정 (Mann-Whitney U Test)
        mss_mut = plot2_df[plot2_df[status_col] == "MSS"]["Mutation Count"]
        msih_mut = plot2_df[plot2_df[status_col] == "MSI-H"]["Mutation Count"]

        if len(mss_mut) > 0 and len(msih_mut) > 0:
            stat, p_val = stats.mannwhitneyu(msih_mut, mss_mut, alternative="greater")
            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("MSS 중앙값 (평균)", f"{mss_mut.median():.0f} ({mss_mut.mean():.1f})")
            col_s2.metric("MSI-H 중앙값 (평균)", f"{msih_mut.median():.0f} ({msih_mut.mean():.1f})")
            col_s3.metric("Mann-Whitney U p-value", f"{p_val:.2e}")

# --- 탭 3: 종양 발생 위치별 그룹 산점도 ---
with tab3:
    st.subheader(f"3. 종양 발생 위치에 따른 {msi_metric}와 돌연변이 수 산점도")

    cols3 = [msi_metric, "Mutation Count", status_col, "Tumor Disease Anatomic Site"]
    plot3_df = filtered_df[cols3].dropna()

    if plot3_df.empty:
        st.warning("선택된 조건에 해당하는 유효 데이터가 없습니다.")
    else:
        # [새로 추가] 종양 발생 위치(Colon, Rectum) 그룹별 각각의 OLS 회귀선(trendline) 추가
        fig3 = px.scatter(
            plot3_df,
            x=msi_metric,
            y="Mutation Count",
            color="Tumor Disease Anatomic Site",
            symbol=status_col,
            hover_data=[status_col],
            log_y=use_log_scale,
            trendline="ols",  # [새로 추가] 각 위치 그룹별 회귀선 자동 계산 및 표시
            labels={
                msi_metric: f"{msi_metric}",
                "Mutation Count": y_label,
                "Tumor Disease Anatomic Site": "종양 발생 부위",
                status_col: "MSI 판정",
            },
            template="plotly_white",
            height=550,
        )
        fig3.add_vline(
            x=threshold,
            line_width=1.5,
            line_dash="dot",
            line_color="gray",
            annotation_text=threshold_label,
        )
        fig3.update_layout(autosize=True)
        st.plotly_chart(fig3, use_container_width=True)

        if selected_site_kr != "전체":
            st.caption(f"💡 현재 사이드바에서 **'{selected_site_kr}'** 부위만 선택되어 단일 그룹 회귀선이 표시됩니다.")

        # 부위별 상태 분포 요약표
        st.markdown("##### 📍 발생 부위별 MSI 분류 현황")
        site_summary = (
            plot3_df.groupby(["Tumor Disease Anatomic Site", status_col])
            .size()
            .unstack(fill_value=0)
        )
        if "MSI-H" in site_summary.columns and "MSS" in site_summary.columns:
            site_summary["합계"] = site_summary["MSS"] + site_summary["MSI-H"]
            site_summary["MSI-H 비율(%)"] = (
                (site_summary["MSI-H"] / site_summary["합계"]) * 100
            ).round(1)
        st.dataframe(site_summary, use_container_width=True)

# --- 탭 4: 원본 데이터 테이블 ---
with tab4:
    st.subheader("4. 정제 데이터셋 테이블")
    st.dataframe(filtered_df, use_container_width=True)

    csv_data = filtered_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 필터링된 데이터 CSV 다운로드",
        data=csv_data,
        file_name="processed_colorectal_msi_filtered.csv",
        mime="text/csv",
    )
