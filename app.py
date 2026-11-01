import glob
import os
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

# 원자료 출처, cBioPortal 플랫폼 정보 및 GitHub README 권장 안내 텍스트
st.markdown(
    """
TCGA 대장선암 데이터를 기반으로 **미세부수체 불안정성(MSI) 지표**와 
**체세포 돌연변이 수(Mutation Count)**의 상관관계 및 종양 발생 부위별 분포를 분석합니다.

- **원자료**: The Cancer Genome Atlas (TCGA), Colorectal Adenocarcinoma (PanCancer Atlas)
- **데이터 제공·탐색 플랫폼**: cBioPortal
- 웹앱을 이용하기 전에 GitHub의 README를 먼저 읽어보시면, 연구의 목적과 데이터 및 분석 방법을 이해하는 데 도움이 됩니다.
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
                    if "Tumor Disease Anatomic Site" in df.columns:
                        df["Tumor Disease Anatomic Site"] = df[
                            "Tumor Disease Anatomic Site"
                        ].apply(lambda x: x.strip() if isinstance(x, str) else x)
                    return df, fpath
            except Exception:
                continue

    st.error(
        "분석용 CSV 파일을 찾을 수 없습니다. 프로젝트 폴더에 데이터 파일을 확인해주세요."
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
    help="초변이(Hypermutation) 샘플로 인한 극단적 왜곡을 완화하여 저변이 구간의 차이를 명확히 확인합니다.",
)

# 3-3. 사이드바 종양 발생 위치 선택
site_selection = st.sidebar.selectbox(
    "3. 종양 발생 위치 선택",
    options=["전체", "대장", "직장"],
    help="특정 발생 부위(전체 / 대장 / 직장)를 선택하여 모든 산점도 및 분석 결과를 동적으로 필터링합니다.",
)

site_mapping = {"대장": "Colon", "직장": "Rectum"}

if site_selection == "전체":
    filtered_df = df.copy()
else:
    target_site = site_mapping[site_selection]
    filtered_df = df[df["Tumor Disease Anatomic Site"] == target_site].copy()

# 사이드바 데이터 요약 현황
st.sidebar.markdown("---")
st.sidebar.subheader("📊 필터 적용 현황")
st.sidebar.caption(f"로드된 파일: `{loaded_filename}`")
st.sidebar.write(f"• 전체 원본 샘플: **{len(df):,}** 건")
st.sidebar.write(
    f"• **현재 선택({site_selection}) 샘플**: **{len(filtered_df):,}** 건"
)

# 기본 색상 매핑
color_discrete_map = {"MSS": "#1f77b4", "MSI-H": "#d62728"}
y_label = "Mutation Count (Log10 Scale)" if use_log_scale else "Mutation Count"

# ----------------- 4. 분석 결과 탭 구성 -----------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "1. MSI 점수 vs 돌연변이 수 산점도",
        "2. MSI 상태별 돌연변이 수 상자그림",
        "3. 종양 위치별 그룹 산점도",
        "4. 원본 데이터 미리보기",
    ]
)

# --- 탭 1: MSI 점수와 돌연변이 수 산점도 ---
with tab1:
    st.subheader(
        f"1. {msi_metric}와 돌연변이 수(Mutation Count) 산점도 [선택 부위: {site_selection}]"
    )

    cols1 = [
        msi_metric,
        "Mutation Count",
        status_col,
        "Tumor Disease Anatomic Site",
    ]
    plot1_df = filtered_df[cols1].dropna()

    if plot1_df.empty:
        st.warning(
            f"'{site_selection}' 부위에 해당하는 유효 데이터가 없습니다."
        )
    else:
        fig1 = px.scatter(
            plot1_df,
            x=msi_metric,
            y="Mutation Count",
            color=status_col,
            color_discrete_map=color_discrete_map,
            hover_data=["Tumor Disease Anatomic Site"],
            log_y=use_log_scale,
            trendline="ols",
            trendline_scope="overall",
            trendline_color_override="#2b2b2b",
            labels={
                msi_metric: f"{msi_metric}",
                "Mutation Count": y_label,
                status_col: "MSI 판정",
                "Tumor Disease Anatomic Site": "종양 위치",
            },
            template="plotly_white",
            height=550,
        )

        fig1.add_vline(
            x=threshold,
            line_width=2,
            line_dash="dash",
            line_color="#e74c3c",
            annotation_text=threshold_label,
            annotation_position="top right",
        )
        fig1.update_layout(
            autosize=True,
            legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
        )
        st.plotly_chart(fig1, use_container_width=True)

        corr, pval = stats.spearmanr(
            plot1_df[msi_metric], plot1_df["Mutation Count"]
        )
        st.info(
            f"📌 **현재 분석 대상**: {site_selection} (유효 샘플 {len(plot1_df):,}건)\n\n"
            f"📌 **Spearman 상관계수**: `{corr:.3f}` (p-value: `{pval:.2e}`) — "
            f"그래프 상의 검은 실선(회귀선) 위에 마우스를 올리면 회귀식과 R²을 확인할 수 있습니다."
        )

    st.markdown(
        """
        > **분석 종합 요약**:  
        > MSIsensor 점수가 높을수록 돌연변이 수도 증가하는 양의 상관관계가 나타나며, 상관계수는 0.348이다. MANTIS 점수와 돌연변이 수 사이에서도 양의 상관관계가 나타나며, 상관계수는 0.360이다. 따라서 두 MSI 지표 모두 돌연변이 수와 양의 상관관계를 보인다. 즉, MSI 정도가 높은 종양에서 돌연변이 수도 많은 경향이 관찰된다. 다만 상관계수가 약 0.35 수준이므로 매우 강한 관계라고 보기는 어렵다.
        """
    )

# --- 탭 2: MSI 상태별 돌연변이 수 상자그림 ---
with tab2:
    st.subheader(
        f"2. {status_col}에 따른 돌연변이 수 분포 (Box Plot) [선택 부위: {site_selection}]"
    )

    cols2 = [status_col, "Mutation Count", "Tumor Disease Anatomic Site"]
    plot2_df = filtered_df[cols2].dropna()

    if plot2_df.empty:
        st.warning(
            f"'{site_selection}' 부위에 해당하는 유효 데이터가 없습니다."
        )
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

        mss_mut = plot2_df[plot2_df[status_col] == "MSS"]["Mutation Count"]
        msih_mut = plot2_df[plot2_df[status_col] == "MSI-H"]["Mutation Count"]

        if len(mss_mut) > 0 and len(msih_mut) > 0:
            stat, p_val = stats.mannwhitneyu(
                msih_mut, mss_mut, alternative="greater"
            )
            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric(
                "MSS 중앙값 (평균)",
                f"{mss_mut.median():.0f} ({mss_mut.mean():.1f})",
            )
            col_s2.metric(
                "MSI-H 중앙값 (평균)",
                f"{msih_mut.median():.0f} ({msih_mut.mean():.1f})",
            )
            col_s3.metric("Mann-Whitney U p-value", f"{p_val:.2e}")

    st.markdown(
        """
        > **집단 비교 분석 요약**:  
        > MSIsensor 기준 MSI-H집단의 돌연변이 수 중앙값은 1,164개로 MSS 집단의 98개보다 약 11.9배 높다. MANTIS기준 MSI-H 집단의 돌연변이 수 중앙값은 1,083개로 MSS 집단의 98개보다 약 11.1배 높다. 두 MSI 판정 기준 모두 MSI-H 집단에서 돌연변이 수가 훨씬 많은 경향을 보인다.
        """
    )

# --- 탭 3: 종양 발생 위치별 그룹 산점도 ---
with tab3:
    st.subheader(
        f"3. 종양 발생 위치에 따른 {msi_metric}와 돌연변이 수 산점도 [선택 부위: {site_selection}]"
    )

    cols3 = [
        msi_metric,
        "Mutation Count",
        status_col,
        "Tumor Disease Anatomic Site",
    ]
    plot3_df = filtered_df[cols3].dropna()

    if plot3_df.empty:
        st.warning(
            f"'{site_selection}' 부위에 해당하는 유효 데이터가 없습니다."
        )
    else:
        fig3 = px.scatter(
            plot3_df,
            x=msi_metric,
            y="Mutation Count",
            color="Tumor Disease Anatomic Site",
            symbol=status_col,
            hover_data=[status_col],
            log_y=use_log_scale,
            trendline="ols",
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

        # [신규 추가] 종양 발생 부위별 Spearman & Pearson 상관계수 및 유의확률 계산
        corr_records = []
        for site, group in plot3_df.groupby("Tumor Disease Anatomic Site"):
            # 상관계수 계산을 위해 최소 3개 이상의 표본이 있는 경우 실행
            if len(group) >= 3:
                sp_r, sp_p = stats.spearmanr(group[msi_metric], group["Mutation Count"])
                pe_r, pe_p = stats.pearsonr(group[msi_metric], group["Mutation Count"])
                corr_records.append({
                    "종양 발생 부위": site,
                    "유효 표본 수 (N)": len(group),
                    "Spearman 상관계수 (r)": round(sp_r, 3),
                    "Spearman p-value": f"{sp_p:.2e}",
                    "Pearson 상관계수 (r)": round(pe_r, 3),
                    "Pearson p-value": f"{pe_p:.2e}",
                    "통계적 유의성": "유의함 (p < 0.05)" if sp_p < 0.05 else "유의하지 않음 (p ≥ 0.05)",
                })

        # [신규 추가] 계산된 상관계수 결과를 데이터프레임으로 변환하여 표 형태로 화면에 출력
        if corr_records:
            st.markdown(f"##### 📊 종양 발생 부위별 {msi_metric} & Mutation Count 상관계수 요약표")
            corr_df = pd.DataFrame(corr_records)
            st.dataframe(corr_df, use_container_width=True)

        st.markdown(f"##### 📍 발생 부위별 MSI 분류 현황 ({site_selection} 기준)")
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

    st.markdown(
        """
        > **종양 발생 부위별 종합 해석**:  
        > MSIsensor 기준으로 대장과 직장의 MSI 점수와 돌연변이 수 사이의 관계는 서로 다른 양상을 보이며, 대장과 달리 직장에서는 MSI 점수가 높아질수록 돌연변이 수가 증가하는 경향이 관찰된다. MANTIS 기준으로 대장과 직장 모두 MSI 점수가 높아질수록 돌연변이 수가 증가하는 경향을 보이며, 발생 위치에 따라 추세의 양상에는 차이가 있다. 그러나 발생 위치별 표본 수의 차이가 있기 때문에 표본 수의 차이를 고려한 추가적인 검토가 필요하다.
        """
    )

# --- 탭 4: 원본 데이터 테이블 ---
with tab4:
    st.subheader(f"4. 데이터셋 테이블 ({site_selection} 기준)")
    st.dataframe(filtered_df, use_container_width=True)

    csv_data = filtered_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label=f"📥 {site_selection} 데이터 CSV 다운로드",
        data=csv_data,
        file_name=f"colorectal_msi_{site_selection}.csv",
        mime="text/csv",
    )
