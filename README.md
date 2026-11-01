# Colorectal Cancer MSI & Mutation Accumulation Dashboard

대장선암(Colorectal Adenocarcinoma, TCGA PanCancer Atlas) 데이터를 활용하여 미세부수체 불안정성(MSI)과 체세포 돌연변이 수(Mutation Count) 간의 관계를 분석하고 대화형 시각화를 제공하는 웹 애플리케이션입니다.

## 1. 프로젝트 개요
- **분석 목표**: DNA Mismatch Repair(MMR) 결손으로 인해 발생하는 MSI 현상이 유전체 내 돌연변이 누적(Hypermutation)에 미치는 영향을 규명합니다.
- **분석 지표**:
  - `MSIsensor Score`: 차세대 염기서열 분석(NGS) 기반 정상-종양 리드 비교 알고리즘 (3.5 이상: MSI-H, 3.5 미만: MSS)
  - `MSI MANTIS Score`: 미세부수체 부위 변이 거리 기반 점수 (0.4 이상: MSI-H, 0.4 미만: MSS)

## 2. 데이터 구조
정제된 파일(`processed_colorectal_msi.csv`)은 원본 63개 열 중 분석에 필요한 7개 컬럼만을 선별하여 구성되었습니다:
1. `Sample ID`: TCGA 환자 검체 식별자
2. `Tumor Disease Anatomic Site`: 종양 원발 부위 (Colon, Rectum 등)
3. `MSIsensor Score`: MSIsensor 지표값
4. `MSIsensor 기반 MSI 상태`: MSIsensor 점수 기준 범주형 상태 (MSS, MSI-H)
5. `MSI MANTIS Score`: MANTIS 지표값
6. `MANTIS 기반 MSI 상태`: MANTIS 점수 기준 범주형 상태 (MSS, MSI-H)
7. `Mutation Count`: 비동의 돌연변이 수

## 3. 설치 및 실행 방법

### 1) 환경 준비 및 패키지 설치
```bash
pip install -r requirements.txt