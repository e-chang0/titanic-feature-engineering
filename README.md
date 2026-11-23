# 타이타닉 생존 예측과 특성 공학 비교

타이타닉 승객 데이터를 이용해 전처리 방법에 따라 생존 예측 성능이 어떻게 달라지는지 비교한 과제이다.

결측치 처리, 범주형 인코딩, 스케일링, 파생 변수 생성, 변수 선택을 적용하고 로지스틱 회귀와 랜덤 포레스트로 성능을 확인했다.

## 먼저 볼 파일

- [실행방법.md](실행방법.md): 설치부터 보고서 생성까지 실행 순서
- [실험설명.md](실험설명.md): 데이터, 전처리 방법, 비교 실험과 결과 해석
- [src/titanic_pipeline.py](src/titanic_pipeline.py): 전처리와 모델 학습의 주요 코드

## 사용한 데이터

Kaggle Titanic의 정답이 있는 학습 데이터 891개를 사용했다. 예측 대상은 `Survived`이며, 0은 사망, 1은 생존이다.

전체 자료를 학습용 712개와 최종 평가용 179개로 나눴다. 생존자 비율이 비슷하게 나뉘도록 층화 분할했고, `random_state=42`로 고정했다. 모델과 전처리 선택에는 학습 자료 안의 5겹 교차검증 결과를 사용했다.

## 비교한 내용

| 구분 | 비교 방법 |
|---|---|
| 수치형 결측치 | 평균, 중앙값, 최빈값 |
| 범주형 인코딩 | 원-핫 인코딩, 정수 인코딩 |
| 스케일링 | StandardScaler, MinMaxScaler, RobustScaler, 미적용 |
| 파생 변수 | 가족 수, 혼자 탑승 여부, 가족 수로 나눈 요금, 객실 기록 유무, 호칭, 나이 구간 |
| 변수 선택 | 랜덤 포레스트 중요도 상위 12개 선택 전후 |
| 모델 | 로지스틱 회귀, 랜덤 포레스트 |
| 평가 지표 | 정확도, 정밀도, 재현율, F1, ROC-AUC |

과제에서 제시한 Base와 Exp-1~3을 비교했다. 여기에 한 가지 조건만 바꾸는 실험을 추가했다. 여러 전처리를 동시에 바꾸면 어떤 방법 때문에 성능이 달라졌는지 알기 어렵기 때문이다.

추가 항목으로 Pipeline, GridSearchCV, SHAP, FLAML AutoML, 변수 중요도 시각화를 포함했다.

## 실행 순서

Python 3.11 기준이며 Windows PowerShell에서는 아래 순서로 실행하면 된다.

```powershell
git clone https://github.com/e-chang0/titanic-feature-engineering.git
cd titanic-feature-engineering
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item submission.example.json submission.json

python 데이터준비.py
python 데이터분석.py
python 모델실험.py --jobs 2 --automl-iterations 30
python 보고서생성.py
python 결과검사.py
```

`submission.json`에는 보고서에 넣을 이름과 학번을 입력한다. 이 파일은 Git에 올라가지 않도록 제외했다. 한글 그래프와 PDF를 만들려면 맑은 고딕 또는 나눔고딕이 필요하다. 설치 관련 설명은 [실행방법.md](실행방법.md)에 정리했다.

## 파일 구성

| 파일 | 하는 일 |
|---|---|
| `데이터준비.py` | 공개 데이터 다운로드 및 파일 확인 |
| `데이터분석.py` | 결측률, 분포, 이상치, 상관관계 분석 |
| `모델실험.py` | 전처리 비교, 교차검증, 모델 튜닝, 최종 평가 |
| `보고서생성.py` | 실행 결과로 한국어 PDF 생성 |
| `결과검사.py` | 데이터 분할, 평가 지표, SHAP, 보고서 검사 |
| `src/titanic_pipeline.py` | 파생 변수와 학습 파이프라인 구현 |
| `src/시각화설정.py` | 그래프에 사용할 한글 글꼴 설정 |
| `requirements.txt` | 필요한 라이브러리와 버전 |
| `requirements-lock.txt` | 처음 실험할 때 설치한 전체 의존성 버전 |

한국어 실행 파일은 기존 코드의 함수를 호출한다. `download_data.py`, `eda.py`, `run_experiments.py`, `make_report.py`, `validate_submission.py` 명령도 사용할 수 있다. 데이터 컬럼과 라이브러리 이름은 원래 이름을 유지했다.

실행하면 `results/`에 비교표와 모델, `figures/`에 EDA 그림, `output/pdf/`에 보고서가 생성된다. 공개 저장소에는 코드와 설명을 올리고, 개인 설정과 생성 파일은 제외했다.

## 데이터 출처

- [Kaggle Titanic 데이터 설명](https://www.kaggle.com/competitions/titanic/data)
- [실제 다운로드에 사용한 공개 사본](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv)
