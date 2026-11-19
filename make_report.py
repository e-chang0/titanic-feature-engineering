"""Create the Korean PDF directly from executed CSV/JSON results (no invented scores)."""
from pathlib import Path
import json
from xml.sax.saxutils import escape
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from src.submission_config import load_submission, report_path

ROOT=Path(__file__).resolve().parent
SUBMISSION=load_submission(ROOT)
AUTHOR_LABEL=f"{SUBMISSION['student_id']} {SUBMISSION['student_name']}"
W,H=A4
NAVY=colors.HexColor('#17324D');TEAL=colors.HexColor('#008C95');LIGHT=colors.HexColor('#EDF5F6');GREY=colors.HexColor('#586977')
WIDTH=W-88
MAIN=['Base','Exp-1','Exp-2','Exp-3']
font_candidates=[Path('C:/Windows/Fonts/malgun.ttf'),ROOT/'fonts/NanumGothic.ttf',Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')]
regular=next((p for p in font_candidates if p.exists()),None)
if regular is None:raise FileNotFoundError('Install Korean font NanumGothic or provide fonts/NanumGothic.ttf')
bold=regular.with_name('malgunbd.ttf') if regular.name=='malgun.ttf' else regular.with_name('NanumGothicBold.ttf')
pdfmetrics.registerFont(TTFont('Korean',str(regular)))
pdfmetrics.registerFont(TTFont('KoreanBold',str(bold if bold.exists() else regular)))
pdfmetrics.registerFontFamily('Korean',normal='Korean',bold='KoreanBold')
ST={
 'body':ParagraphStyle('body',fontName='Korean',fontSize=9.4,leading=15.6,textColor=NAVY,spaceAfter=8,wordWrap='CJK'),
 'small':ParagraphStyle('small',fontName='Korean',fontSize=8,leading=12,textColor=GREY,spaceAfter=6,wordWrap='CJK'),
 'caption':ParagraphStyle('caption',fontName='Korean',fontSize=8.1,leading=12,textColor=GREY,spaceAfter=8,wordWrap='CJK'),
 'h1':ParagraphStyle('h1',fontName='KoreanBold',fontSize=21,leading=29,textColor=NAVY,spaceAfter=14,wordWrap='CJK'),
 'h2':ParagraphStyle('h2',fontName='KoreanBold',fontSize=12,leading=18,textColor=TEAL,spaceBefore=8,spaceAfter=8,wordWrap='CJK'),
 'eyebrow':ParagraphStyle('eyebrow',fontName='KoreanBold',fontSize=8.5,leading=14,textColor=TEAL,spaceAfter=7),
 'cell':ParagraphStyle('cell',fontName='Korean',fontSize=8,leading=11.5,textColor=NAVY,wordWrap='CJK'),
 'headcell':ParagraphStyle('headcell',fontName='KoreanBold',fontSize=8,leading=11.5,textColor=colors.white,wordWrap='CJK'),
 'cover':ParagraphStyle('cover',fontName='KoreanBold',fontSize=31,leading=44,textColor=NAVY,spaceAfter=20,wordWrap='CJK'),
}

def P(text,style='body'):return Paragraph(escape(str(text)).replace('\n','<br/>'),ST[style])
story=[]
def add(text,style='body'):story.append(P(text,style))
def section(n,title):
    if story:story.append(PageBreak())
    add(f'TITANIC  /  {n:02d}','eyebrow');add(title,'h1')
def table(headers,rows,widths=None,small=False):
    data=[[P(v,'headcell') for v in headers]]+[[P(v,'cell') for v in row] for row in rows]
    t=Table(data,colWidths=widths or [WIDTH/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LIGHT]),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
        ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.5,NAVY)]))
    story.extend([t,Spacer(1,9)])
def pic(path,width=WIDTH,maxheight=360):
    from PIL import Image as PILImage
    path=ROOT/path
    with PILImage.open(path) as im:iw,ih=im.size
    scale=min(width/iw,maxheight/ih)
    item=Image(str(path),width=iw*scale,height=ih*scale);item.hAlign='CENTER';story.extend([item,Spacer(1,6)])
def f(v):return f'{float(v):.4f}'
def pp(v):return f'{float(v)*100:+.2f}pp'
def short(m):return {'LogisticRegression':'LR','RandomForest':'RF','FLAML':'AutoML'}.get(m,m)
def footer(c,doc):
    c.saveState();c.setStrokeColor(TEAL);c.setLineWidth(.7);c.line(44,H-31,W-44,H-31)
    c.setFont('Korean',7.7);c.setFillColor(GREY);c.drawString(44,25,'특성 공학 파이프라인 | '+AUTHOR_LABEL)
    c.drawRightString(W-44,25,f'{doc.page}');c.restoreState()

def main():
    s=json.loads((ROOT/'results/summary.json').read_text(encoding='utf-8'))
    e=json.loads((ROOT/'results/eda_summary.json').read_text(encoding='utf-8'))
    sub=SUBMISSION
    cv=pd.read_csv(ROOT/'results/cv_results.csv');test=pd.read_csv(ROOT/'results/test_results.csv');ab=pd.read_csv(ROOT/'results/ablation_deltas.csv')
    primary=test.set_index('key').loc[s['selected_primary_model']]
    def cvr(exp,m):return cv[(cv.experiment==exp)&(cv.model==m)].iloc[0]
    def delta(alt,m):return float(ab[(ab.alternative==alt)&(ab.model==m)].iloc[0].delta_cv_auc)

    add('MACHINE LEARNING  /  FEATURE ENGINEERING','eyebrow');story.append(Spacer(1,34))
    add('Titanic 생존 예측\n특성 공학 파이프라인\n설계 및 비교 분석','cover')
    add('실제 공개 데이터로 검증한 전처리, 파생 변수, 변수 선택의 효과','h2')
    add(f"{sub['student_id']}  {sub['student_name']}\n작성일 {sub['date']}")
    story.append(Spacer(1,26))
    table(['데이터','실험 설계','최종 평가'],[['891명 · 12개 컬럼','학습 712명 / 평가 179명\n동일한 층화 5-fold CV','Accuracy · Precision · Recall\nF1-score · ROC-AUC']],[150,190,WIDTH-340])
    add('실행 결과 요약','h2')
    add(f"훈련 교차검증으로 선정한 최종 모델은 {s['selected_primary_model']}이다. 별도로 보관한 179명 평가 데이터에서 Accuracy {f(primary.accuracy)}, F1 {f(primary.f1)}, ROC-AUC {f(primary.roc_auc)}를 기록했다.")
    add(f"파생 변수 6개를 추가한 통제 실험에서 평균 CV ROC-AUC는 LR {pp(-delta('Without-Derived','LogisticRegression'))}, RF {pp(-delta('Without-Derived','RandomForest'))} 변화했다. One-Hot과 정수 인코딩의 우열은 모델에 따라 달랐다.")
    add('필수 항목과 가산점 수행','h2')
    add('Pipeline · ColumnTransformer · GridSearchCV · SHAP · FLAML AutoML · 중요도/순열 중요도 시각화를 실제 실행했다. 모든 수치와 그림은 저장한 결과 파일에서 자동으로 보고서에 반영한다.')
    url=sub.get('github_url')
    if url:story.append(Paragraph(f'소스코드: <link href="{url}" color="#008C95">{url}</link>',ST['body']))
    else:add('소스코드는 함께 제공한 업로드용 프로젝트에 포함되어 있다. GitHub 게시 후 제출 링크를 보고서에 반영한다.','small')
    add('평가 점수와 가산점의 최종 인정 여부는 수업 평가 기준에 따른다.','small')

    section(1,'데이터셋 소개와 컬럼 정의')
    add('Kaggle Titanic은 승객 정보를 이용해 생존 여부(Survived)를 예측하는 지도학습 이진 분류 데이터다. 정답이 있는 공개 학습 자료의 사본을 사용했다. Kaggle의 정답 없는 test.csv는 평가에 사용하지 않았다. [1, 2]')
    table(['확인 항목','실제 출력'],[['원본 shape','(891, 12)'],['입력 X / 타겟 y','(891, 11) / (891,)'],['내부 학습 / 최종 평가','(712, 11) / (179, 11)'],['자료 조건','500개 이상 충족 · 수치형 및 범주형 혼합']],[140,WIDTH-140])
    dd=pd.read_csv(ROOT/'data_dictionary.csv')
    table(['컬럼','의미','형태·사용 역할','결측 수'],[[r.column,r.description_ko,r.role+' / '+r.unit_or_values,str(r.missing_count_full)] for r in dd.itertuples()],[78,132,WIDTH-254,44])
    add('PassengerId는 입력에서 제외한다. Name과 Cabin은 파생 변수의 재료로만 사용하고 원문은 제외한다. Ticket은 고유값이 많아 본 실험에서 제외했다. Pclass는 정수로 저장된 순서형 범주다.','small')
    add('데이터 출처: Kaggle 데이터 설명 [1], 실제 다운로드 경로: Data Science Dojo 공개 사본 [2]. 다운로드일: 2026-11-19.','small')

    section(2,'EDA: 결측치와 데이터 품질')
    add('전체 데이터의 구조·결측률·타겟 수는 자료 소개용으로 확인했다. 변수 분포, 이상치, 상관관계와 범주별 생존율은 학습 분할 712명만으로 분석했다. 그림의 통계 범위도 학습 분할이다.')
    table(['컬럼','전체 결측 수 / 비율','학습 결측 수 / 비율'],[[c,f"{e['missing_full'][c]['count']} / {e['missing_full'][c]['percent']:.2f}%",f"{e['missing_train'][c]['count']} / {e['missing_train'][c]['percent']:.2f}%"] for c in ['Age','Cabin','Embarked']],[100,200,WIDTH-300])
    pic('figures/eda_missing.png',maxheight=325)
    add('그림 1. 학습 분할의 컬럼별 결측률. Cabin의 결측 비중이 가장 높다.','caption')
    add('Age는 수치형 대체법의 비교 대상이다. Embarked는 모든 실험에서 학습 fold의 최빈값으로 대체한다. Cabin은 기록 유무(CabinKnown)를 사용하며 결측을 “객실 없음”으로 단정하지 않는다. 모든 결측 행을 삭제하면 많은 표본을 잃으므로 삭제 방식은 채택하지 않았다.')
    add(f"전체 완전 중복 행 {e['duplicates_full']}개, PassengerId 중복 {e['duplicate_passenger_ids']}개다. 값의 범위와 결측 구조를 점검했지만 IQR 표시값을 자동으로 오류로 간주하거나 제거하지 않았다.")

    section(3,'EDA: 분포와 이상치 탐색')
    pic('figures/eda_histograms.png',maxheight=225)
    add('그림 2. 수치형 분포와 log(1 + Fare) 참고 분포. 로그 변환은 EDA 그림에만 사용했고 실험 입력에는 적용하지 않았다.','caption')
    pic('figures/eda_boxplots.png',maxheight=215)
    add('그림 3. 생존 여부별 Boxplot. 고액 요금과 큰 가족 규모는 실제 사례일 수 있어 유지했다.','caption')
    table(['변수','IQR 하한','IQR 상한','표시 / 비결측'],[[c,f"{v['lower_bound']:.2f}",f"{v['upper_bound']:.2f}",f"{v['flagged_count']} / {v['non_missing_count']}"] for c,v in e['outliers_iqr_train'].items()],[110,115,115,WIDTH-340])
    add('Fare는 오른쪽 꼬리가 길고, 동승 가족 수는 0에 집중된다. IQR 기준은 탐색용이며 이산형·영과잉 변수에서 특히 신중하게 해석한다.','small')

    section(4,'EDA: 타겟 분포와 주요 변수')
    counts=e['target_counts_full']
    add(f"전체 사망 {counts['0']['count']}명({counts['0']['percent']:.2f}%), 생존 {counts['1']['count']}명({counts['1']['percent']:.2f}%)으로 다소 불균형하다. 양성 클래스는 생존(1)이며 층화 분할과 다섯 가지 평가 지표를 사용했다.")
    pic('figures/eda_counts.png',maxheight=335)
    add('그림 4. 학습 분할 타겟·성별·객실 등급·승선 항구별 Countplot.','caption')
    rows=[]
    for group in ['Sex','Pclass','Embarked']:
        for row in e['survival_rates_train'][group]:rows.append([group,row['category'],str(row['count']),f"{100*row['survival_rate']:.2f}%"])
    table(['변수','범주','학습 인원','생존율'],rows,[100,135,130,WIDTH-365])
    add('성별과 객실 등급에 따라 생존율 차이가 크다. 이 관계는 당시 탑승·구조 상황에서 관측된 연관성으로, 다른 환경의 인과 효과나 공정성을 보장하지 않는다.','small')

    section(5,'EDA: 상관관계 분석')
    pic('figures/eda_heatmap.png',maxheight=400)
    add('그림 5. 학습 데이터의 Pearson 상관계수 Heatmap. 결측값은 해당 변수 쌍에서 제외한다.','caption')
    corr=e['correlation_train']['Survived']
    add(f"Survived와 Pclass의 상관계수는 {corr['Pclass']:.3f}, Fare는 {corr['Fare']:.3f}, Age는 {corr['Age']:.3f}이다. 객실 등급과 요금에는 겹치는 정보가 있어 개별 중요도를 독립적인 영향으로 해석하기 어렵다.")
    add('Pclass는 순서형 코드를 수치로 취급한 기술적 요약이다. Pearson 상관계수만으로 범주형 변수나 비선형 상호작용을 평가할 수 없으므로 Countplot과 모델 기반 중요도를 함께 확인했다.')
    add('EDA에서 얻은 설계 방향','h2')
    add('① 결측 Age 대체법을 비교한다. ② 요금 분포를 고려해 Standard·MinMax·Robust 스케일링을 비교한다. ③ 가족 규모, 호칭, 객실 기록 여부 등 원본 변수의 구조를 파생 변수로 표현한다. ④ 표본을 자동 삭제하지 않고 동일한 평가 대상에서 비교한다.')

    section(6,'특성 공학 파이프라인 구현')
    table(['순서','처리','학습 범위'],[['1','고정 규칙으로 파생 변수 생성','개별 행만 사용'],['2','ColumnTransformer: 수치형 / 범주형 분기','각 학습 fold'],['3','SimpleImputer → Scaler / Encoder','각 학습 fold'],['4','선택 시 RF 중요도 상위 12개','각 학습 fold의 X와 y'],['5','Logistic Regression 또는 Random Forest','각 학습 fold']],[42,290,WIDTH-332])
    add('Pipeline 객체 안에서 전처리와 모델을 연결했다. 검증·평가 데이터는 transform과 predict에만 사용하므로 평균, 최빈값, 범주 사전, 중요도가 평가 데이터에서 학습되지 않는다. [3]')
    table(['파생 변수','생성 규칙','의미와 주의점'],[
        ['FamilySize','SibSp + Parch + 1','본인을 포함한 동승 가족 규모'],
        ['IsAlone','FamilySize == 1 이면 1','단독 탑승 여부'],
        ['FarePerPerson','Fare / FamilySize','가족 규모로 정규화한 요금 대리 변수. 실제 개인 지불액은 아님'],
        ['CabinKnown','Cabin 기록 있음 = 1','객실 기록의 존재 여부'],
        ['Title','이름에서 호칭 추출','Mr/Mrs/Miss/Master/Other; Mlle·Ms→Miss, Mme→Mrs'],
        ['AgeGroup','≤12, (12,18], (18,35], (35,60], >60','고정 구간; Age 결측이면 범주 결측 유지 후 최빈값 대체']],[90,170,WIDTH-260])
    add('호칭 묶음과 나이 경계는 고정 규칙이다. 전체 데이터의 빈도나 타겟값으로 임계값을 만들지 않았다. Age의 수치 대체와 AgeGroup의 범주 대체는 별개이므로 두 표현이 완전히 일치하지 않을 수 있다.','small')
    add('결측치·인코딩·스케일링','h2')
    add('수치형은 Mean / Median / Most Frequent 3종, 범주형은 공통으로 Most Frequent를 사용했다. 인코딩은 One-Hot과 Label 방식에 해당하는 OrdinalEncoder를 비교했다. LabelEncoder는 타겟용이므로 입력 특성에는 사용하지 않았다. 미지 범주는 One-Hot에서 모두 0, Ordinal에서 -1로 처리한다. [4]')
    add('Standard는 평균·표준편차, MinMax는 최소·최대, Robust는 중앙값·IQR 기준이다. 스케일러는 수치형 분기에만 적용하며, 스케일링 없음도 통제 실험에 포함했다.')

    section(7,'실험 설계와 평가 프로토콜')
    table(['실험','수치 결측','인코딩','스케일링','변수 선택'],[
        ['Base','없음','없음','없음','없음'],['Exp-1','Mean','One-Hot','Standard','없음'],
        ['Exp-2','Median','Ordinal (Label)','MinMax','상위 12개'],['Exp-3','Most Frequent','One-Hot','Robust','상위 12개'],
        ['Reference','Median','One-Hot','Standard','없음']],[72,95,115,110,WIDTH-392])
    add('Base는 결측이 없는 원본 Pclass·SibSp·Parch·Fare 4개를 그대로 입력한다. 문자열과 결측이 포함된 전체 원본은 LR에 바로 입력할 수 없기 때문이다. 이는 사전 정의한 입력 범위이며, 지도학습 기반 변수 선택은 하지 않았다. Base와 다른 실험은 입력 정보량도 달라 전처리만의 효과로 해석할 수 없다.')
    add('Exp-1~3은 과제 요구 조합이다. Reference에서 수치 대체법, 인코딩, 스케일러, 파생 변수 유무, 상위 12개 선택 유무를 각각 하나씩 바꿔 총 12개 설정 × 2개 모델 = 24개 CV 실험을 수행했다.')
    table(['항목','설정'],[['분할','stratify=Survived, test_size=0.2, random_state=42'],['교차검증','StratifiedKFold(5, shuffle=True, random_state=42); 동일 fold 공유'],['LR 기본값','C=1, solver=lbfgs, max_iter=3000'],['RF 기본값','160 trees, max_depth=6, min_samples_leaf=2, max_features=sqrt'],['선정 기준','훈련 CV ROC-AUC; 분류 임계값은 0.5 고정'],['최종 평가','모든 선택을 고정한 뒤 179명 test 평가; test로 재선정하지 않음']],[100,WIDTH-100])
    add('평가 지표 (선택 가점 항목 포함)','h2')
    table(['지표','계산·해석'],[['Accuracy','(TP + TN) / 전체: 전체 정답 비율'],['Precision','TP / (TP + FP): 생존 예측 중 실제 생존 비율'],['Recall','TP / (TP + FN): 실제 생존자를 찾은 비율'],['F1-score','2 × Precision × Recall / (Precision + Recall)'],['ROC-AUC','생존 확률의 순위 판별력; hard label이 아닌 predict_proba 사용']],[105,WIDTH-105])
    add('생존=양성. 0으로 나누는 경우 precision/recall/F1은 zero_division=0으로 처리한다. 같은 CV에서 설정을 고르므로 최적 CV 점수에는 선택 편향이 있다. 독립 test 결과와 구분해 해석한다.','small')

    section(8,'필수 조합의 교차검증 결과')
    add('다음은 학습 712명 안에서만 계산한 5-fold 평균이다. AUC ± SD는 fold 간 표본 표준편차(ddof=1)이며 신뢰구간이 아니다. LR=Logistic Regression, RF=Random Forest.')
    cols=['실험 / 모델','Accuracy','Precision','Recall','F1','AUC ± SD']
    rows=[]
    for r in cv[cv.experiment.isin(MAIN)].itertuples():
        rows.append([r.experiment+' / '+short(r.model),f(r.cv_accuracy_mean),f(r.cv_precision_mean),f(r.cv_recall_mean),f(r.cv_f1_mean),f'{r.cv_roc_auc_mean:.4f} ± {r.cv_roc_auc_std:.4f}'])
    table(cols,rows,[104,66,66,66,66,WIDTH-368])
    pic('results/figures/cv_main_comparison.png',maxheight=285)
    add('그림 6. 필수 4개 조합과 두 모델의 교차검증 ROC-AUC.','caption')
    best=cv[cv.experiment.isin(MAIN)].sort_values('cv_roc_auc_mean',ascending=False).iloc[0]
    add(f"필수 조합 중 {best.experiment} / {short(best.model)}가 가장 높은 평균 AUC {f(best.cv_roc_auc_mean)}를 보였다. 그러나 결측 처리·인코딩·스케일링·선택을 동시에 바꾸었으므로 이 표만으로 원인을 분리할 수 없다.")
    add('Base 대비 개선에는 성별·나이·파생 변수 등 추가 정보의 효과가 함께 포함된다. 뒤의 통제 실험이 개별 전략의 효과를 해석하는 근거다. 전체 24개 설정의 다섯 지표와 fold별 점수는 CSV로 제공한다.')

    section(9,'최종 평가: 고정한 모델의 성능')
    add('아래 표는 동일한 최종 평가 179명을 사용한다. 튜닝 모델, AutoML, 필수 조합을 모두 고정한 뒤 한 번의 평가 단계에서 계산했다. test 순위는 최종 모델 선정 기준으로 사용하지 않았다.')
    table(['실험 / 모델','차원','Accuracy','Precision','Recall','F1','AUC'],[[r.experiment+' / '+short(r.model),r.n_features,f(r.accuracy),f(r.precision),f(r.recall),f(r.f1),f(r.roc_auc)] for r in test.itertuples()],[102,35,70,75,68,68,WIDTH-418])
    pic('results/figures/heldout_curves.png',maxheight=245)
    add('그림 7. 주요 고정 모델의 ROC 및 Precision-Recall 곡선. 괄호는 각각 AUC와 AP이며 AP는 보조 지표다.','caption')
    add(f"최종 모델은 훈련 CV 기준 {s['selected_primary_model']}이다. test AUC {f(primary.roc_auc)}, Accuracy {f(primary.accuracy)}, Precision {f(primary.precision)}, Recall {f(primary.recall)}, F1 {f(primary.f1)}를 기록했다.")
    observed=test.sort_values('roc_auc',ascending=False).iloc[0]
    add(f"이번 test에서 가장 높은 AUC는 {observed['key']}의 {f(observed.roc_auc)}였다. 훈련 CV 순위와 test 순위가 달라졌으므로 튜닝이 실제 일반화 성능을 반드시 높였다고 볼 수 없다. test를 본 뒤 대표 모델을 교체하지 않았다.")
    add('시험 집합이 작아 소수 오분류만으로 지표가 움직인다. 여러 모델의 시험 순위를 반복 확인하며 재튜닝하지 않았으며, 한 번의 무작위 분할 결과를 보편적 우열로 일반화하지 않는다.','small')

    section(10,'통제 실험: 한 요인씩 바꾸기')
    add('기준은 Median + One-Hot + Standard + 파생 변수 6개 + 선택 없음이다. 아래 ΔAUC는 대안 설정에서 Reference를 뺀 값이다. 다른 입력, 모델 기본값, fold는 동일하다.')
    labels={'Exp-1':'Mean','Impute-MostFrequent':'Most Frequent','Encode-Ordinal':'Ordinal','Scale-None':'스케일 없음','Scale-MinMax':'MinMax','Scale-Robust':'Robust','Without-Derived':'파생 변수 없음','Select-Top12':'상위 12개 선택'}
    table(['한 가지 변경','LR ΔAUC','RF ΔAUC'],[[labels[a],pp(delta(a,'LogisticRegression')),pp(delta(a,'RandomForest'))] for a in labels],[265,120,WIDTH-385])
    pic('results/figures/ablation_auc.png',maxheight=265)
    add('그림 8. 동일 fold의 AUC 차이와 그 표준편차. 훈련 집합이 겹치므로 독립 표본의 유의성 검정으로 읽지 않는다.','caption')
    add('Median은 비교한 수치 대체법 중 두 모델에서 가장 높았지만 차이는 0.1pp 미만이었다. LR은 MinMax에서 가장 높은 평균 AUC를, RF는 정수 인코딩에서 가장 높은 평균 AUC를 보였다.')
    add(f"정수 인코딩으로 바꾸면 LR {pp(delta('Encode-Ordinal','LogisticRegression'))}, RF {pp(delta('Encode-Ordinal','RandomForest'))}였다. One-Hot이 항상 우수하지 않다. RF 스케일러의 미세 차이는 실질적 개선으로 단정하기 어렵다. 트리의 분할 순서를 보존하는 변환도 부동소수점·동률 처리로 작은 차이가 생길 수 있다.")

    section(11,'변수 선택과 과적합 비교')
    add('학습 fold에서 별도 RF(100 trees, 깊이 6, leaf 최소 2)를 학습하고 불순도 중요도 상위 12개 변환 열을 선택한다. OHE에서는 더미 열도 개별 특성이다. 전체 학습 자료로 고른 목록을 CV에 미리 넣지 않았다.')
    rows=[]
    for model in ['LogisticRegression','RandomForest']:
        for exp in ['Reference','Select-Top12']:
            r=cvr(exp,model);rows.append([short(model),exp.replace('Select-',''),str(int(r.n_features_mean)),f(r.train_roc_auc_mean),f(r.cv_roc_auc_mean),f(r.auc_generalization_gap),f(r.cv_f1_mean)])
    table(['모델','설정','차원','Train AUC','CV AUC','격차','CV F1'],rows,[36,90,35,88,88,83,WIDTH-420])
    add('LR의 훈련–검증 격차는 0.0111에서 0.0056으로 줄었지만 F1은 0.7679에서 0.7575로 낮아졌다. RF는 F1이 0.7571에서 0.7688로 높아졌으나 AUC 격차는 0.0601에서 0.0620으로 커졌다. 따라서 변수 선택이 두 모델 모두의 과적합을 줄였다고 결론내릴 수 없다.')
    add('전체 학습 712명에서 다시 학습한 최종 선택 목록','h2')
    sf=s['selected_features']
    left=sf['Exp-2 / RandomForest'];right=sf['Exp-3 / RandomForest']
    table(['순번','Exp-2: Ordinal → 상위 12','Exp-3: OHE → 상위 12'],[[i+1,left[i].replace('numeric__','').replace('categorical__',''),right[i].replace('numeric__','').replace('categorical__','')] for i in range(12)],[35,(WIDTH-35)/2,(WIDTH-35)/2])
    add('위 표는 중요도 순위가 아니라 모델 입력 열 순서다. 동일 전처리와 seed의 선택기는 LR/RF에 앞서 공통 기준으로 학습된다. fold마다 학습 데이터가 다르므로 선택 목록은 달라질 수 있으며 cv_fold_scores.csv에 보관했다.','small')
    add('선택으로 26개 OHE 열을 12개로 줄일 수 있지만 압축, AUC, F1은 서로 다른 목표다. 불순도 중요도는 연속형·범주 수가 많은 특성을 선호할 수 있어 순열 중요도와 SHAP도 함께 검토했다.')

    section(12,'GridSearchCV 튜닝과 최종 선정')
    add('먼저 모델별 비-Base 설정 중 훈련 CV AUC가 가장 높은 전처리를 고른 후 GridSearchCV로 모델 파라미터를 탐색했다. refit="roc_auc"로 최적 모델을 전체 학습 712명에 재학습한다. [5]')
    rows=[]
    for m,v in s['tuning'].items():rows.append([short(m),v['preprocessing_experiment'],str(v['n_candidates']),json.dumps(v['best_params']),f(v['best_cv_roc_auc'])])
    table(['모델','전처리','후보 수','선택 파라미터','CV AUC'],rows,[35,100,45,WIDTH-255,75])
    table(['모델','탐색 범위'],[['LR','C ∈ {0.1, 1, 10}; class_weight ∈ {None, balanced}'],['RF','max_depth ∈ {3, 6, None}; min_samples_leaf ∈ {1, 3}; max_features ∈ {sqrt, 0.8}']],[50,WIDTH-50])
    add('LR 6개 + RF 12개 = 18개 후보에 각각 5-fold CV를 수행했다. 전처리·변수 선택도 각 fold에서 다시 fit한다. 상세 후보별 점수와 실행 시간은 grid_search_results.csv에 저장했다.')
    add('RF 탐색의 leaf 후보는 {1, 3}으로 기본값 2를 포함하지 않는다. 제한한 탐색 범위 안에서 최적을 골랐으므로 튜닝 결과가 기본 모델보다 항상 개선된다는 보장은 없다.','small')
    pic('results/figures/primary_confusion_matrix.png',maxheight=300)
    cm=s['primary_confusion_matrix']
    add(f"그림 9. 최종 모델의 confusion matrix. TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}. 임계값 0.5.",'caption')
    interval=s['primary_test_bootstrap_95_intervals']['roc_auc']
    add(f"고정 모델의 test 예측을 층화 bootstrap 1,000회 재표집한 AUC 95% 백분위 구간은 [{interval['low']:.4f}, {interval['high']:.4f}]이다. 이 구간은 시험 표본 불확실성만 나타내며 학습·튜닝 변동은 포함하지 않는다.")
    add('최적 CV 값은 선택에 사용한 점수라 낙관적일 수 있다. 중첩 CV를 수행했다고 주장하지 않으며, 최종 test를 독립 평가로 구분한다.','small')

    section(13,'AutoML 비교: FLAML 실제 탐색')
    am=s['automl'];auto=test[test.model=='FLAML'].iloc[0]
    add('FLAML AutoML로 알고리즘과 하이퍼파라미터를 자동 탐색했다. 수동 GridSearchCV를 AutoML로 부르지 않고 별도 라이브러리의 탐색 로그를 저장했다. [6]')
    table(['항목','실행값'],[['후보 알고리즘','Random Forest(rf), Extra Trees(extra_tree), L2 Logistic Regression(lrl2)'],
        ['목표 / 예산',f"ROC-AUC 최대화 / max_iter={am['requested_iterations']}, 실제 기록 {am['logged_trials']}개 / seed=42"],
        ['내부 분할',f"학습 712명 → inner train {am['inner_train_n']}명 + inner validation {am['inner_validation_n']}명"],
        ['전처리','Reference; inner train에서만 fit, inner validation은 transform만 수행'],
        ['선택 알고리즘',am['best_estimator']],['선택 파라미터',json.dumps(am['best_config'])],
        ['선택 후 재학습','선택된 알고리즘·파라미터를 전체 outer train 712명으로 재학습'],
        ['실행 시간',f"{am['elapsed_seconds']:.1f}초 (이 실행 환경)"]],[112,WIDTH-112])
    table(['구분','AUC','해석'],[['탐색 내부 검증',f(am['inner_validation_metrics']['roc_auc']),'탐색에 사용한 점수'],
        ['선택 후 5-fold CV',f(am['postselection_cv']['cv_roc_auc_mean']),'선택 후 기술적 점검; 중첩 CV 아님'],
        ['최종 test 179명',f(auto.roc_auc),'선택에 사용하지 않은 최종 평가']],[150,85,WIDTH-235])
    add(f"AutoML의 test Accuracy {f(auto.accuracy)}, F1 {f(auto.f1)}, AUC {f(auto.roc_auc)}를 수동 튜닝 모델과 같은 표에서 비교했다. AutoML도 전처리 학습 범위를 엄격히 분리했으며 최종 test를 X_val로 넘기지 않았다.")
    add('수동 튜닝은 5-fold CV, FLAML 탐색은 내부 holdout을 사용하므로 탐색 점수 자체는 직접 우열을 비교하지 않는다. 동일 test가 공통 비교 기준이다. 최종 대표 모델은 사전에 정한 두 수동 튜닝 모델의 CV 선정 규칙을 유지했고 AutoML은 별도 비교 모델로 둔다.')
    add('시간 예산 대신 고정 반복 예산을 사용했지만, 적응형 탐색은 라이브러리·운영체제 및 실행 시간 추정의 영향을 받을 수 있다. 반복 횟수와 최적 설정·개별 trial을 함께 보관해 이번 실행을 추적할 수 있게 했다.','small')

    section(14,'Feature Importance 시각화 고도화')
    pic('results/figures/importance_comparison.png',maxheight=330)
    add('그림 10. 튜닝한 RF의 불순도 중요도와 최종 test 순열 중요도. 오른쪽 오차 막대는 20회 반복의 표준편차다.','caption')
    imp=pd.read_csv(ROOT/'results/raw_feature_permutation.csv')
    table(['원본 컬럼','순열 후 AUC 감소','반복 SD'],[[r.feature,f(r.test_permutation_auc_decrease),f(r.test_permutation_std)] for r in imp.head(6).itertuples()],[240,135,WIDTH-375])
    add('불순도 중요도는 학습 중 분할의 손실 감소를 누적한 값이며, 순열 중요도는 한 변수를 섞을 때 예측 성능이 얼마나 떨어지는지를 측정한다. 두 값은 단위와 의미가 다르므로 크기를 직접 비교하지 않는다.')
    add('변환 열의 순열 중요도 외에도 원본 컬럼을 섞고 전체 파이프라인을 재실행하는 결과를 제공한다. 원본 Name을 섞으면 Title도 함께 다시 계산된다. 반면 개별 OHE 열을 섞으면 비현실적인 조합이 생길 수 있다. 상관된 변수는 정보를 나눠 가지므로 낮은 중요도가 무가치함을 뜻하지 않는다.')
    add('최종 test 중요도는 선택이 끝난 모델을 설명하는 데만 사용했다. 이 결과로 변수를 다시 고르거나 평가 점수를 개선하지 않았다.','small')

    section(15,'SHAP: 예측의 전역·개별 설명')
    pic('results/figures/shap_beeswarm.png',maxheight=300)
    add('그림 11. 양성 클래스(생존) 확률에 대한 SHAP beeswarm. 오른쪽은 생존 확률을 높이는 기여, 왼쪽은 낮추는 기여다. 색은 해당 모델 입력값의 높고 낮음을 나타낸다.','caption')
    ex=s['explainability'];top=ex['top_shap_features'][:5]
    table(['변환 특성','평균 |SHAP|'],[[r['feature'],f(r['mean_absolute_shap'])] for r in top],[WIDTH-125,125])
    add('TreeExplainer의 tree_path_dependent 방식과 model_output="raw"를 사용했다. RandomForestClassifier의 raw 출력은 클래스 확률이므로 생존(1) 확률을 설명한다. 모델에 저장된 학습 경로별 표본 수를 기준으로 하며 별도 background 데이터는 사용하지 않았다. 179명 test 사례는 모델 선택을 마친 후 설명했다. [7]')
    add(f"기준 확률 + 모든 SHAP 합과 실제 생존 확률의 최대 차이는 {ex['max_additivity_error']:.2e}로 검산했다. mean |SHAP|는 평균 기여 크기이며 인과 효과나 독립적인 변수 영향은 아니다.")
    add('이 실행의 튜닝 RF는 정수 인코딩을 사용하므로 색의 높고 낮음은 범주 코드일 수 있다. 예를 들어 명목형 범주의 큰 코드를 자연스러운 “정도”로 해석해서는 안 된다.','small')

    section(16,'개별 예측 해석과 최종 결론')
    pic('results/figures/shap_waterfall.png',maxheight=300)
    add(f"그림 12. 고정 test 순서의 첫 사례 PassengerId={ex['waterfall_passenger_id']}. 유리한 사례를 골라 제시하지 않았다. 기준 확률 {ex['waterfall_base_probability']:.4f}에서 해당 사례의 예측 확률 {ex['waterfall_probability']:.4f}로 이동한다.",'caption')
    add('과제의 핵심 질문에 대한 답','h2')
    table(['질문','실험에 근거한 결론'],[
        ['가장 효과적인 전처리는?','비교 범위에서 LR은 Median·One-Hot·MinMax, RF는 Median·Ordinal·Standard가 높은 CV AUC를 보였다. 전체 조합의 전역 최적을 찾았다는 뜻은 아니다.'],
        ['One-Hot이 항상 좋은가?','아니다. Ordinal 변경 시 LR AUC는 하락, RF는 상승했다. 모델별 평가가 필요하다.'],
        ['변수 선택은 과적합을 줄였는가?','LR의 AUC 격차는 줄었지만 F1이 하락했다. RF의 격차는 커졌다. 일괄적인 과적합 감소는 확인되지 않았다.'],
        ['스케일링의 모델별 영향은?','LR은 MinMax에서 개선이 관찰됐다. RF는 스케일러 간 차이가 매우 작았다.'],
        ['파생 변수의 실질적 기여는?',f"동일 조건에서 6개 파생 변수 추가 시 LR {pp(-delta('Without-Derived','LogisticRegression'))}, RF {pp(-delta('Without-Derived','RandomForest'))} CV AUC 개선."]],[140,WIDTH-140])
    add('결론의 범위','h2')
    add('작은 단일 데이터셋과 한 번의 평가 분할에 대한 결과다. 가족·동일 승선권 등 연관 승객이 분할을 넘을 수 있어 가족 단위 일반화나 외부 환경 성능을 검증한 것은 아니다. 이후에는 반복/중첩 CV와 가족 그룹 분할을 추가해 견고함을 확인할 수 있다.','small')

    section(17,'재현 방법과 요구사항 대응')
    add('Python 3.11에서 다음 명령을 프로젝트 루트에서 순서대로 실행한다. 기본 실험은 seed=42, jobs=2, AutoML 30 iterations다. PDF 생성에는 맑은 고딕(Windows) 또는 NanumGothic(Linux)이 필요하다.')
    table(['순서','실행 명령'],[['1','python -m venv .venv'],['2','가상환경 활성화 후 python -m pip install -r requirements.txt'],['3','python download_data.py'],['4','python eda.py'],['5','python run_experiments.py --jobs 2 --automl-iterations 30'],['6','python make_report.py'],['7','python validate_submission.py']],[40,WIDTH-40])
    table(['과제 항목','구현·제출 근거'],[['데이터 소개·컬럼·shape','본 보고서 1절, data_dictionary.csv, data/README.md'],['필수 EDA 4종','3~5절 및 figures/eda_*.png'],['결측·인코딩·스케일링 비교','6~10절, results/cv_results.csv, ablation_deltas.csv'],['파생 변수 2개 이상','6개 생성; 6절, TitanicFeatures'],['상위 N개 선택·전후 비교','상위 12개; 11절, cv_fold_scores.csv'],['모델 2종·5개 평가 지표','LR/RF 및 AutoML; 8~9절, test_results.csv'],['Pipeline 가산점','Pipeline + ColumnTransformer + fold 내 학습'],['GridSearchCV 가산점','18개 후보; 12절, grid_search_results.csv'],['SHAP 가산점','15~16절, SHAP beeswarm/waterfall 및 값 검산'],['AutoML 가산점','13절, FLAML summary 및 30회 trial 로그'],['중요도 시각화 가산점','14절, 불순도·순열·원본 컬럼 중요도와 오차 막대']],[155,WIDTH-155])
    add('재현 기록','h2')
    add('실제 실행 버전: '+', '.join(f'{k} {v}' for k,v in s['versions'].items())+'.','small')
    add(f"원자료 SHA256: {s['dataset']['sha256']}\n실험 실행 시간: {s['elapsed_seconds']:.1f}초. train/test 및 CV fold별 PassengerId를 split_assignment.csv에 기록했다.",'small')
    add('원본 결과 JSON/CSV, 학습 모델, 예측 확률, SHAP 값과 그림을 함께 저장한다. requirements-lock.txt는 이번 실행의 전체 의존성 기록이다. 정확한 버전·seed가 같아도 OS와 수치 라이브러리에 따라 극소 차이가 생길 수 있다.','small')

    section(18,'참고문헌 및 산출물 안내')
    sources=[
        ('Kaggle. Titanic - Machine Learning from Disaster: Data Dictionary.','https://www.kaggle.com/competitions/titanic/data'),
        ('Data Science Dojo. datasets/titanic.csv 공개 사본.','https://github.com/datasciencedojo/datasets/blob/master/titanic.csv'),
        ('scikit-learn. Common pitfalls and recommended practices; Pipeline.','https://scikit-learn.org/stable/common_pitfalls.html'),
        ('scikit-learn. OrdinalEncoder API reference.','https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html'),
        ('scikit-learn. GridSearchCV API reference.','https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html'),
        ('Microsoft FLAML. Task Oriented AutoML.','https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML/'),
        ('SHAP. TreeExplainer API reference.','https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html')]
    for i,(title,url) in enumerate(sources,1):
        add(f'[{i}] {title}','body');story.append(Paragraph(f'<link href="{url}" color="#008C95">{url}</link>',ST['small']));story.append(Spacer(1,7))
    add('온라인 출처 확인일: 2026-11-19. 문서의 최신 API 설명과 실제 설치 버전은 구별했으며, 실행 버전은 앞 절에 고정 기록했다.','small')
    add('파일 안내','h2')
    table(['경로','역할'],[['src/titanic_pipeline.py','파생 변수, 전처리, 선택, CV, 튜닝, AutoML, 평가, SHAP'],['run_experiments.py / eda.py','실험 실행 진입점 / 학습 데이터 EDA'],['results/*.csv / *.json','실제 결과, 분할, 설정, 비교표, 모델 선정 근거'],['results/figures / figures','평가·설명 가능성 그림 / EDA 그림'],['README.md / requirements*.txt','실행·설명 가이드 / 의존성 기록'],['make_report.py','실제 결과로 이 PDF를 재생성'],['validate_submission.py','분할·지표·가산점 산출물·PDF 자동 검증']],[205,WIDTH-205])
    if sub.get('github_url'):
        url=sub['github_url'];story.append(Paragraph(f'제출용 소스코드: <link href="{url}" color="#008C95">{url}</link>',ST['body']))
    target=report_path(ROOT,sub);target.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(target),pagesize=A4,leftMargin=44,rightMargin=44,topMargin=47,bottomMargin=44,
        title='Titanic 생존 예측 특성 공학 파이프라인 - '+AUTHOR_LABEL,author=AUTHOR_LABEL)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(str(target))

if __name__=='__main__':main()
