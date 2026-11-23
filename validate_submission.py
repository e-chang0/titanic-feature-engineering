"""저장한 결과의 분할, 점수, 예측, SHAP, 최종 PDF를 검사한다."""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from pypdf import PdfReader
from src.titanic_pipeline import METRICS, MAIN, MODELS, metrics, feature_names, TitanicFeatures
from src.submission_config import load_submission, report_path

ROOT=Path(__file__).resolve().parent

def main():
    checks=[]
    def require(condition,label):
        if not condition:raise AssertionError(label)
        checks.append(label)
    raw=pd.read_csv(ROOT/'data/train.csv')
    require(raw.shape==(891,12),'원본 데이터 크기: 891행, 12열')
    require(hashlib.sha256((ROOT/'data/train.csv').read_bytes()).hexdigest()=='4a437fde05fe5264e1701a7387ac6fb75393772ba38bb2c9c566405af5af4bd7','원본 데이터의 SHA256 일치')
    summary=json.loads((ROOT/'results/summary.json').read_text(encoding='utf-8'))
    require(summary['status']=='complete','전체 실험 완료 여부')
    split=pd.read_csv(ROOT/'results/split_assignment.csv')
    train_ids=set(split.loc[split.partition=='train','PassengerId']);test_ids=set(split.loc[split.partition=='test','PassengerId'])
    require(len(train_ids)==712 and len(test_ids)==179 and train_ids.isdisjoint(test_ids),'학습 712명과 평가 179명의 승객 번호가 겹치지 않음')
    require(train_ids|test_ids==set(raw.PassengerId),'모든 승객이 한 번씩 분할됨')
    require(set(split.loc[split.partition=='train','validation_fold'])=={1,2,3,4,5} and split.loc[split.partition=='test','validation_fold'].isna().all(),'교차검증은 학습 자료 안에서만 수행')
    cv=pd.read_csv(ROOT/'results/cv_results.csv');folds=pd.read_csv(ROOT/'results/cv_fold_scores.csv')
    require(len(cv)==24 and len(folds)==120,'24개 실험·모델 조합과 120개 검증 구간 기록')
    require(set(zip(cv.loc[cv.experiment.isin(MAIN),'experiment'],cv.loc[cv.experiment.isin(MAIN),'model']))=={(a,b) for a in MAIN for b in MODELS},'필수 실험 4개 × 모델 2개 확인')
    for row in cv.itertuples():
        subset=folds[(folds.experiment==row.experiment)&(folds.model==row.model)]
        require(len(subset)==5,f'검증 구간 5개 확인: {row.experiment}/{row.model}')
        for m in METRICS:
            require(np.isclose(subset['validation_'+m].mean(),getattr(row,'cv_'+m+'_mean'),atol=1e-12),f'교차검증 평균 일치: {row.experiment}/{row.model}/{m}')
    metric_cols=[f'cv_{m}_mean' for m in METRICS]
    require(np.isfinite(cv[metric_cols]).all().all() and cv[metric_cols].ge(0).all().all() and cv[metric_cols].le(1).all().all(),'교차검증 지표가 유효한 0~1 범위의 값임')
    am=json.loads((ROOT/'results/automl_summary.json').read_text(encoding='utf-8'))
    ai=set(am['inner_train_passenger_ids']);av=set(am['inner_validation_passenger_ids'])
    require(ai.isdisjoint(av) and ai|av==train_ids,'AutoML 내부 분할에 최종 평가 자료가 들어가지 않음')
    require(am['logged_trials']==30 and len(am['estimators'])>=2,'AutoML 탐색 30회와 여러 후보 알고리즘 확인')
    grid=pd.read_csv(ROOT/'results/grid_search_results.csv')
    require(len(grid)==18 and set(grid.model)==set(MODELS),'GridSearchCV: 두 모델의 파라미터 후보 총 18개')
    manifest=json.loads((ROOT/'results/frozen_model_manifest.json').read_text(encoding='utf-8'))
    chosen='Tuned / '+max(MODELS,key=lambda m:summary['tuning'][m]['best_cv_roc_auc'])
    require(manifest['test_not_used_for_selection'] and chosen==summary['selected_primary_model']==manifest['primary_model'],'대표 모델이 학습 CV 기준의 선정 규칙과 일치')
    pred=pd.read_csv(ROOT/'results/test_predictions.csv');table=pd.read_csv(ROOT/'results/test_results.csv')
    require(set(pred.PassengerId)==test_ids and len(pred)==179 and len(table)==11,'설정을 고정한 11개 모델이 같은 179명으로 평가됨')
    X=raw.set_index('PassengerId').loc[pred.PassengerId].reset_index().drop(columns='Survived')
    require(np.array_equal(pred.Survived,raw.set_index('PassengerId').loc[pred.PassengerId,'Survived']),'저장한 정답이 원자료의 승객별 정답과 일치')
    for row in table.itertuples():
        pipe=joblib.load(ROOT/'results/models'/f"{row.key.replace(' / ','__')}.joblib")
        prob=pipe.predict_proba(X)[:,1]
        require(np.allclose(prob,pred[row.key+' probability'],atol=1e-12),f'저장 모델의 예측 확률 재현: {row.key}')
        recomputed=metrics(pred.Survived,prob)
        require(all(np.isclose(recomputed[m],getattr(row,m),atol=1e-12) for m in METRICS),f'평가 지표 5개 재계산 일치: {row.key}')
        require(len(feature_names(pipe))==row.n_features,f'입력 변수 수 일치: {row.key}')
    # 학습 때 없던 범주와 결측 나이를 넣어도 예측이 되는지 확인한다.
    reference=joblib.load(ROOT/'results/models/Exp-1__LogisticRegression.joblib')
    edge=X.iloc[:3].copy();edge['Embarked']='UNSEEN';edge['Age']=np.nan
    require(np.isfinite(reference.predict_proba(edge)).all(),'새 범주와 Age 결측이 있어도 파이프라인이 예측 가능')
    original=X.iloc[:3].copy(deep=True);TitanicFeatures().transform(original)
    require(original.equals(X.iloc[:3]),'파생 변수 생성 시 원본 입력을 변경하지 않음')
    sv=np.load(ROOT/'results/shap_values.npz',allow_pickle=False)
    rf=joblib.load(ROOT/'results/models/Tuned__RandomForest.joblib')
    require(np.allclose(sv['base_values']+sv['values'].sum(axis=1),rf.predict_proba(X)[:,1],atol=1e-5),'SHAP 합산값과 고정한 RF 생존 확률 일치')
    require(list(sv['feature_names'])==feature_names(rf).tolist(),'SHAP 변수 순서와 모델 입력 순서 일치')
    for relative in ['figures/eda_histograms.png','figures/eda_boxplots.png','figures/eda_heatmap.png','figures/eda_counts.png','figures/eda_missing.png',
        'results/figures/shap_beeswarm.png','results/figures/shap_waterfall.png','results/figures/importance_comparison.png','results/figures/heldout_curves.png']:
        require((ROOT/relative).stat().st_size>1000,'그래프 파일 확인: '+relative)
    submission=load_submission(ROOT)
    pdf=report_path(ROOT,submission)
    reader=PdfReader(pdf);text='\n'.join(p.extract_text() or '' for p in reader.pages)
    require(len(reader.pages)>=18,'최종 PDF의 전체 보고서 분량 확인')
    for token in [submission['student_id'],submission['student_name'],'EDA','GridSearchCV','SHAP','AutoML','ROC-AUC','최종 결론']:
        require(token in text,'PDF 본문에서 확인: '+token)
    require('\ufffd' not in text and '\u25a0' not in text,'PDF 추출문에 대체 문자나 검은 사각형이 없음')
    if submission.get('github_url'):
        url=submission['github_url']
        require(url in text,'PDF에 제출용 소스코드 주소 포함')
        urls=[]
        for page in reader.pages:
            for reference in page.get('/Annots',[]):
                annotation=reference.get_object()
                action=annotation.get('/A',{})
                if action.get('/URI'):urls.append(str(action['/URI']))
        require(urls.count(url)>=2,'처음과 마지막 소스코드 주소가 클릭 가능한 PDF 링크임')
    result={'status':'passed','checks_passed':len(checks),'pdf_pages':len(reader.pages),'checks':checks}
    (ROOT/'results/validation_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"검사 통과: {len(checks)}개 항목, PDF {len(reader.pages)}쪽")

if __name__=='__main__':main()
