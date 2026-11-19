"""Check executed evidence, split boundaries, predictions, SHAP and final PDF."""
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
    require(raw.shape==(891,12),'Raw dataset: 891 rows, 12 columns')
    require(hashlib.sha256((ROOT/'data/train.csv').read_bytes()).hexdigest()=='4a437fde05fe5264e1701a7387ac6fb75393772ba38bb2c9c566405af5af4bd7','Original data SHA256 matches')
    summary=json.loads((ROOT/'results/summary.json').read_text(encoding='utf-8'))
    require(summary['status']=='complete','Full experiment completed')
    split=pd.read_csv(ROOT/'results/split_assignment.csv')
    train_ids=set(split.loc[split.partition=='train','PassengerId']);test_ids=set(split.loc[split.partition=='test','PassengerId'])
    require(len(train_ids)==712 and len(test_ids)==179 and train_ids.isdisjoint(test_ids),'712/179 disjoint PassengerId partitions')
    require(train_ids|test_ids==set(raw.PassengerId),'Every passenger assigned once')
    require(set(split.loc[split.partition=='train','validation_fold'])=={1,2,3,4,5} and split.loc[split.partition=='test','validation_fold'].isna().all(),'CV folds confined to training partition')
    cv=pd.read_csv(ROOT/'results/cv_results.csv');folds=pd.read_csv(ROOT/'results/cv_fold_scores.csv')
    require(len(cv)==24 and len(folds)==120,'24 configurations/models and 120 CV fold rows')
    require(set(zip(cv.loc[cv.experiment.isin(MAIN),'experiment'],cv.loc[cv.experiment.isin(MAIN),'model']))=={(a,b) for a in MAIN for b in MODELS},'Required 4 experiments x 2 models present')
    for row in cv.itertuples():
        subset=folds[(folds.experiment==row.experiment)&(folds.model==row.model)]
        require(len(subset)==5,f'Five validation folds: {row.experiment}/{row.model}')
        for m in METRICS:
            require(np.isclose(subset['validation_'+m].mean(),getattr(row,'cv_'+m+'_mean'),atol=1e-12),f'CV aggregate {row.experiment}/{row.model}/{m}')
    metric_cols=[f'cv_{m}_mean' for m in METRICS]
    require(np.isfinite(cv[metric_cols]).all().all() and cv[metric_cols].ge(0).all().all() and cv[metric_cols].le(1).all().all(),'All required CV metrics finite and within [0,1]')
    am=json.loads((ROOT/'results/automl_summary.json').read_text(encoding='utf-8'))
    ai=set(am['inner_train_passenger_ids']);av=set(am['inner_validation_passenger_ids'])
    require(ai.isdisjoint(av) and ai|av==train_ids,'AutoML inner partitions exclude outer test')
    require(am['logged_trials']==30 and len(am['estimators'])>=2,'Real AutoML: 30 logged trials and multiple algorithm candidates')
    grid=pd.read_csv(ROOT/'results/grid_search_results.csv')
    require(len(grid)==18 and set(grid.model)==set(MODELS),'GridSearchCV: 18 candidates across two models')
    manifest=json.loads((ROOT/'results/frozen_model_manifest.json').read_text(encoding='utf-8'))
    chosen='Tuned / '+max(MODELS,key=lambda m:summary['tuning'][m]['best_cv_roc_auc'])
    require(manifest['test_not_used_for_selection'] and chosen==summary['selected_primary_model']==manifest['primary_model'],'Primary model follows prespecified training CV selection rule')
    pred=pd.read_csv(ROOT/'results/test_predictions.csv');table=pd.read_csv(ROOT/'results/test_results.csv')
    require(set(pred.PassengerId)==test_ids and len(pred)==179 and len(table)==11,'Eleven frozen models evaluated on same 179 passengers')
    X=raw.set_index('PassengerId').loc[pred.PassengerId].reset_index().drop(columns='Survived')
    require(np.array_equal(pred.Survived,raw.set_index('PassengerId').loc[pred.PassengerId,'Survived']),'Stored labels agree with original passengers')
    for row in table.itertuples():
        pipe=joblib.load(ROOT/'results/models'/f"{row.key.replace(' / ','__')}.joblib")
        prob=pipe.predict_proba(X)[:,1]
        require(np.allclose(prob,pred[row.key+' probability'],atol=1e-12),f'Saved model reproduces probabilities: {row.key}')
        recomputed=metrics(pred.Survived,prob)
        require(all(np.isclose(recomputed[m],getattr(row,m),atol=1e-12) for m in METRICS),f'All five metrics independently recomputed: {row.key}')
        require(len(feature_names(pipe))==row.n_features,f'Feature dimension matches: {row.key}')
    # Explicit inference edge case: unseen categories and missing numeric values.
    reference=joblib.load(ROOT/'results/models/Exp-1__LogisticRegression.joblib')
    edge=X.iloc[:3].copy();edge['Embarked']='UNSEEN';edge['Age']=np.nan
    require(np.isfinite(reference.predict_proba(edge)).all(),'Pipeline handles unseen category and missing Age at inference')
    original=X.iloc[:3].copy(deep=True);TitanicFeatures().transform(original)
    require(original.equals(X.iloc[:3]),'Feature transformer does not mutate its input')
    sv=np.load(ROOT/'results/shap_values.npz',allow_pickle=False)
    rf=joblib.load(ROOT/'results/models/Tuned__RandomForest.joblib')
    require(np.allclose(sv['base_values']+sv['values'].sum(axis=1),rf.predict_proba(X)[:,1],atol=1e-5),'SHAP reconstructs frozen RF survival probabilities')
    require(list(sv['feature_names'])==feature_names(rf).tolist(),'SHAP feature order matches model input')
    for relative in ['figures/eda_histograms.png','figures/eda_boxplots.png','figures/eda_heatmap.png','figures/eda_counts.png','figures/eda_missing.png',
        'results/figures/shap_beeswarm.png','results/figures/shap_waterfall.png','results/figures/importance_comparison.png','results/figures/heldout_curves.png']:
        require((ROOT/relative).stat().st_size>1000,'Figure exists: '+relative)
    submission=load_submission(ROOT)
    pdf=report_path(ROOT,submission)
    reader=PdfReader(pdf);text='\n'.join(p.extract_text() or '' for p in reader.pages)
    require(len(reader.pages)>=18,'Final PDF includes complete report')
    for token in [submission['student_id'],submission['student_name'],'EDA','GridSearchCV','SHAP','AutoML','ROC-AUC','최종 결론']:
        require(token in text,'PDF text includes '+token)
    require('\ufffd' not in text and '\u25a0' not in text,'No replacement glyphs in extracted PDF text')
    if submission.get('github_url'):
        url=submission['github_url']
        require(url in text,'Published source URL appears in PDF text')
        urls=[]
        for page in reader.pages:
            for reference in page.get('/Annots',[]):
                annotation=reference.get_object()
                action=annotation.get('/A',{})
                if action.get('/URI'):urls.append(str(action['/URI']))
        require(urls.count(url)>=2,'First and last source links are clickable PDF annotations')
    result={'status':'passed','checks_passed':len(checks),'pdf_pages':len(reader.pages),'checks':checks}
    (ROOT/'results/validation_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"PASS: {len(checks)} checks; PDF {len(reader.pages)} pages")

if __name__=='__main__':main()
