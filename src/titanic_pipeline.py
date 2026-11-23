"""타이타닉 데이터의 전처리 방법을 비교하고 생존 여부를 예측한다.

전처리와 모델은 학습 자료의 교차검증으로 고른다. 평가 자료의 정보가
미리 들어가지 않도록 결측 대체, 인코딩, 스케일링, 변수 선택도 각 학습
구간에서 수행한다. AutoML은 학습 자료 안에 별도의 검증 자료를 둔다.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, make_scorer, ConfusionMatrixDisplay, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from threadpoolctl import threadpool_limits
from src.시각화설정 import korean_font

METRICS = ['accuracy','precision','recall','f1','roc_auc']
SCORING = {m:m for m in ['accuracy','roc_auc']}
SCORING.update({m:make_scorer(f,zero_division=0) for m,f in [('precision',precision_score),('recall',recall_score),('f1',f1_score)]})
MODELS = ['LogisticRegression','RandomForest']
MODEL_LABELS = {'LogisticRegression':'로지스틱 회귀','RandomForest':'랜덤 포레스트'}
BASE_COLUMNS = ['Pclass','SibSp','Parch','Fare']

def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [clean(v) for v in value]
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,(np.floating,float)):return float(value) if np.isfinite(value) else None
    if isinstance(value,np.bool_):return bool(value)
    if isinstance(value,Path):return str(value)
    return value

def dump(path,value):
    Path(path).write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

class BaselineFeatures(BaseEstimator,TransformerMixin):
    """별도 전처리 없이 사용할 수 있는 원본 수치형 변수 4개를 가져온다."""
    def fit(self,X,y=None):
        if X[BASE_COLUMNS].isna().any().any():raise ValueError('Base 입력 변수에 결측값이 있습니다.')
        self.n_features_in_=X.shape[1]
        return self
    def transform(self,X):return X[BASE_COLUMNS].to_numpy(dtype=float)
    def get_feature_names_out(self,input_features=None):return np.array(BASE_COLUMNS,dtype=object)

class TitanicFeatures(BaseEstimator,TransformerMixin):
    """승객 한 명의 정보만으로 파생 변수를 만든다. 타겟 통계는 사용하지 않는다."""
    def __init__(self,enabled=True):self.enabled=enabled
    def fit(self,X,y=None):
        self.n_features_in_=X.shape[1]
        return self
    def transform(self,X):
        out=X[['Age','SibSp','Parch','Fare','Sex','Embarked']].copy()
        out['Pclass']=X.Pclass.map(lambda v:np.nan if pd.isna(v) else str(int(v)))
        if self.enabled:
            # 본인을 포함한 가족 수와 혼자 탑승했는지를 구한다.
            family=X.SibSp+X.Parch+1
            out['FamilySize']=family
            out['FarePerPerson']=X.Fare/family
            out['IsAlone']=(family==1).astype(float)
            out['CabinKnown']=X.Cabin.notna().astype(float)
            # 이름 전체 대신 호칭만 추출하고, 자주 쓰는 호칭 외에는 Other로 묶는다.
            title=X.Name.str.extract(r',\s*([^.]*)\.',expand=False).str.strip()
            title=title.replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
            out['Title']=title.where(title.isin(['Mr','Mrs','Miss','Master']),'Other')
            out.loc[title.isna(),'Title']=np.nan
            out['AgeGroup']=pd.cut(X.Age,[-np.inf,12,18,35,60,np.inf],labels=['Child','Teen','YoungAdult','Adult','Senior']).astype(object)
        return out

class ForestTopK(BaseEstimator,TransformerMixin):
    """현재 학습 구간의 RF 중요도로 상위 k개 변수를 고른다."""
    def __init__(self,k=12,random_state=42):self.k=k;self.random_state=random_state
    def fit(self,X,y):
        self.n_features_in_=X.shape[1]
        self.estimator_=RandomForestClassifier(n_estimators=100,max_depth=6,min_samples_leaf=2,random_state=self.random_state,n_jobs=1).fit(X,y)
        self.feature_importances_=self.estimator_.feature_importances_
        self.selected_indices_=np.sort(np.argsort(-self.feature_importances_,kind='stable')[:min(self.k,X.shape[1])])
        return self
    def transform(self,X):return np.asarray(X)[:,self.selected_indices_]
    def get_feature_names_out(self,input_features=None):
        if input_features is None:input_features=[f'x{i}' for i in range(self.n_features_in_)]
        return np.asarray(input_features,dtype=object)[self.selected_indices_]

@dataclass(frozen=True)
class Experiment:
    name:str
    imputation:str='median'
    encoding:str='onehot'
    scaling:str='standard'
    engineered:bool=True
    selection:bool=False
    baseline:bool=False

# 과제의 필수 조합 4개와 한 조건씩 바꾸는 비교 실험이다.
EXPERIMENTS=[Experiment('Base',baseline=True,engineered=False),Experiment('Exp-1',imputation='mean'),
    Experiment('Exp-2',encoding='ordinal',scaling='minmax',selection=True),
    Experiment('Exp-3',imputation='most_frequent',scaling='robust',selection=True),
    Experiment('Reference'),Experiment('Impute-MostFrequent',imputation='most_frequent'),
    Experiment('Encode-Ordinal',encoding='ordinal'),Experiment('Scale-None',scaling='none'),
    Experiment('Scale-MinMax',scaling='minmax'),Experiment('Scale-Robust',scaling='robust'),
    Experiment('Without-Derived',engineered=False),Experiment('Select-Top12',selection=True)]
BY_NAME={s.name:s for s in EXPERIMENTS}
MAIN=['Base','Exp-1','Exp-2','Exp-3']
ABLATIONS=[('Imputation','Exp-1'),('Imputation','Impute-MostFrequent'),('Encoding','Encode-Ordinal'),
    ('Scaling','Scale-None'),('Scaling','Scale-MinMax'),('Scaling','Scale-Robust'),
    ('Derived features','Without-Derived'),('Feature selection','Select-Top12')]

def build_pipeline(spec,model_name,seed=42):
    """실험 설정에 맞춰 파생 변수 → 전처리 → 변수 선택 → 모델을 연결한다."""
    model=(LogisticRegression(C=1,solver='lbfgs',max_iter=3000,random_state=seed) if model_name=='LogisticRegression'
        else RandomForestClassifier(n_estimators=160,max_depth=6,min_samples_leaf=2,max_features='sqrt',random_state=seed,n_jobs=1))
    if spec.baseline:return Pipeline([('baseline',BaselineFeatures()),('model',model)])
    num=['Age','SibSp','Parch','Fare'];cat=['Pclass','Sex','Embarked']
    if spec.engineered:num+=['FamilySize','FarePerPerson','IsAlone','CabinKnown'];cat+=['Title','AgeGroup']
    scale={'none':'passthrough','standard':StandardScaler(),'minmax':MinMaxScaler(),'robust':RobustScaler()}[spec.scaling]
    encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False) if spec.encoding=='onehot' else OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
    pre=ColumnTransformer([
        ('numeric',Pipeline([('imputer',SimpleImputer(strategy=spec.imputation,keep_empty_features=True)),('scaler',scale)]),num),
        ('categorical',Pipeline([('imputer',SimpleImputer(strategy='most_frequent',keep_empty_features=True)),('encoder',encoder)]),cat)])
    return Pipeline([('features',TitanicFeatures(spec.engineered)),('preprocess',pre),
        ('selection',ForestTopK(12,seed) if spec.selection else 'passthrough'),('model',model)])

def feature_names(pipeline):
    if 'baseline' in pipeline.named_steps:return pipeline.named_steps['baseline'].get_feature_names_out()
    names=pipeline.named_steps['preprocess'].get_feature_names_out()
    if pipeline.named_steps['selection']!='passthrough':names=pipeline.named_steps['selection'].get_feature_names_out(names)
    return np.asarray(names,dtype=object)

def metrics(y,p):
    """생존 확률로 AUC를 구하고, 0.5 기준으로 나머지 분류 지표를 구한다."""
    pred=(np.asarray(p)>=.5).astype(int)
    return dict(accuracy=accuracy_score(y,pred),precision=precision_score(y,pred,zero_division=0),
        recall=recall_score(y,pred,zero_division=0),f1=f1_score(y,pred,zero_division=0),roc_auc=roc_auc_score(y,p))

def run_cv(pipe,X,y,folds,jobs):
    """같은 분할을 사용해 전처리와 모델 전체를 교차검증한다."""
    return cross_validate(pipe,X,y,cv=folds,scoring=SCORING,return_train_score=True,return_estimator=True,n_jobs=jobs,error_score='raise')

def cv_rows(result,experiment,model):
    row={'experiment':experiment,'model':model};fold_rows=[]
    for part,prefix in [('train','train'),('test','cv')]:
        for metric in METRICS:
            v=result[f'{part}_{metric}'];row[f'{prefix}_{metric}_mean']=v.mean();row[f'{prefix}_{metric}_std']=v.std(ddof=1)
    counts=[len(feature_names(est)) for est in result['estimator']]
    row.update(n_features_mean=np.mean(counts),n_features_min=min(counts),n_features_max=max(counts),
        auc_generalization_gap=row['train_roc_auc_mean']-row['cv_roc_auc_mean'])
    for fold,est in enumerate(result['estimator']):
        fr={'experiment':experiment,'model':model,'fold':fold+1,'n_features':counts[fold],
            'selected_features':json.dumps(feature_names(est).tolist())}
        for part,prefix in [('train','train'),('test','validation')]:
            for metric in METRICS:fr[f'{prefix}_{metric}']=result[f'{part}_{metric}'][fold]
        fold_rows.append(fr)
    return row,fold_rows

def fit_automl(X,y,seed,iterations,output):
    """학습 자료를 다시 나눠 FLAML로 모델 종류와 파라미터를 탐색한다."""
    from flaml import AutoML
    it,iv=train_test_split(np.arange(len(X)),test_size=.2,stratify=y,random_state=seed+1)
    reference=build_pipeline(BY_NAME['Reference'],'LogisticRegression',seed)
    # 내부 검증 자료에는 fit하지 않고 transform만 적용한다.
    prep=clone(reference[:-1]);xt=prep.fit_transform(X.iloc[it],y.iloc[it]);xv=prep.transform(X.iloc[iv])
    log=output/'automl_trials.jsonl'
    if log.exists():log.unlink()
    automl=AutoML();start=time.perf_counter()
    automl.fit(X_train=xt,y_train=y.iloc[it].to_numpy(),X_val=xv,y_val=y.iloc[iv].to_numpy(),
        task='classification',metric='roc_auc',eval_method='holdout',estimator_list=['rf','extra_tree','lrl2'],
        max_iter=iterations,time_budget=-1,sample=False,retrain_full=False,seed=seed,n_jobs=1,verbose=0,
        log_file_name=str(log),log_type='all')
    valp=automl.predict_proba(xv)[:,1]
    pipeline=Pipeline([('features',clone(reference.named_steps['features'])),('preprocess',clone(reference.named_steps['preprocess'])),
        ('selection','passthrough'),('model',clone(automl.model.estimator))]).fit(X,y)
    trials=[json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    meta={'library':'FLAML','estimators':['rf','extra_tree','lrl2'],'best_estimator':automl.best_estimator,
        'best_config':automl.best_config,'requested_iterations':iterations,'logged_trials':sum('record_id' in v for v in trials),
        'elapsed_seconds':time.perf_counter()-start,'inner_train_n':len(it),'inner_validation_n':len(iv),
        'inner_validation_metrics':metrics(y.iloc[iv],valp),'inner_train_passenger_ids':X.iloc[it].PassengerId.tolist(),
        'inner_validation_passenger_ids':X.iloc[iv].PassengerId.tolist(),
        'preprocessing':'Reference 전처리를 내부 학습 자료에만 fit한 뒤, 선택된 모델을 전체 학습 자료로 재학습',
        'note':'탐색 횟수는 고정했다. 환경과 버전에 따라 결과가 달라질 수 있으며, 선택 후 CV는 중첩 CV가 아니다.'}
    return pipeline,meta

def explain(pipeline,Xtrain,Xtest,ytest,seed,output,figures):
    """선택이 끝난 RF의 중요도와 SHAP 값을 확인한다. 이 결과로 재선정하지 않는다."""
    import shap
    names=feature_names(pipeline);train=pipeline[:-1].transform(Xtrain);test=pipeline[:-1].transform(Xtest);model=pipeline.named_steps['model']
    perm=permutation_importance(model,test,ytest,scoring='roc_auc',n_repeats=20,random_state=seed,n_jobs=1)
    importance=pd.DataFrame({'feature':names,'impurity_importance':model.feature_importances_,
        'test_permutation_auc_decrease':perm.importances_mean,'test_permutation_std':perm.importances_std})
    importance.to_csv(output/'feature_importance.csv',index=False)
    rawperm=permutation_importance(pipeline,Xtest,ytest,scoring='roc_auc',n_repeats=20,random_state=seed,n_jobs=1)
    pd.DataFrame({'feature':Xtest.columns,'test_permutation_auc_decrease':rawperm.importances_mean,
        'test_permutation_std':rawperm.importances_std}).sort_values('test_permutation_auc_decrease',ascending=False).to_csv(output/'raw_feature_permutation.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(14,7))
    for ax,col,title in [(axes[0],'impurity_importance','학습 자료의 불순도 중요도'),(axes[1],'test_permutation_auc_decrease','평가 자료의 순열 중요도')]:
        d=importance.nlargest(12,col).sort_values(col)
        ax.barh(d.feature,d[col],xerr=d.test_permutation_std if 'permutation' in col else None,capsize=2,color='#008C95')
        ax.set(title=title,xlabel='중요도' if 'impurity' in col else '변수를 섞었을 때 AUC 감소량 (반복 표준편차)')
    fig.tight_layout();fig.savefig(figures/'importance_comparison.png',dpi=200);plt.close(fig)
    # RF의 raw 출력은 클래스 확률이다. 학습된 트리의 경로별 표본 수를
    # 사용하는 방식으로 SHAP을 구하고, SHAP 합이 예측 확률과 맞는지 확인한다.
    exp=shap.TreeExplainer(model,feature_perturbation='tree_path_dependent',model_output='raw')(test,check_additivity=True)
    if exp.values.ndim!=3:raise ValueError(f'RF SHAP 값의 차원이 예상과 다릅니다: {exp.values.shape}')
    vals=exp.values[:,:,1];base=exp.base_values[:,1];prob=model.predict_proba(test)[:,1]
    error=float(np.max(np.abs(base+vals.sum(axis=1)-prob)))
    if error>1e-5:raise ValueError(f'SHAP 합산 검사를 통과하지 못했습니다: {error}')
    si=pd.DataFrame({'feature':names,'mean_absolute_shap':np.abs(vals).mean(axis=0),'mean_signed_shap':vals.mean(axis=0)}).sort_values('mean_absolute_shap',ascending=False)
    si.to_csv(output/'shap_importance.csv',index=False)
    np.savez_compressed(output/'shap_values.npz',values=vals,base_values=base,features=test,feature_names=names.astype(str))
    shap.summary_plot(vals,test,feature_names=names.tolist(),max_display=15,show=False)
    plt.gcf().set_size_inches(10,7);plt.title('튜닝한 랜덤 포레스트의 생존 확률 SHAP');plt.xlabel('SHAP 값 (생존 확률에 대한 기여)');plt.tight_layout()
    plt.savefig(figures/'shap_beeswarm.png',dpi=200,bbox_inches='tight');plt.close('all')
    shap.plots.waterfall(shap.Explanation(values=vals[0],base_values=float(base[0]),data=test[0],feature_names=names.tolist()),max_display=12,show=False)
    plt.gcf().set_size_inches(10,7);plt.tight_layout();plt.savefig(figures/'shap_waterfall.png',dpi=200,bbox_inches='tight');plt.close('all')
    meta={'model':'Tuned / RandomForest','output':'생존 확률 (랜덤 포레스트의 raw 클래스 확률)',
        'feature_perturbation':'tree_path_dependent','background_n':None,'background_partition':'학습된 트리에 저장된 경로별 학습 표본 수',
        'n_cases':len(test),'max_additivity_error':error,'waterfall_passenger_id':int(Xtest.iloc[0].PassengerId),
        'waterfall_probability':float(prob[0]),'waterfall_base_probability':float(base[0]),'top_shap_features':si.head(15).to_dict('records'),
        'limits':['예측 설명은 인과관계가 아니다.','상관된 변수와 파생 변수는 정보를 공유한다.',
            '원-핫 열을 개별적으로 섞으면 비현실적인 조합이 생길 수 있어 원본 열 순열 결과도 제공한다.',
            '평가 자료의 설명 결과로 모델이나 변수를 다시 고르지 않았다.']}
    dump(output/'explainability_summary.json',meta)
    return meta

def plot_evaluation(cv,test,abl,pred,y,primary,figures):
    sns.set_theme(style='whitegrid',palette=['#17324D','#008C95'],font=korean_font())
    fig,ax=plt.subplots(figsize=(10,5))
    for i,model in enumerate(MODELS):
        d=cv[(cv.model==model)&cv.experiment.isin(MAIN)].set_index('experiment').loc[MAIN]
        ax.bar(np.arange(4)+(i-.5)*.34,d.cv_roc_auc_mean,width=.34,yerr=d.cv_roc_auc_std,capsize=3,label=MODEL_LABELS[model])
    ax.set(xticks=np.arange(4),xticklabels=MAIN,ylim=(.5,1),ylabel='5겹 교차검증 AUC (평균과 표준편차)',title='필수 실험 조합 비교');ax.legend()
    fig.tight_layout();fig.savefig(figures/'cv_main_comparison.png',dpi=200);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,6),sharey=True)
    for ax,model in zip(axes,MODELS):
        d=abl[abl.model==model]
        ax.errorbar(d.delta_cv_auc,np.arange(len(d)),xerr=d.paired_fold_delta_std,fmt='o',capsize=3)
        ax.axvline(0,c='grey',ls='--');ax.set(yticks=np.arange(len(d)),yticklabels=d.alternative,title=MODEL_LABELS[model],xlabel='Reference 대비 AUC 차이 (동일 검증 구간 차이의 표준편차)')
    axes[0].invert_yaxis();fig.tight_layout();fig.savefig(figures/'ablation_auc.png',dpi=200);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for key in ['Base / LogisticRegression','Base / RandomForest','Tuned / LogisticRegression','Tuned / RandomForest','AutoML / FLAML']:
        p=pred[key];fpr,tpr,_=roc_curve(y,p);pr,rc,_=precision_recall_curve(y,p)
        axes[0].plot(fpr,tpr,label=f'{key} ({roc_auc_score(y,p):.3f})');axes[1].plot(rc,pr,label=f'{key} ({average_precision_score(y,p):.3f})')
    axes[0].plot([0,1],[0,1],c='grey',ls='--');axes[1].axhline(y.mean(),c='grey',ls='--')
    axes[0].set(xlabel='거짓 양성 비율',ylabel='참 양성 비율',title='최종 평가 ROC 곡선 (AUC)')
    axes[1].set(xlabel='재현율',ylabel='정밀도',title='최종 평가 정밀도-재현율 곡선 (AP)')
    for ax in axes:ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(figures/'heldout_curves.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5));ConfusionMatrixDisplay.from_predictions(y,(pred[primary]>=.5).astype(int),display_labels=['사망','생존'],ax=ax,colorbar=False,cmap='Blues')
    ax.set(xlabel='예측값',ylabel='실제값',title=f'{primary}\n분류 임계값 = 0.5');fig.tight_layout();fig.savefig(figures/'primary_confusion_matrix.png',dpi=200);plt.close(fig)

def execute(data_path,output,seed=42,jobs=2,automl_iterations=30):
    """데이터 분할부터 최종 평가와 설명 그림 저장까지 순서대로 실행한다."""
    start=time.perf_counter();output=Path(output);figures=output/'figures';models=output/'models'
    for d in [output,figures,models]:d.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(data_path)
    assert raw.shape==(891,12) and raw.PassengerId.is_unique and set(raw.Survived)=={0,1}
    X=raw.drop(columns='Survived');y=raw.Survived.astype(int)
    ti,vi=train_test_split(np.arange(len(raw)),test_size=.2,stratify=y,random_state=seed)
    Xt,Xv,yt,yv=X.iloc[ti],X.iloc[vi],y.iloc[ti],y.iloc[vi]
    folds=list(StratifiedKFold(5,shuffle=True,random_state=seed).split(Xt,yt))
    assignment=pd.DataFrame({'PassengerId':raw.PassengerId,'partition':'test','validation_fold':pd.Series([pd.NA]*len(raw),dtype='Int64')})
    assignment.loc[ti,'partition']='train'
    for k,(_,va) in enumerate(folds,1):assignment.loc[ti[va],'validation_fold']=k
    assignment.to_csv(output/'split_assignment.csv',index=False)
    print(f'데이터 크기={raw.shape}; 학습={len(Xt)}개; 최종 평가={len(Xv)}개',flush=True)
    cvs=[];frs=[]
    for spec in EXPERIMENTS:
        for model in MODELS:
            print(f'교차검증: {spec.name} / {model}',flush=True)
            result=run_cv(build_pipeline(spec,model,seed),Xt,yt,folds,jobs)
            row,fr=cv_rows(result,spec.name,model);cvs.append(row);frs+=fr
    cv=pd.DataFrame(cvs);fold_table=pd.DataFrame(frs)
    cv.to_csv(output/'cv_results.csv',index=False);fold_table.to_csv(output/'cv_fold_scores.csv',index=False)
    ab=[]
    for model in MODELS:
        reference=fold_table[(fold_table.model==model)&(fold_table.experiment=='Reference')].set_index('fold')
        for factor,alt in ABLATIONS:
            alternative=fold_table[(fold_table.model==model)&(fold_table.experiment==alt)].set_index('fold')
            delta=alternative.validation_roc_auc-reference.validation_roc_auc
            ab.append({'model':model,'factor':factor,'reference':'Reference','alternative':alt,'delta_cv_auc':delta.mean(),
                'paired_fold_delta_std':delta.std(ddof=1),'delta_cv_f1':(alternative.validation_f1-reference.validation_f1).mean(),
                'delta_train_auc':(alternative.train_roc_auc-reference.train_roc_auc).mean()})
    abl=pd.DataFrame(ab);abl.to_csv(output/'ablation_deltas.csv',index=False)
    tuned={};tuning={};grids=[]
    for model in MODELS:
        exp=cv[(cv.model==model)&(cv.experiment!='Base')].sort_values(['cv_roc_auc_mean','experiment'],ascending=[False,True]).iloc[0].experiment
        grid=({'model__C':[.1,1,10],'model__class_weight':[None,'balanced']} if model=='LogisticRegression' else
            {'model__max_depth':[3,6,None],'model__min_samples_leaf':[1,3],'model__max_features':['sqrt',.8]})
        print(f'파라미터 탐색: {model}, 전처리={exp}',flush=True)
        search=GridSearchCV(build_pipeline(BY_NAME[exp],model,seed),grid,scoring=SCORING,refit='roc_auc',cv=folds,n_jobs=jobs,return_train_score=True,error_score='raise').fit(Xt,yt)
        tuned[model]=search.best_estimator_;b=search.best_index_
        tuning[model]={'preprocessing_experiment':exp,'best_params':search.best_params_,'best_cv_roc_auc':search.best_score_,
            'n_candidates':len(search.cv_results_['params']),'train_roc_auc':search.cv_results_['mean_train_roc_auc'][b],
            'cv_metrics':{m:{'mean':search.cv_results_[f'mean_test_{m}'][b],'std':search.cv_results_[f'std_test_{m}'][b]} for m in METRICS}}
        gf=pd.DataFrame(search.cv_results_);gf['model']=model;gf['preprocessing_experiment']=exp;gf['params']=gf.params.map(lambda p:json.dumps(p));grids.append(gf)
    pd.concat(grids,ignore_index=True).to_csv(output/'grid_search_results.csv',index=False);dump(output/'tuning_summary.json',tuning)
    print('FLAML: 학습 자료 내부의 별도 검증 자료로 탐색합니다.',flush=True)
    automl,am=fit_automl(Xt,yt,seed,automl_iterations,output)
    ar,afr=cv_rows(run_cv(clone(automl),Xt,yt,folds,jobs),'AutoML','FLAML')
    am['postselection_cv']=ar;dump(output/'automl_summary.json',am)
    pd.DataFrame([ar]).to_csv(output/'automl_cv.csv',index=False);pd.DataFrame(afr).to_csv(output/'automl_cv_fold_scores.csv',index=False)
    primary='Tuned / '+max(MODELS,key=lambda m:tuning[m]['best_cv_roc_auc'])
    frozen={f'{s} / {m}':build_pipeline(BY_NAME[s],m,seed).fit(Xt,yt) for s in MAIN for m in MODELS}
    frozen.update({f'Tuned / {m}':pipe for m,pipe in tuned.items()});frozen['AutoML / FLAML']=automl
    dump(output/'frozen_model_manifest.json',{'primary_model':primary,'selection_basis':'두 튜닝 모델 중 학습 CV ROC-AUC가 높은 모델',
        'test_not_used_for_selection':True,'classification_threshold':.5,'models':list(frozen),'tuning':tuning,'automl':am})
    print('모델 선택을 마쳤습니다. 별도로 보관한 평가 자료의 점수를 계산합니다.',flush=True)
    rows=[];preds={};pt=pd.DataFrame({'PassengerId':Xv.PassengerId.to_numpy(),'Survived':yv.to_numpy()})
    for key,pipe in frozen.items():
        p=pipe.predict_proba(Xv)[:,1];preds[key]=p
        row={'key':key,'experiment':key.split(' / ')[0],'model':key.split(' / ')[1],'n_features':len(feature_names(pipe)),**metrics(yv,p),
            **{'train_'+m:v for m,v in metrics(yt,pipe.predict_proba(Xt)[:,1]).items()}}
        row['train_test_auc_gap']=row['train_roc_auc']-row['roc_auc'];rows.append(row)
        pt[key+' probability']=p;pt[key+' prediction']=(p>=.5).astype(int)
        joblib.dump(pipe,models/(key.replace(' / ','__')+'.joblib'))
    test=pd.DataFrame(rows);test.to_csv(output/'test_results.csv',index=False);pt.to_csv(output/'test_predictions.csv',index=False)
    # 모델은 고정하고 평가 예측만 재표집한다. 학습·튜닝의 변동까지 나타내는 구간은 아니다.
    rng=np.random.default_rng(seed+2);yv_np=yv.to_numpy();groups=[np.flatnonzero(yv_np==k) for k in [0,1]];bs=[]
    for _ in range(1000):
        ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]);bs.append(metrics(yv_np[ix],preds[primary][ix]))
    intervals={m:{'low':np.quantile([b[m] for b in bs],.025),'high':np.quantile([b[m] for b in bs],.975)} for m in METRICS}
    plot_evaluation(cv,test,abl,preds,yv,primary,figures)
    print('불순도·순열 중요도와 SHAP을 계산합니다.',flush=True)
    explanations=explain(tuned['RandomForest'],Xt,Xv,yv,seed,output,figures)
    versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','matplotlib','seaborn','shap','flaml','joblib']}
    summary={'status':'complete','dataset':{'shape':list(raw.shape),'sha256':hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        'target':'Survived','class_counts':y.value_counts().to_dict(),'missing_counts':raw.isna().sum().to_dict()},
        'protocol':{'seed':seed,'train_n':len(Xt),'test_n':len(Xv),'cv':'섞은 층화 5겹 교차검증; 모든 실험에 같은 분할 적용',
            'primary_metric':'roc_auc','threshold':.5,'baseline_columns':BASE_COLUMNS,'selection':'변환된 열의 RF 중요도 상위 12개',
            'limitations':['같은 학습 CV에서 가장 높은 점수를 골랐으므로 선택 편향이 있다. 중첩 CV는 수행하지 않았다.',
                '연관된 승객이 양쪽 자료에 나뉠 수 있다. 외부 자료나 가족 단위 일반화는 검증하지 않았다.',
                'AutoML은 내부 검증 자료로 탐색했다. 선택 후 CV는 독립적인 추정값으로 보기 어렵다.']},
        'experiments':[asdict(s) for s in EXPERIMENTS],'cv_results':cv.to_dict('records'),'ablation_results':abl.to_dict('records'),
        'tuning':tuning,'automl':am,'test_results':test.to_dict('records'),'selected_primary_model':primary,
        'primary_confusion_matrix':confusion_matrix(yv,(preds[primary]>=.5).astype(int)).tolist(),
        'primary_test_bootstrap_95_intervals':intervals,'selected_features':{key:feature_names(pipe).tolist() for key,pipe in frozen.items()},
        'explainability':explanations,'versions':versions,'python_version':platform.python_version(),'platform':platform.platform(),
        'elapsed_seconds':time.perf_counter()-start}
    dump(output/'summary.json',summary)
    print('실험 완료: '+str(output/'summary.json'),flush=True)

def main():
    parser=argparse.ArgumentParser(description='타이타닉 전처리 비교와 모델 학습·평가')
    parser.add_argument('--data',type=Path,default=Path('data/train.csv'),help='입력 CSV 경로')
    parser.add_argument('--output',type=Path,default=Path('results'),help='실험 결과를 저장할 폴더')
    parser.add_argument('--seed',type=int,default=42,help='난수 시드 (기본값: 42)')
    parser.add_argument('--jobs',type=int,default=2,help='교차검증 병렬 작업 수 (기본값: 2)')
    parser.add_argument('--automl-iterations',type=int,default=30,help='AutoML 탐색 횟수 (기본값: 30)')
    a=parser.parse_args()
    with threadpool_limits(limits=1):execute(a.data,a.output,a.seed,a.jobs,a.automl_iterations)

if __name__=='__main__':main()
