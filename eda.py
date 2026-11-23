"""학습 자료의 결측치, 분포, 이상치, 상관관계를 살펴보고 그림으로 저장한다."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from src.시각화설정 import korean_font

ROOT = Path(__file__).resolve().parent
NAVY, TEAL, GOLD = '#17324D', '#008C95', '#E9A23B'
DEFINITIONS = [
 ('PassengerId', '승객 식별 번호', '식별자', '예측 입력 제외'),
 ('Survived', '생존 여부', '타겟', '0=사망, 1=생존'),
 ('Pclass', '객실 등급', '순서형 범주', '1=1등석, 2=2등석, 3=3등석'),
 ('Name', '승객 이름', '문자열', '호칭 Title 추출 후 원문 제외'),
 ('Sex', '성별', '범주형', 'female / male'),
 ('Age', '나이', '수치형', '년; 1세 미만은 소수'),
 ('SibSp', '동승 형제자매 및 배우자 수', '수치형', '명'),
 ('Parch', '동승 부모 및 자녀 수', '수치형', '명'),
 ('Ticket', '승선권 번호', '문자열', '고유값이 많아 예측 입력 제외'),
 ('Fare', '승선 요금', '수치형', '공유 승선권 등으로 해석에 주의'),
 ('Cabin', '객실 번호', '문자열', '기록 존재 여부 CabinKnown 생성'),
 ('Embarked', '승선 항구', '범주형', 'C=Cherbourg, Q=Queenstown, S=Southampton'),
]

def run():
    # 최종 평가 자료는 따로 두고, 변수 간 관계는 학습 자료에서 살펴본다.
    for folder in ['figures', 'results', 'docs']:
        (ROOT / folder).mkdir(exist_ok=True)
    raw = pd.read_csv(ROOT / 'data/train.csv')
    train, test = train_test_split(raw, test_size=.2, stratify=raw.Survived, random_state=42)
    assert raw.shape == (891, 12)
    assert set(train.PassengerId).isdisjoint(test.PassengerId)
    numeric = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    missing = lambda df: {c: {'count': int(df[c].isna().sum()), 'percent': float(df[c].isna().mean()*100)} for c in df}
    counts = lambda df: {str(k): {'count':int(v), 'percent':float(v/len(df)*100)} for k,v in df.Survived.value_counts().sort_index().items()}
    outliers = {}
    for col in ['Age','Fare','SibSp','Parch']:
        v = train[col].dropna(); q1,q3 = v.quantile([.25,.75]); iqr=q3-q1
        lo,hi=q1-1.5*iqr,q3+1.5*iqr
        outliers[col] = {'q1':q1,'q3':q3,'lower_bound':lo,'upper_bound':hi,
            'flagged_count':int(((v<lo)|(v>hi)).sum()),'non_missing_count':len(v), 'maximum':v.max()}
    corr=train[['Survived']+numeric].corr()
    rates={}
    for col in ['Sex','Pclass','Embarked']:
        work=train[[col,'Survived']].copy(); work[col]=work[col].astype('string').fillna('(Missing)')
        rates[col]=[{'category':str(k),'count':int(v['count']),'survival_rate':float(v['mean'])} for k,v in work.groupby(col).Survived.agg(['count','mean']).iterrows()]
    summary={'shape_full':list(raw.shape),'shape_train':list(train.shape),'shape_test':list(test.shape),
        'target_counts_full':counts(raw),'target_counts_train':counts(train),'target_counts_test':counts(test),
        'missing_full':missing(raw),'missing_train':missing(train),'outliers_iqr_train':outliers,
        'numeric_describe_train':train[numeric].describe().to_dict(),'correlation_train':corr.to_dict(),
        'survival_rates_train':rates,'duplicates_full':int(raw.duplicated().sum()),
        'duplicate_passenger_ids':int(raw.PassengerId.duplicated().sum()),
        'split':{'seed':42,'train_passenger_ids':train.PassengerId.tolist(),'test_passenger_ids':test.PassengerId.tolist()},
        'eda_scope':'전체 구조·결측·타겟 수를 제외한 탐색 분석은 학습 자료만 사용했다.'}
    (ROOT/'results/eda_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),encoding='utf-8')
    dictionary=pd.DataFrame(DEFINITIONS,columns=['column','description_ko','role','unit_or_values'])
    dictionary['dtype']=[str(raw[c].dtype) for c in dictionary.column]
    dictionary['missing_count_full']=[int(raw[c].isna().sum()) for c in dictionary.column]
    dictionary.to_csv(ROOT/'data_dictionary.csv',index=False,encoding='utf-8-sig')
    sns.set_theme(style='whitegrid',palette=[NAVY,TEAL],font=korean_font())
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
    def save(fig,name,title):
        fig.suptitle(title,fontsize=16,color=NAVY,fontweight='bold',y=1.025)
        fig.text(.01,-.018,'타이타닉 | 학습 자료 712개 | 층화 분할 80/20, seed=42',fontsize=8,color='#536575')
        fig.savefig(ROOT/f'figures/{name}.png',dpi=210,bbox_inches='tight',facecolor='white');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12.5,7.3),constrained_layout=True)
    for ax,col in zip(axes.flat,['Age','Fare','LogFare','SibSp','Parch','Pclass']):
        v=np.log1p(train.Fare) if col=='LogFare' else train[col]
        sns.histplot(v.dropna(),ax=ax,color=TEAL,**({'discrete':True} if col in ['SibSp','Parch','Pclass'] else {'bins':25}))
        ax.set(title=col,xlabel='log(1 + Fare)' if col=='LogFare' else col,ylabel='승객 수')
    save(fig,'eda_histograms','수치형 변수의 분포')
    train=train.copy();train['Outcome']=train.Survived.map({0:'사망',1:'생존'})
    fig,axes=plt.subplots(2,2,figsize=(10.5,7.3),constrained_layout=True)
    for ax,col in zip(axes.flat,['Age','Fare','SibSp','Parch']):
        sns.boxplot(data=train,x='Outcome',y=col,hue='Outcome',order=['사망','생존'],hue_order=['사망','생존'],palette=[NAVY,TEAL],legend=False,ax=ax,width=.55,fliersize=3)
        ax.set(title=col,xlabel='')
    save(fig,'eda_boxplots','생존 여부별 분포와 이상치 탐색')
    fig,ax=plt.subplots(figsize=(8,6.5),constrained_layout=True)
    sns.heatmap(corr,annot=True,fmt='.2f',cmap='RdBu_r',vmin=-1,vmax=1,square=True,linewidths=.5,ax=ax)
    save(fig,'eda_heatmap','수치형 변수의 피어슨 상관관계')
    fig,axes=plt.subplots(2,2,figsize=(11.5,7.5),constrained_layout=True)
    for ax,col in zip(axes.flat,['Outcome','Sex','Pclass','Embarked']):
        plot=train.copy();plot['Embarked']=plot.Embarked.fillna('(결측)')
        sns.countplot(data=plot,x=col,hue='Outcome',hue_order=['사망','생존'],order=['사망','생존'] if col=='Outcome' else None,palette=[NAVY,TEAL],ax=ax)
        label={'Outcome':'생존 여부','Sex':'성별','Pclass':'객실 등급','Embarked':'승선 항구'}[col]
        ax.set(title=label,xlabel=label,ylabel='승객 수')
        handles,labels=ax.get_legend_handles_labels()
        if handles:ax.legend(title='',fontsize=8)
    save(fig,'eda_counts','타겟과 범주형 변수의 분포')
    vals=train.drop(columns='Outcome').isna().mean().mul(100).sort_values()
    fig,ax=plt.subplots(figsize=(9.5,6.2),constrained_layout=True)
    ax.barh(vals.index,vals.values,color=[GOLD if x>50 else TEAL for x in vals.values])
    for i,(c,v) in enumerate(vals.items()):ax.text(v+1,i,f'{v:.1f}% ({train[c].isna().sum()}/712)',va='center',fontsize=9)
    ax.set(xlim=(0,105),xlabel='결측 비율 (%)');ax.grid(axis='y',visible=False)
    save(fig,'eda_missing','변수별 결측 비율')
    (ROOT/'docs/eda_notes.md').write_text('''# EDA 분석 범위와 주의점\n\n전체 891행의 구조, 결측률, 타겟 분포는 자료 소개용이다. 분포, 이상치, 변수-타겟 관계는 층화 분할한 학습 712행에서만 탐색하였다. 정확한 수치는 results/eda_summary.json에 저장한다.\n\nCabin 결측은 객실이 없었다는 뜻이 아니다. IQR 이상치는 자동 삭제하지 않는다. 요금의 긴 오른쪽 꼬리는 실제 고액 요금일 수 있다. SibSp/Parch는 0이 많아 IQR 기준에 주의한다. Pearson 상관관계는 인과관계가 아니며 Pclass는 순서형 코드의 기술적 요약이다.\n\n출처: https://www.kaggle.com/competitions/titanic/data\n''',encoding='utf-8')
    print(f'데이터 분석 완료: 전체={raw.shape}, 학습={summary["shape_train"]}, 평가={test.shape}',flush=True)

if __name__=='__main__':run()
