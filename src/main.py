from pathlib import Path
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,RocCurveDisplay
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
df=pd.read_csv(ROOT/'data/customers.csv'); X=df.drop(columns='Churn'); y=df.Churn; num=['TenureMonths','MonthlyCharges','SupportTickets','AutoPay']; cat=['Contract','InternetService']
pre=ColumnTransformer([('num',StandardScaler(),num),('cat',OneHotEncoder(handle_unknown='ignore'),cat)])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=42)
models={'LogisticRegression':LogisticRegression(max_iter=2000,class_weight='balanced',random_state=42),'RandomForest':RandomForestClassifier(n_estimators=300,max_depth=9,min_samples_leaf=4,class_weight='balanced',n_jobs=-1,random_state=42)}
rows=[]; pipes={}
for name,model in models.items():
    pipe=Pipeline([('pre',pre),('model',model)]); pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); prob=pipe.predict_proba(Xte)[:,1]; pipes[name]=pipe
    rows.append({'model':name,'accuracy':accuracy_score(yte,pred),'precision':precision_score(yte,pred),'recall':recall_score(yte,pred),'f1':f1_score(yte,pred),'roc_auc':roc_auc_score(yte,prob)})
pd.DataFrame(rows).to_csv(OUT/'model_metrics.csv',index=False)
rf=pipes['RandomForest']; names=rf.named_steps['pre'].get_feature_names_out(); imp=pd.Series(rf.named_steps['model'].feature_importances_,index=names).sort_values(ascending=False); imp.to_csv(OUT/'feature_importance.csv',header=['importance'])
plt.figure(figsize=(7,4)); imp.head(10).sort_values().plot(kind='barh'); plt.title('Top Churn Drivers'); plt.tight_layout(); plt.savefig(OUT/'feature_importance.png',dpi=160); plt.close()
best=max(rows,key=lambda r:r['roc_auc'])['model']; prob=pipes[best].predict_proba(Xte)[:,1]; preds=Xte.copy(); preds['actual_churn']=yte.values; preds['churn_probability']=prob; preds.sort_values('churn_probability',ascending=False).head(100).to_csv(OUT/'highest_risk_customers.csv',index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False)); print('Churn rate:',round(y.mean()*100,2),'%')
