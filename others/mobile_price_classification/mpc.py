import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import LogisticRegression
import seaborn as sns

data = pd.read_csv("train.csv")
X = data.iloc[:,0:20]  #independent columns
y = data.iloc[:,-1]    #target column i.e price range

#1
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k="all")
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
f1 = pd.concat([dfcolumns,dfscores],axis=1)
f1.columns = ['Specs','Score']  #naming the dataframe columns
f1.nlargest(10,'Score')  #print 10 best features

#2
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
f2 = feat_importances.nlargest(10)
f2.plot(kind='barh')
plt.show()

#3
#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")

#4
model = LogisticRegression()
# create the RFE model and select 3 attributes
rfe = RFE(model, 10)
rfe = rfe.fit(X, y)
# summarize the selection of the attributes
rfe.support_
rfe.ranking_
f3 = list(enumerate(X.columns))
#rank features
f3.sort(key = lambda idx_feature: rfe.ranking_[idx_feature[0]], reverse = True)
