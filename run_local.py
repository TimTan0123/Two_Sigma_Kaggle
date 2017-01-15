# 把kagglegym.py文件拷贝到运行的本地目录,该程序就可以在本地运行了
# 在用pandas读取.h5文件时,需要安装PyTables,安装指令sudo pip install tables
# kagglegym.py 在这个repository中了
import kagglegym
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression,Ridge
import xgboost as xgb

env = kagglegym.make()
o = env.reset()
print o.train.shape
#excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]
excl = ['id', 'sample', 'y', 'timestamp']
col = [c for c in o.train.columns if c not in excl]

train = pd.read_hdf('train.h5')
train = train[col]
d_mean= train.median(axis=0)

train = o.train[col]
n = train.isnull().sum(axis=1)
for c in train.columns:
    train[c + '_nan_'] = pd.isnull(train[c])
    d_mean[c + '_nan_'] = 0
train = train.fillna(d_mean)
train['znull'] = n
n = []
print "start train for Trees"
rfr = ExtraTreesRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=17, verbose=0)
model1 = rfr.fit(train, o.train['y'])
# clf_gbdt = GradientBoostingRegressor(n_estimators=15, learning_rate=0.5,max_depth=6, random_state=0)
# model1 = clf_gbdt.fit(train, o.train['y'])
#model1 = rfr.fit(train, o.train['y'])
#model_xgb = runXGB(train,o.train['y'])
print "finised train"

#https://www.kaggle.com/bguberfain/two-sigma-financial-modeling/univariate-model-with-clip/run/482189
low_y_cut = -0.06
high_y_cut = 0.06
y_is_above_cut = (o.train.y > high_y_cut)
y_is_below_cut = (o.train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)
model2 = LinearRegression()
X_T20 = np.array(o.train[col].fillna(d_mean).loc[y_is_within_cut, 'technical_20'].values).reshape(-1,1)
X_T30 = np.array(o.train[col].fillna(d_mean).loc[y_is_within_cut, 'technical_30'].values).reshape(-1,1)
X = np.concatenate((X_T20,X_T20**2,X_T30**2),axis = 1)
model2.fit(X, o.train.loc[y_is_within_cut, 'y'])
train = []

#https://www.kaggle.com/ymcdull/two-sigma-financial-modeling/ridge-lb-0-0100659
ymean_dict = dict(o.train.groupby(["id"])["y"].median())

while True:
    test = o.features[col]
    n = test.isnull().sum(axis=1)
    for c in test.columns:
        test[c + '_nan_'] = pd.isnull(test[c])
    test = test.fillna(d_mean)
    test['znull'] = n
    pred = o.target
    test_T20 = np.array(o.features[col].fillna(d_mean)['technical_20'].values).reshape(-1,1)
    test_T30 = np.array(o.features[col].fillna(d_mean)['technical_30'].values).reshape(-1,1)
    test2 = np.concatenate((test_T20,test_T20**2,test_T30**2),axis = 1)
    #pred['y'] = (model1.predict(test).clip(low_y_cut, high_y_cut) * 0.65) + (model2.predict(test2).clip(low_y_cut, high_y_cut) * 0.35)
    pred['y'] = (model1.predict(test).clip(low_y_cut, high_y_cut) * 0.65) + (model2.predict(test2).clip(low_y_cut, high_y_cut) * 0.35)
    pred['y'] = pred.apply(lambda r: 0.95 * r['y'] + 0.05 * ymean_dict[r['id']] if r['id'] in ymean_dict else r['y'], axis = 1)
    pred['y'] = [float(format(x, '.6f')) for x in pred['y']]
    o, reward, done, info = env.step(pred)
    if done:
        print("el fin ...", info["public_score"])
        break
    if o.features.timestamp[0] % 100 == 0:
        print(reward)
