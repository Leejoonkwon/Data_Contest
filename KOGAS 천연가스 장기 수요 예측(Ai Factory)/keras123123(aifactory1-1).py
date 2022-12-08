import keras_tuner as kt
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
# from keras.optimizer_v2 import adam # tf 2.7 
from keras.optimizers import Adam     # tf 2.8 이상
from sklearn.metrics import accuracy_score
path = 'D:\\ai_data/'
data_a = pd.read_csv(path+'월별공급량및비중.csv')
data_b = pd.read_csv(path+'상업용 상대가격(기준=2015).csv')
data_c = pd.read_csv(path+'제조업 부가가치(분기별).csv')
data_d = pd.read_csv(path+'추가데이터(1).csv',thousands=',')

data_c = data_c.rename(columns={'제조업부가가치액':'MFG'})
# print(data_c.columns)

data_c['MFG'] = np.round(data_c[['MFG']].transform(lambda x : x /3),0)
data_c = pd.concat([data_c,data_c,data_c], ignore_index=True)
data_c = data_c.sort_values(by=['YEAR','QUARTER'],axis=0,ascending=True)
idx = pd.date_range('01-01-1996','12-31-2020', freq='M')

data_c['date']= pd.to_datetime(idx)
data_c['MONTH']=data_c['date'].dt.month
# data_c = data_c.drop(['date'],axis=1)
data_c = data_c.reset_index(drop=True)
data_c = data_c[['YEAR','MONTH','date','MFG','QUARTER']]

# 데이터 병합
data_all = pd.merge(data_a,data_b)
data_all = data_all.merge(data_c)
data_all = data_all.merge(data_d)

#분류형 피처 OneHot 
data_all = pd.get_dummies(data_all,columns=['QUARTER'])
# Index(['YEAR', 'MONTH', '도시가스(톤)_민수용', '도시가스(톤)_산업용', '도시가스(톤)_총합(민수용+산업용)',
#        '민수용비중', '산업용비중', 'RP(상대가격)', 'GAS_PRICE(산업용도시가스)',
#        'OIL_PRICE(원유정제처리제품)', 'MFG', '평균기온', '난방도일', '냉방도일', '최저기온', '최고기온',
#        '천연가스생산량(백만 m₂)', '산업소비량(백만 m₂)', '가정소비량(백만 m₂)', 'QUARTER_Q1',
#        'QUARTER_Q2', 'QUARTER_Q3', 'QUARTER_Q4'],
data_all = data_all.rename(
    columns={'도시가스(톤)_민수용':'CIVIL','RP(상대가격)':'RP',
             '도시가스(톤)_산업용':'IND','산업용비중':'INGper',
             '민수용비중':'CIVILper','GAS_PRICE(산업용도시가스)':'GAS_PRICE',
             'OIL_PRICE(원유정제처리제품)':'OIL_PRICE','평균기온':'Meantemp',
             '난방도일':'nanbangdoil','냉방도일':'nanenbangdoil',
             '최저기온':'lowtemp','최고기온':'hightemp',
             '천연가스생산량(백만 m₂)':'amount_of_gas','산업소비량(백만 m₂)':'INDcon',
             '가정소비량(백만 m₂)':'CIVILcon','도시가스(톤)_총합(민수용+산업용)':'Total'
             })
# print(data_all['YEAR'].head())
data_all['MOM_CIVIL'] = data_all['CIVIL'].pct_change()
data_all['MOM_IND'] = data_all['IND'].pct_change()
# print(data_all.columns)
# Index(['YEAR', 'MONTH', 'CIVIL', 'IND', 'Total', 'CIVILper', 'INGper', 'RP',
#        'GAS_PRICE', 'OIL_PRICE', 'MFG', 'Meantemp', 'nanbangdoil',
#        'nanenbangdoil', 'lowtemp', 'hightemp', 'amount_of_gas', 'INDcon',
#        'CIVILcon', 'QUARTER_Q1', 'QUARTER_Q2', 'QUARTER_Q3', 'QUARTER_Q4']


month_data = data_all['MONTH'].unique()
# data_all.plot()
# plt.title("시간에 따른 추이")
# plt.xlabel("시간")
# plt.ylabel("data")
# plt.show()
# data_year1996 = data_all[['date','CIVIL']][(data_all['YEAR'] == 1996)].reset_index(drop=True)
CIVIL_MEAN = []
IND_MEAN = []
for j in month_data:
    civil_month = data_all[['CIVIL']][(data_all['MONTH']==j)].reset_index(drop=True)
    ind_month = data_all[['IND']][(data_all['MONTH']==j)].reset_index(drop=True)
    CIVIL = []
    IND = []
    for i in range(len(civil_month)):
        if i+1 == len(civil_month):
            break
        else :
            civil2 = civil_month.loc[i+1] - civil_month.loc[i]
            ind2 = ind_month.loc[i+1] - ind_month.loc[i]
            CIVIL.append(civil2)
            IND.append(ind2)
    CIVIL = np.round(np.mean(CIVIL),0)
    CIVIL_MEAN.append(CIVIL)
    IND = np.round(np.mean(IND),0)
    IND_MEAN.append(IND)
CIVIL_MEAN = (np.array(CIVIL_MEAN)).reshape(-1,1)
IND_MEAN = (np.array(IND_MEAN)).reshape(-1,1)
total = np.concatenate((CIVIL_MEAN,IND_MEAN),1)
np.save('D:\\ai_data\mean_data.npy',arr = total)
data_all.to_csv('D:\\ai_data\data_all.csv',index=True)
# data_year1996 = data_all[['CIVIL']][(data_all['YEAR'] == 1996)].reset_index(drop=True)
# data_year1997 = data_all[['CIVIL']][(data_all['YEAR'] == 1997)].reset_index(drop=True)
# print(np.mean(data_year1996))
# print(np.mean(data_year1997))
# data_year = data_all[['date','CIVIL','IND']]
# data_year = data_year.set_index('date')
# print(data_year)
# data_year.plot()
# plt.show()
'''
class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(
                units=hp.Int("units1", min_value=32, max_value=512, step=32),
                activation=hp.Choice('activation1',values=['relu','selu','elu'])))
        model.add(Dropout(hp.Choice('dropout1',values =[0.0, 0.2, 0.3, 0.4, 0.5])))
        
        model.add(Dense(10, activation="softmax"))
        
        model.compile(
            optimizer=Adam(lr = hp.Choice('learning_rate', values = [1e-2,5e-3,1e-3,5e-4,1e-4])), 
            loss="sparse_categorical_crossentropy", metrics=["accuracy"],
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [16, 32]),
            **kwargs,
        )

kerastuner = kt.RandomSearch(
    MyHyperModel(),
    objective="val_accuracy",
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="tune_hypermodel",
)
# kerastuner = kt.Hyperband(get_model,
#                           directory = 'my_dir',
#                           objective = 'val_acc',
#                           max_epochs = 6,
#                           project_name = 'kerastuner-mnist2')

kerastuner.search(x_train,y_train,
                  validation_data=(x_test,y_test),
                  epochs=5)
best_hps = kerastuner.get_best_hyperparameters(num_trials=2)[0]

print('best parameter - units1 : ',best_hps.get('units1'))
# print('best parameter - units2 : ',best_hps.get('units2'))
# print('best parameter - units3 : ',best_hps.get('units3'))
# print('best parameter - units4 : ',best_hps.get('units4'))

print('best parameter - dropout1 : ',best_hps.get('dropout1'))
# print('best parameter - dropout2 : ',best_hps.get('dropout2'))
# print('best parameter - dropout3 : ',best_hps.get('dropout3'))
# print('best parameter - dropout4 : ',best_hps.get('dropout4'))

print('best parameter - activation1 : ',best_hps.get('activation1'))
# print('best parameter - activation2 : ',best_hps.get('activation2'))
# print('best parameter - activation3 : ',best_hps.get('activation3'))
# print('best parameter - activation4 : ',best_hps.get('activation4'))

print('best parameter - learning_rate : ',best_hps.get('learning_rate'))
'''







