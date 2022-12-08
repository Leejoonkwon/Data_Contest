
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

data_c = data_c.rename(columns={'제조업부가가치액':'MFG'})
# print(data_c.columns)


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







