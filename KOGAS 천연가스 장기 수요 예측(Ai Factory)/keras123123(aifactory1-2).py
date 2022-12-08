import keras_tuner as kt
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import time
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,LSTM,Bidirectional
from keras.layers import LayerNormalization
# from keras.optimizer_v2 import adam # tf 2.7 
from keras.optimizers import Adam     # tf 2.8 이상
from sklearn.metrics import accuracy_score,mean_absolute_error,mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

path = 'D:\\ai_data/'

data_all = pd.read_csv(path+'data_all.csv')
mean_data = np.load(path+'mean_data.npy')

# data_all = data_all.drop(['RP','Total','amount_of_gas','hightemp'],axis=1)
data_all = data_all.drop(['date','RP','Total','amount_of_gas','hightemp'],axis=1)
data_all = data_all.fillna(0)
# print(data_all.isnull().sum())

size = 13
def split_data(dataset,size):
    aaa = []
    for i in range(len(dataset)- size +1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)
x = data_all
y = data_all[['CIVIL','IND']]


bbb = split_data(y, size)
ccc = split_data(x ,size)

x1 = ccc[:,:12].astype(float)
y1 = bbb[:,-1].astype(float)
# print(x1.shape,y1.shape) # (288, 12, 27) (288, 2)

x_train,x_test,y_train,y_test = train_test_split(x1,y1,train_size=0.9,
                                                 shuffle=False)
# print(x_train.shape,y_train.shape) # (259, 12, 27) (259, 2)
# print(x_test.shape,y_test.shape) # (29, 12, 27) (29, 2)

class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(Bidirectional(LSTM(units=hp.Int("units1", min_value=32, max_value=512, step=32),
                                     return_sequences=True,
                                     input_shape=(x1.shape[1],x1.shape[2]))))
        model.add(LayerNormalization())
        model.add(Bidirectional(LSTM(units=hp.Int("units2", min_value=32, max_value=512, step=32))))
        model.add(LayerNormalization())
        # model.add(Flatten())
        model.add(Dense(
                units=hp.Int("units3", min_value=32, max_value=512, step=32),
                activation=hp.Choice('activation1',values=['relu','selu','elu'])))
        model.add(Dropout(hp.Choice('dropout1',values =[0.0, 0.2, 0.3, 0.4, 0.5])))
        model.add(Dense(
                units=hp.Int("units4", min_value=32, max_value=512, step=32),
                activation=hp.Choice('activation2',values=['relu','selu','elu'])))
        model.add(Dropout(hp.Choice('dropout2',values =[0.0, 0.2, 0.3, 0.4, 0.5])))        
        model.add(Dense(2))
        
        model.compile(
            optimizer=Adam(lr = hp.Choice('learning_rate', values = [1e-2,5e-3,1e-3,5e-4,1e-4])), 
            loss="mae", metrics=["mape"],
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [8, 16, 32]),
            **kwargs,
        )
start_time = time.time()
kerastuner = kt.RandomSearch(
    MyHyperModel(),
    objective="val_loss",
    max_trials=10,
    overwrite=True,
    directory="my_dir",
    project_name="tune_hypermodel1",
)
# kerastuner = kt.Hyperband(get_model,
#                           directory = 'my_dir',
#                           objective = 'val_acc',
#                           max_epochs = 6,
#                           project_name = 'kerastuner-mnist2')

kerastuner.search(x_train,y_train,
                  validation_data=(x_test,y_test),
                  epochs=50)
best_hps = kerastuner.get_best_hyperparameters(num_trials=30)[0]
end_time = time.time()
print('best parameter - units1 : ',best_hps.get('units1'))
print('best parameter - units2 : ',best_hps.get('units2'))
print('best parameter - units3 : ',best_hps.get('units3'))
print('best parameter - units4 : ',best_hps.get('units4'))

print('best parameter - dropout1 : ',best_hps.get('dropout1'))
print('best parameter - dropout2 : ',best_hps.get('dropout2'))
# print('best parameter - dropout3 : ',best_hps.get('dropout3'))
# print('best parameter - dropout4 : ',best_hps.get('dropout4'))

print('best parameter - activation1 : ',best_hps.get('activation1'))
print('best parameter - activation2 : ',best_hps.get('activation2'))
# print('best parameter - activation3 : ',best_hps.get('activation3'))
# print('best parameter - activation4 : ',best_hps.get('activation4'))

print('best parameter - learning_rate : ',best_hps.get('learning_rate'))
print('걸린시간',end_time-start_time)

from keras.callbacks import EarlyStopping,ReduceLROnPlateau
es = EarlyStopping(patience=10,mode='min',monitor='val_loss',verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=5,mode='auto',verbose=1,
                              factor=0.5)
model = kerastuner.hypermodel.build(best_hps)

history = model.fit(x_train,y_train,
          validation_split=0.2,epochs=5,
          callbacks=[es,reduce_lr]
          )
loss= model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
print('Train_LOSS : ',loss)

print('MAE :',mean_absolute_error(y_test,y_predict))
print('MAPE :',mean_absolute_percentage_error(y_test,y_predict))

# Best val_loss So Far: 154244.9375
# Total elapsed time: 00h 01m 08s
# best parameter - units1 :  448
# best parameter - units2 :  320
# best parameter - units3 :  448
# best parameter - units4 :  352
# best parameter - dropout1 :  0.4
# best parameter - dropout2 :  0.3
# best parameter - activation1 :  elu
# best parameter - activation2 :  selu
# best parameter - learning_rate :  0.005
# 걸린시간 68.69731831550598