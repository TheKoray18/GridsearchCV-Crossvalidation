# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 22:53:30 2020

@author: Koray
"""
import pandas as pd 

#Datamızı Çağırdık

data=pd.read_csv("kc_house_data.csv")

#%% Data Preprocessing
 
data=data.iloc[:,2:]

waterfront=data.iloc[:,6:7]

data=data.drop(["waterfront"],axis=1)

print(data.info)
print(data.isnull())

#%% Train- Test Olarak Ayrılması

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(data,waterfront, test_size=0.33,random_state=0)


#%% Standardizasyon

from sklearn.preprocessing import StandardScaler

s_scaler=StandardScaler()

x_train=s_scaler.fit_transform(x_train)

x_test=s_scaler.fit_transform(x_test)
#%% Datalarımızın Boyutları

print("x_train shape:",x_train.shape)
print("x_test shape:",x_test.shape)
print("y_train shape:",y_train.shape)
print("y_test shape:",y_test.shape)

#%% Sinir Ağımızı Oluşturuyoruz

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

def sınıragı(init='uniform',optimizer='Adam'):
    
    model=Sequential()
    
    model.add(Dense(units=16, activation='relu', init=init,input_dim=18))
    model.add(Dense(units=16, activation='relu', init=init))
    model.add(Dense(units=16, activation='relu', init=init))
    model.add(Dense(units=1,  activation='sigmoid',init=init))
    
    model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    
    return model

sınır_model=KerasClassifier(build_fn=sınıragı,verbose=1) 

#%% GridSearchCV

#Modelimizin karşılaştıracağı Parametreler

optimizers=['Adam','rmsprop']
epochs=[10,20,40]   
batch_size=[100,200,300]

#Parametreleri dictionary yaptık

parameter=dict(optimizer=optimizers,epochs=epochs,batch_size=batch_size)

from sklearn.model_selection import GridSearchCV

gs=GridSearchCV(estimator=sınır_model,param_grid=parameter,scoring='accuracy',cv=2)

grid_search=gs.fit(x_train,y_train)

best_score=grid_search.best_score_
best_parameters=grid_search.best_params_

#En iyi skoru ve modelimiz için en iyi parametrelerimize bakıyoruz

print("Best Score:",best_score)
print("Best Parameters:",best_parameters)

#%% Crossvalidation

from sklearn.model_selection import cross_val_score

cross=cross_val_score(estimator=sınır_model, X=x_train,y=y_train,cv=4,verbose=0)

#Modelimizin Ortalamasına ve Standart Sapmasına bakıyoruz

print("Mean:",cross.mean())
print("Std:",cross.std())




























