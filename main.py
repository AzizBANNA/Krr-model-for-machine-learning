import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 
from sklearn.metrics import mean_squared_error
import math as m
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from joblib import dump, load
import json


df = pd.read_csv('__files/wave.csv', sep = ','  )
X= df['x'].to_numpy()
y= df['y'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 100) 
'''dftest = pd.DataFrame(X_test , y_test)
dftrain= pd.DataFrame(X_train,y_train)
dftrain.to_csv('train.csv' , index=False)
dftest.to_csv('test.csv', index=False)
'''
dftest = pd.DataFrame()
dftrain= pd.DataFrame()
dftest['x']= X_test
dftest['y']= y_test
dftrain['x']= X_train
dftrain['y']= y_train
dftrain.to_csv('train.csv' , index=None)
dftest.to_csv('test.csv', index=None)
model= GridSearchCV(KernelRidge(kernel="rbf"),param_grid={"alpha": [0.001 , 0.1 , 1], "gamma": [0.001 , 0.1 , 1]} , cv=5 , refit=True)
med = model.fit(X_train.reshape(-1,1), y_train.reshape(-1,1)) 
dump(med, 'model.joblib')
y_pred=med.predict(X_test.reshape(-1,1))
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test,y_pred)
r2= r2_score(y_test,y_pred)
mse1 = mean_squared_error(y_train, med.predict(X_train.reshape(-1,1)))
mae1 = mean_absolute_error(y_train,med.predict(X_train.reshape(-1,1)))
r21= r2_score(y_train,med.predict(X_train.reshape(-1,1)))
mydict= dict()
mydict= {
"test_mae" : mae,
"test_mse" : mse,
"test_r2" : r2,
"train_mae": mae1,
"train_mse" : mse1,
"train_r2" : r21
}
with open("scores.json", "w") as outfile:
    json.dump(mydict, outfile)

fig = plt.figure()
xplot= np.linspace(-10,10,10000)
fx= np.exp(-((xplot*xplot)/16))*np.cos(4*xplot)
ax = fig.add_subplot() 
ax.plot(xplot, fx , color ='blue', label='true function')
y_predplot= med.predict(xplot.reshape(-1,1))
plt.plot(xplot,y_predplot , color='orange' , label='prediction')
ax.scatter(X_train, y_train , color='blue' , label= 'train')
ax.scatter(X_test, y_test , color= 'orange', label ='test')
ax.set_title(f'MSE :{mse}, MAE :{mae}, R2 :{r2}' ,fontsize=8)
plt.legend()
plt.savefig('plot.pdf')
plt.show()


