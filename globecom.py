import numpy as np
import optunity
import optunity.metrics
import time
import pandas as pd
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.svm import SVC,SVR
from sklearn import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,Dropout
import scipy.stats as stats
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
data = pd.read_csv("C:\\Users\\pshkr\\Downloads\\DL\\DL\\features_after_corr.csv")
y_lB = data['label']
y_lB.tolist().count(7)
list_train_row = [312, 12, 9, 464, 24, 1, 8, 3444, 35, 60, 32, 12, 217, 2045]
row_till = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
train_list =[]
test_list = []
for i in range(len(data)):
    row_label = int(data.loc[i].at['label'])
    if row_till[row_label] <= list_train_row[row_label]:
        train_list.append(list(data.loc[i]))
        row_till[row_label] =  row_till[row_label]+1
    else:
        test_list.append(list(data.loc[i]))

train_df= pd.DataFrame(train_list)     
test_df= pd.DataFrame(test_list) 
x_train = train_df.copy()
x_train = train_df.drop(1024, axis=1) 
x_test = test_df.copy()
x_test = test_df.drop(1024,axis=1)
y_train =  train_df.iloc[:,-1]
y_test =  test_df.iloc[:,-1]
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
data = data.drop("label",axis=1)
data_com = sc.fit_transform(data_com)
x_test = sc.transform(x_test)
y_train_det = y_train.replace(to_replace=[0,1,2,3,4,5,6,7,8,9,10,11,12,13],
                                    value=[1,1,1,1,1,1,1,0,1,1,1,1,1,1])

y_test_det = y_test.replace(to_replace=[0,1,2,3,4,5,6,7,8,9,10,11,12,13],
                                    value=[1,1,1,1,1,1,1,0,1,1,1,1,1,1])

x_train = x_train.reshape((6658,32, 32, 1))
x_test = x_test.reshape((1657,32, 32, 1))
data_com = data_com.reshape((8315,32, 32, 1))
X= X.reshape((1797,8,8, 1))
x_test = x_test.reshape((1657,32, 32, 1))
label = data.iloc[:,-1]
label_y = data.iloc[:,-1]
d = datasets.load_digits()
X = d.data
y = d.target
datasets.load_digits()
clf = RandomForestClassifier()
clf.fit(X,y)
scores = cross_val_score(clf, X, y, cv=3,scoring='accuracy')
print("Accuracy:"+ str(scores.mean()))
test=0
scores=[]
def ANN(optimizer = 'sgd',neurons=120,batch_size=32,epochs=25,activation='relu',patience=2,loss='sparse_categorical_crossentropy'):
    model = Sequential()
    model.add(Conv2D(filters = 6, 
                 kernel_size = 5, 
                 strides = 1, 
                 activation = activation, 
                 input_shape = (32,32,1)))
    model.add(MaxPooling2D(pool_size = 2, strides = 2))
    model.add(Conv2D(filters = 16, 
                 kernel_size = 5, 
                 strides = 1, 
                 activation = activation))
    model.add(MaxPooling2D(pool_size = 2, strides = 2))
    model.add(Flatten())    
    model.add(Dense(units=neurons, activation=activation))
    model.add(Dense(units=neurons, activation=activation))
    model.add(Dense(units=15,activation='softmax'))
    
    model.summary()
    
    
    model.compile(optimizer = optimizer, loss=loss)
    early_stopping = EarlyStopping(monitor="loss", patience = patience)# early stop patience
    history = model.fit(data_com,label_y,
              batch_size=batch_size,
              epochs=epochs,
              callbacks = [early_stopping],
              verbose=0) #verbose set to 1 will show the training process
    # score = model.predict(x_test)
    # y_pred = np.argmax(score,axis=1)
    return model
clf = KerasClassifier(build_fn=ANN, verbose=0)
scores = cross_val_score(clf,x,label, cv=3,scoring='accuracy')
print("Accuracy:"+ str(scores.mean()))
score = model.predict(x_test)
rf_params = {
    'optimizer': ['adam','sgd'],
    'activation': ['relu','tanh'],
    'batch_size': [16,32],
    'neurons':[80,120],
    'epochs':[20,25],
    'patience':[2,5]
}
import sys
clf = KerasClassifier(build_fn=ANN, verbose=0)
grid = GridSearchCV(clf,rf_params,cv=3,scoring='accuracy',verbose=3,error_score=0)
t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
sys.stdout = open('pikas.txt', 'w')
grid.fit(data_com,label)
grid.fit(x_train, y_train)
sys.stdout.close()
t2 = time.localtime()
last_time = time.strftime("%H:%M:%S", t2)
print(grid.best_params_)

score_det = model.predict(x_test)
y_pred_det = np.argmax(score_det,axis=1)
print(accuracy_score(y_test, y_pred_det))

search = {
    'optimizer':[0,2],
    'activation':[0,2],
    'batch_size': [0, 2],
    'neurons': [10, 100],
    'epochs': [20, 50],
    'patience': [3, 20],
         }
@optunity.cross_validated(x=data_com, y=label, num_folds=3)
def performance(x_train, y_train,optimizer=None,activation=None,batch_size=None,neurons=None,epochs=None,patience=None):
    # fit the model
    if optimizer<1:
        op='adam'
    else:
        op='sgd'
    if activation<1:
        ac='relu'
    else:
        ac='tanh'
    if batch_size<1:
        ba=16
    else:
        ba=32
    model = ANN(optimizer=op,
                activation=ac,
                batch_size=ba,
                neurons=int(neurons),
                epochs=int(epochs),
                patience=int(patience)
                                  )
    clf = KerasClassifier(build_fn=ANN, verbose=0)
    scores=np.mean(cross_val_score(clf,data_com,label, cv=3, 
                                    scoring="accuracy"))

    return scores

optimal_configuration, info, _ = optunity.maximize(performance,
                                                  solver_name='particle swarm',
                                                  num_evals=20,
                                                   **search
                                                  )
print(optimal_configuration)
print("MSE:"+ str(info.optimum))




data=X
labels=y.tolist()
search = {
    'optimizer':[0,2],
    'activation':[0,2],
    'batch_size': [16,32],
    'neurons': [80, 120],
    'epochs': [20,25],
    'patience': [2,5],
         }
@optunity.cross_validated(x=data_com, y=label, num_folds=3)
def performance(x_train, y_train, x_test, y_test,optimizer=None,activation=None,batch_size=None,neurons=None,epochs=None,patience=None):
    # fit the model
    if optimizer<1:
        op='adam'
    else:
        op='sgd'
    if activation<1:
        ac='relu'
    else:
        ac='tanh'
    if batch_size<1:
        ba=16
    else:
        ba=32
    model = ANN(optimizer=op,
                activation=ac,
                batch_size=ba,
                neurons=int(neurons),
                epochs=int(epochs),
                patience=int(patience)
                                  )
    clf = KerasClassifier(build_fn=ANN, verbose=0)
    scores=np.mean(cross_val_score(clf,data_com, label_y, cv=3, 
                                    scoring="accuracy"))

    return scores
optimal_configuration, info, _ = optunity.maximize(performance,
                                                  solver_name='particle swarm',
                                                  num_evals=2,
                                                   **search
                                                  )
print(optimal_configuration)
print("MSE:"+ str(info.optimum))
