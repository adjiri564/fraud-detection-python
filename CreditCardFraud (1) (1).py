# Libraries used for the project
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve, accuracy_score
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score, precision_score
from sklearn.preprocessing import StandardScaler
from pylab import rcParams
from tensorflow.keras.models import  Sequential,Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Flatten, BatchNormalization, Conv1D
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import regularizers


#Loading the dataset 
cc = pd.read_csv("creditcard.csv",sep = ',')
cc.head()

#General Descriptive Statistics
cc.describe()

# A bar graph representing the normal and fraud transactions in the dataset
LABELS = ["Normal", "Fraud"]
rcParams['figure.figsize'] = 14,8
count_classes = cc.value_counts(cc['Class'], sort = True)
count_classes.plot(kind = 'bar',  color = ['blue', 'red'], rot = 0)
plt.title("Original Class Distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")

#The number and percentage of fraud transactions in the dataset
neg, pos = np.bincount(cc['Class'])
total = neg + pos
print('Total: {}\n Fraud: {} ({:.3f}% of total)\n'.format(total, pos, 100*pos / total))

cc.mean()
cc.median()
cc.mode()

#Checking for duplicated items in the dataset
cc.duplicated(subset=None, keep= 'first')

#Checking for missing values in the dataset
cc.isnull()


#Standardising the dataset
cc['Time'] = StandardScaler().fit_transform(cc['Time'].values.reshape(-1,1))
cc['Amount'] = StandardScaler().fit_transform(cc['Amount'].values.reshape(-1,1))

#Dividing the dataset into independent instances(from Time-Amount) and dependent instances(Class)
X = cc.drop(['Class'], axis =1)
y = cc['Class']

#Splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#Balancing the train dataset
from mahakil import *
mahak = MAHAKIL()
x_re, y_re = mahak.fit_sample(X_train,y_train)
x_re.shape, y_re.shape

#comparing the 'Class' of the original dataset and the 'Class' of the resampled dataset
from collections import Counter
print('Train dataset shape{}'.format(Counter(y_train)))
print('Resampled Train dataset shape{}'.format(Counter(y_re)))

# The number of normal and fraudulent transactions in the test dataset
from collections import Counter
print('Test dataset shape{}'.format(Counter(y_test)))

# A bar graph representing the new balanced train dataset
count_classes = pd.value_counts(y_re, sort = True)
count_classes.plot(kind = 'bar', color = ['blue', 'red'], rot = 0)
plt.title("Re-sampled Class Distribution")
plt.xticks(range(2), LABELS,)

plt.xlabel("Class")
plt.ylabel("Frequency")

X_test = np.array(X_test, dtype=float)
y_test = np.array(y_test, dtype=float)

#Reshaping the train and test dataset to meet the requirements of the CNN and LSTM model
Xtrain = x_re.reshape(x_re.shape[0], x_re.shape[1], 1)
Xtest =  X_test.reshape( X_test.shape[0],  X_test.shape[1], 1)

# Using statified kfold on the train dataset
kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)
cvscores = []





#The CNN Model
for train, test in kfold.split(Xtrain, y_re):
     
    x_train = Xtrain[train]
    y_train = y_re[train]
    x_traintest = Xtrain[test]
    y_traintest = y_re[test]
    
    model1 = Sequential()
    model1.add(Conv1D(32, 2, activation='relu', input_shape = x_train[0].shape))
    model1.add(BatchNormalization())
    model1.add(Dropout(0.2))

    model1.add(Conv1D(64, 2, activation='relu'))
    model1.add(BatchNormalization())
    model1.add(Dropout(0.5))

    model1.add(Flatten())
    model1.add(Dense(64, activation='relu'))
    model1.add(Dropout(0.5)) 

    model1.add(Dense(1, activation='sigmoid'))

    model1.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    history = model1.fit(x_train, y_train,validation_data=(x_traintest,y_traintest),epochs=5,verbose=1)
    
    scores = model1.evaluate(x_traintest, y_traintest, verbose=0)
    print("%s: %.4f%%" % (model1.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    
    

# The CNN model predicting normal and fraud transactions using Xtest
y_pred = model1.predict(Xtest)
y_pred = (y_pred>0.5)

#The confusion matrix for the CNN model
LABELS = ['Normal', 'Fraud'] 
conf_matrix = confusion_matrix(y_test, y_pred) 
plt.figure(figsize =(12, 12)) 
sns.heatmap(conf_matrix, xticklabels = LABELS,  
            yticklabels = LABELS, annot = True, fmt ="d"); 
plt.title("Confusion matrix") 
plt.ylabel('True class')
plt.xlabel('Predicted class') 
plt.show()

#Calculating for accuracy, precision, f1 score and recall score for the CNN model
accuracy1 = accuracy_score(y_test, y_pred)
precision1 = precision_score(y_test, y_pred)
f1score1 = f1_score(y_test, y_pred) 
recallScore1 = recall_score(y_test, y_pred)
print('Accuracy: %f' % accuracy1)
print('Precision: %f' % precision1)
print('F1 score: %f' % f1score1)
print('Recall score: %f' % recallScore1)





#The LSTM model
for train, test in kfold.split(Xtrain, y_re):
    
    x_train = Xtrain[train]
    y_train = y_re[train]
    x_traintest = Xtrain[test]
    y_traintest = y_re[test]
    
    model2 = Sequential()

    model2.add(LSTM(20, input_shape=Xtrain.shape[1:], kernel_initializer='lecun_uniform', activation='relu',
    kernel_regularizer=regularizers.l1(0.1), 
    recurrent_regularizer=regularizers.l1(0.01), bias_regularizer=None, 
    activity_regularizer=None, dropout=0.2, recurrent_dropout=0.2))

    model2.add(Dense(1, kernel_initializer='lecun_uniform', activation='sigmoid'))

    model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    history = model2.fit(x_train, y_train,validation_data=(x_traintest,y_traintest),epochs=5,verbose=1)
    
    scores = model2.evaluate(x_traintest, y_traintest, verbose=0)
    print("%s: %.4f%%" % (model2.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    


# The LSTM model predicting normal and fraud transactions using Xtest
y_pred1 = model2.predict(Xtest)
y_pred1 = (y_pred1>0.5)

#The confusion matrix for the LSTM model
LABELS = ['Normal', 'Fraud'] 
conf_matrix = confusion_matrix(y_test, y_pred1) 
plt.figure(figsize =(12, 12)) 
sns.heatmap(conf_matrix, xticklabels = LABELS,  
            yticklabels = LABELS, annot = True, fmt ="d",cmap = 'gist_gray'); 
plt.title("Confusion matrix") 
plt.ylabel('True class')
plt.xlabel('Predicted class') 
plt.show()

#Calculating for accuracy, precision, f1 score and recall score for the LSTM model
accuracy2 = accuracy_score(y_test, y_pred1)
precision2 = precision_score(y_test, y_pred1)
f1score2 = f1_score(y_test, y_pred1)
recallScore2 = recall_score(y_test, y_pred1)
print('Accuracy: %f' % accuracy2)
print('Precision: %f' % precision2)
print('F1 score: %f' % f1score2)
print('Recall score: %f' % recallScore2)





# The Autoencoder Model
for train, test in kfold.split(x_re, y_re):
    
    x_train = x_re[train]
    y_train = y_re[train]
    x_traintest = x_re[test]
    y_traintest = y_re[test]
    
    nb_epoch = 5
    batch_size = 128
    input_dim = x_train.shape[1] #num of columns, 30
    encoding_dim = 14
    hidden_dim = int(encoding_dim / 2) #i.e. 7
    learning_rate = 1e-7

    input_layer = Input(shape=(input_dim, ))
    encoder = Dense(encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(learning_rate))(input_layer)
    encoder = Dense(hidden_dim, activation="relu")(encoder)
    decoder = Dense(hidden_dim, activation='tanh')(encoder)
    decoder = Dense(input_dim, activation='relu')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    
    autoencoder.compile(metrics=['accuracy'], loss='mean_squared_error', optimizer='adam')
    
    cp = ModelCheckpoint(filepath="autoencoder1_fraud.h5", save_best_only=True, verbose=0)

    tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    
    history = autoencoder.fit(x_train, x_train, epochs=nb_epoch, batch_size=batch_size, shuffle=True, 
                          validation_data=(x_traintest, x_traintest), verbose=1, callbacks=[cp, tb]).history
    
    scores = autoencoder.evaluate(x_traintest, x_traintest, verbose=0)
    print("%s: %.4f%%" % (autoencoder.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    
    

test_x_predictions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - test_x_predictions, 2), axis=1)
error_df = pd.DataFrame({'Reconstruction_error': mse,
                        'True_class': y_test})

#The confusion matrix for the Autoencoder model

threshold_fixed = 5
LABELS = ["Normal","Fraud"]
y_pred2 = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.True_class, y_pred2)

plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d",cmap = 'BuGn');
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

#Calculating for accuracy, precision, f1 score and recall score for the Auto encoders model
accuracy3 = accuracy_score(y_test, y_pred2)
precision3 = precision_score(y_test, y_pred2)
f1score3 = f1_score(y_test, y_pred2)
recallScore3 = recall_score(y_test, y_pred2) 
print('Accuracy: %f' % accuracy3)
print('Precision: %f' % precision3)
print('F1 score: %f' % f1score3)
print('Recall score: %f' % recallScore3)



