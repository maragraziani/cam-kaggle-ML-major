
# coding: utf-8

# In[1]:

from sklearn.svm import SVR
from PIL import Image
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import pandas as pd
import csv
from PIL import Image
import numpy as np
from numpy import *
from sklearn import preprocessing as prep
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sklearn.cross_validation as crval
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import fetch_mldata
from scipy.interpolate import interp1d
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import pylab
import csv
from sklearn.cross_validation import *
from sklearn.linear_model import * 
from sklearn import neighbors
#file = 'reg_train_in.csv'
#tr_in1 = np.genfromtxt(file, delimiter = ',', filling_values = np.nan, names = True, dtype = type(None))
#print len(tr_in1)


tr_in=[]
fp=open('reg_train_in.csv','r')
trash=fp.readline()
#print trash
lines=fp.readlines()
print len(lines)
for line in lines:
    data=line.split(',')
    cleaned_data=data[1:len(data)]
    tr_in.append(cleaned_data)

tr_out=[]
fp=open('reg_train_out.csv','r')
trash=fp.readline()
#print trash
lines=fp.readlines()
print len(lines)
for line in lines:
    data=line.split(',')
    cleaned_data=data[1:len(data)]
    tr_out.append(cleaned_data)

te_in=[]
fp=open('reg_test_in.csv','r')
trash=fp.readline()
#print trash
lines=fp.readlines()
print len(lines)
for line in lines:
    data=line.split(',')
    cleaned_data=data[1:len(data)]
    te_in.append(cleaned_data)


# In[2]:

# build arrays

tr_in=np.asarray(tr_in)
tr_in=tr_in.astype(np.float)
tr_in=np.ravel(tr_in)
print(shape(tr_in))  
tr_in= tr_in.reshape(34200,14)

tr_out=np.asarray(tr_out)
#tr_out=tr_in.astype(np.float)
tr_out=np.ravel(tr_out)
print(shape(tr_out))

te_in=np.asarray(te_in)
te_in=te_in.astype(np.float)
te_in=np.ravel(te_in)
print(shape(te_in))  
te_in= te_in.reshape(1800,14)


# In[3]:

print(len(te_in))
nan1=te_in[400:600,0].reshape(200,1)
nan2=te_in[1300:1400,0].reshape(100,1)
nan3=te_in[1600:1700,0].reshape(100,1)

training=np.zeros((1200,14))
training[0:400]=te_in[0:400]
training[400:1100]=te_in[600:1300]
training[1100:1200]=te_in[1700:len(te_in)]

#for t in training:
#    print t
#training=training[0]
print(mean(training))


# In[4]:

n_neighbors = 5
ynan1=np.zeros((14,200))
ynan2=np.zeros((14,100))
ynan3=np.zeros((14,100))
for j in range(13):
    train_data=training[:,0].ravel().reshape(1200,1)
    #print(np.shape(train_data))
    train_labels_column=training[:,j].ravel().reshape(1200,1)
    #print np.shape(train_labels_column)
    #print shape(nan1)
   
    for i, weights in enumerate(['uniform', 'distance']):
        knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights) 
        model = knn.fit(train_data, train_labels_column)
        predictions=model.predict(nan1)  
        
        y1 = model.predict(nan1)
    
        ynan1[j]=y1.ravel()
        ynan2[j] = model.predict(nan2).ravel()
        ynan3[j] = model.predict(nan3).ravel()
     
ynan1=np.transpose(ynan1)
print(np.shape(ynan1))
ynan2=np.transpose(ynan2)
print(np.shape(ynan2))
ynan3=np.transpose(ynan3)
print(np.shape(ynan3))
print mean(ynan1)
print mean(ynan2)
print mean(ynan3)


# In[5]:

#test=np.zeros(len(te_in),14)
test=te_in
test[400:600]=ynan1
test[1300:1400]=ynan2
test[1600:1700]=ynan3
print mean(test)
print mean(tr_in)


# In[6]:

#NORMALIZING DATA
selX = tr_in
std_scal=prep.StandardScaler()
selX_scaled=std_scal.fit_transform(selX)
print('it had mean: ' +str(mean(selX))+' and std:'+ str(std(selX)))
print('NOW it has mean: '+str(mean(selX_scaled))+' and std: '+str(std(selX_scaled)))

selX = test
std_scal=prep.StandardScaler()
test_scaled=std_scal.fit_transform(selX)
print('it had mean: ' +str(mean(selX))+' and std:'+ str(std(selX)))
print('NOW it has mean: '+str(mean(test_scaled))+' and std: '+str(std(test_scaled)))

test=test_scaled
tr_in=selX_scaled



# In[ ]:

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
#y_rbf = svr_rbf.fit(X, y).predict(X)
#y_lin = svr_lin.fit(X, y).predict(X)
#y_poly = svr_poly.fit(X, y).predict(X)

print "Predict rbf..."
y_rbf = svr_rbf.fit(tr_in, tr_out).predict(test)
print "Predict lin..."
#y_lin = svr_lin.fit(tr_in, tr_out).predict(test)
print "Predict poly..."
#y_poly = svr_poly.fit(tr_in, tr_out).predict(test)


# In[ ]:

fp=open('answers_rbf.txt','w')
fp.write('Point_ID,Output\n')
i=1

for a in y_rbf:
    string=str(a)
    fp.write(str(i)+','+string+'\n')
    i+=1
if i%1000==0:
    print str(i)
fp.close()


# In[ ]:

n_neightbors=5
print np.shape(tr_in)
print np.shape(tr_out.ravel().reshape(34200,1))

print np.shape(test)

for i, weights in enumerate(['uniform', 'distance']):
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights) 
    model = knn.fit(tr_in, tr_out)
    predictions=model.score(test)  
    
print predictions


# In[ ]:

fp = open('acc_svr_rbf.txt')


# In[ ]:

lw = 2
plt.scatter(X, y, color='darkorange', label='data')
plt.hold('on')
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()


# In[ ]:

X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
clf = svm.SVR()
clf.fit(X, y) 
SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
clf.predict([[1, 1]])
array([ 1.5])

