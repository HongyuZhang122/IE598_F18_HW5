# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 14:51:49 2018

@author: hongy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'                      
                      'machine-learning-databases/wine/wine.data',                      
                      header=None)

df_wine.columns = ['Class label', 'Alcohol', 
                   'Malic acid', 'Ash', 
                   'Alcalinity of ash', 
                   'Magnesium', 'Total phenols', 
                   'Flavanoids', 'Nonflavanoid phenols', 
                   'Proanthocyanins', 
                   'Color intensity', 'Hue', 
                   'OD280/OD315 of diluted wines', 
                   'Proline']

print(df_wine)


#import the package and create the scatterplot matrix 
cols = df_wine.columns

sns.pairplot(df_wine[cols], size=2.5) 
plt.tight_layout() 
plt.show()


#use Seaborn's heatmap function to plot the correlation matrix array as a heat map
cm = np.corrcoef(df_wine[cols].values.T) 
sns.set(font_scale=1.5) 
hm = sns.heatmap(cm,             
                 cbar=True, 
                 annot=True, 
                 square=True, 
                 fmt='.2f',               
                 annot_kws={'size': 7}, 
                 yticklabels=cols, 
                 xticklabels=cols) 

plt.tick_params(axis='x',labelsize=7)
plt.tick_params(axis='y',labelsize=7)
plt.tight_layout()
plt.show()

#from sklearn.model_selection import train_test_split 
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values 
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, 
                                                     stratify=y, 
                                                     random_state=42)
 
# standardize the features 
#from sklearn.preprocessing import StandardScaler 
sc = StandardScaler() 
X_train_std = sc.fit_transform(X_train) 
X_test_std = sc.transform(X_test) 


#obtain the eigenpairs of the Wine covariance matrix
cov_mat = np.cov(X_train_std.T) 
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat) 
print('\nEigenvalues \n%s' % eigen_vals)

tot = sum(eigen_vals) 
var_exp = [(i / tot) for i in 
           sorted(eigen_vals, reverse=True)] 
cum_var_exp = np.cumsum(var_exp) 

#import matplotlib.pyplot as plt 
plt.bar(range(1,14), var_exp, alpha=0.5, align='center', 
        label='individual explained variance') 
plt.step(range(1,14), cum_var_exp, where='mid', 
         label='cumulative explained variance') 
plt.ylabel('Explained variance ratio') 
plt.xlabel('Principal component index') 
plt.legend(loc='best') 
plt.show()

#We start by sorting the eigenpairs by decreasing order of the eigenvalues
 # Make a list of (eigenvalue, eigenvector) tuples 
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) 
               for i in range(len(eigen_vals))] 
# Sort the (eigenvalue, eigenvector) tuples from high to low 
eigen_pairs.sort(key=lambda k: k[0], reverse=True) 

w = np.hstack((eigen_pairs[0][1][:, np.newaxis], 
               eigen_pairs[1][1][:, np.newaxis])) 
print('Matrix W:\n', w)




# Training Simple Machine Learning Algorithms for Classification
from matplotlib.colors import ListedColormap 
def plot_decision_regions(X, y, classifier, resolution=0.02):    
    # setup marker generator and color map    
    markers = ('s', 'x', 'o', '^', 'v')    
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')    
    cmap = ListedColormap(colors[:len(np.unique(y))])    
    # plot the decision surface    
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1    
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1    
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),                           
                           np.arange(x2_min, x2_max, resolution))    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)    
    Z = Z.reshape(xx1.shape)    
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)    
    plt.xlim(xx1.min(), xx1.max())    
    plt.ylim(xx2.min(), xx2.max())    
    # plot class samples    
    for idx, cl in enumerate(np.unique(y)):        
        plt.scatter(x=X[y == cl, 0],                     
                    y=X[y == cl, 1],                    
                    alpha=0.6,                     
                    c=cmap(idx),                    
                    edgecolor='black',                    
                    marker=markers[idx],                     
                    label=cl) 


 
from sklearn.linear_model import LogisticRegression     
lr = LogisticRegression() 
lr.fit(X_train_std,y_train)
y_train_pred = lr.predict(X_train_std) 
y_test_pred = lr.predict(X_test_std) 
print("Logistic regression classifier's train accuracy:")
print(accuracy_score(y_train,y_train_pred))
print("Logistic regression classifier's test accuracy:")
print(accuracy_score(y_test,y_test_pred))

from sklearn.svm import SVC 
svm = SVC(kernel='linear', C=1.0, random_state=1) 
svm.fit(X_train_std, y_train)
y_train_pred = svm.predict(X_train) 
y_test_pred = svm.predict(X_test) 
print("SVM classifier's train accuracy:")
print(accuracy_score(y_train,y_train_pred))
print("SVM classifier's test accuracy:")
print(accuracy_score(y_test,y_test_pred))
    
###PCA & LR
print('\n' * 3)
print("<1.1> refit logistic regression classifier on the PCA transformed datasets.")
from sklearn.decomposition import PCA 
pca = PCA(n_components=2)
lr = LogisticRegression() 
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std) 
lr.fit(X_train_pca, y_train)
y_train_pred = lr.predict(X_train_pca) 
y_test_pred = lr.predict(X_test_pca)
plot_decision_regions(X_train_pca, y_train, classifier=lr) 
plt.xlabel('PC 1') 
plt.ylabel('PC 2') 
plt.legend(loc='lower left') 
plt.show() 
print("LR classifier's train accuracy on PCA:")
print(accuracy_score(y_train,y_train_pred))

#plot the decision regions of the logistic regression on the transformed test dataset 
plot_decision_regions(X_test_pca, y_test, classifier=lr) 
plt.xlabel('PC1') 
plt.ylabel('PC2') 
plt.legend(loc='lower left') 
plt.show()
print("LR classifier's test accuracy on PCA:")
print(accuracy_score(y_test,y_test_pred))

###PCA & SVM
print('\n' * 3)
print("<1.2> refit SVM classifier on the PCA transformed datasets.")
from sklearn.decomposition import PCA 
from sklearn.svm import SVC 
svm = SVC(kernel='linear', C=1.0, random_state=1) 
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std) 
svm=svm.fit(X_train_pca, y_train) 
y_train_pred = svm.predict(X_train_pca) 
y_test_pred = svm.predict(X_test_pca)
plot_decision_regions(X_train_pca, y_train, classifier=svm) 
plt.xlabel('PC 1') 
plt.ylabel('PC 2') 
plt.legend(loc='lower left') 
plt.show() 
print("SVM classifier's train accuracy on PCA:")
print(accuracy_score(y_train,y_train_pred))


#plot the decision regions of the logistic regression on the transformed test dataset 
plot_decision_regions(X_test_pca, y_test, classifier=svm) 
plt.xlabel('PC1') 
plt.ylabel('PC2') 
plt.legend(loc='lower left') 
plt.show()
print("SVM classifier's test accuracy on PCA:")
print(accuracy_score(y_test,y_test_pred))




###LDA & LR
print('\n' * 3)
print("<2.1> refit logistic regression classifier on the LDA transformed datasets.") 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
lda = LDA(n_components=2) 
lr = LogisticRegression() 
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)
lr = lr.fit(X_train_lda, y_train) 
y_train_pred = lr.predict(X_train_lda) 
y_test_pred = lr.predict(X_test_lda)
plot_decision_regions(X_train_lda, y_train, classifier=lr) 
plt.xlabel('LD 1') 
plt.ylabel('LD 2') 
plt.legend(loc='lower left') 
plt.show()
print("LR classifier's train accuracy on LDA:")
print(accuracy_score(y_train,y_train_pred))
 
plot_decision_regions(X_test_lda, y_test, classifier=lr) 
plt.xlabel('LD 1') 
plt.ylabel('LD 2') 
plt.legend(loc='lower left') 
plt.show()
print("LR classifier's test accuracy on LDA:")
print(accuracy_score(y_test,y_test_pred))




###LDA & SVM
print('\n' * 3)
print("<2.2> refit SVM classifier on the LDA transformed datasets.") 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA  
from sklearn.svm import SVC 
lda = LDA(n_components=2) 
svm = SVC(kernel='linear', C=1.0, random_state=1) 
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)

svm=svm.fit(X_train_lda, y_train) 
y_train_pred = svm.predict(X_train_lda) 
y_test_pred = svm.predict(X_test_lda)
plot_decision_regions(X_train_lda, y_train, classifier=svm) 
plt.xlabel('LD 1') 
plt.ylabel('LD 2') 
plt.legend(loc='lower left') 
plt.show() 
print("SVM classifier's train accuracy on LDA:")
print(accuracy_score(y_train,y_train_pred))

plot_decision_regions(X_test_pca, y_test, classifier=svm) 
plt.xlabel('LD 1') 
plt.ylabel('LD 2') 
plt.legend(loc='lower left') 
plt.show()
print("SVM classifier's test accuracy on LDA:")
print(accuracy_score(y_test,y_test_pred))



###kPCA & LR
print('\n' * 3)
print("<3.1> refit logistic regression classifier on the kPCA transformed datasets.")  
from sklearn.decomposition import KernelPCA 
lr = LogisticRegression() 
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15) 


parameters=[0.005,0.010,0.015,0.020,0.025]
for i in range(5):
    kpca.gamma=parameters[i]
    X_train_kpca = kpca.fit_transform(X_train_std)
    X_test_kpca = kpca.transform(X_test_std)
    lr.fit(X_train_kpca, y_train) 
    y_train_pred = lr.predict(X_train_kpca) 
    y_test_pred = lr.predict(X_test_kpca)
print("LR classifier's train accuracy on kPCA:")
print(accuracy_score(y_train,y_train_pred))
print("LR classifier's test accuracy on kPCA:")
print(accuracy_score(y_test,y_test_pred))

###kPCA & SVM
print('\n' * 3)
print("<3.2> refit SVM regression classifier on the kPCA transformed datasets.")  
from sklearn.svm import SVC 
svm = SVC(kernel='linear', C=1.0, random_state=1) 
for i in range(5):
    kpca.gamma=parameters[i]
    svm.fit(X_train_kpca, y_train) 
    y_train_pred = lr.predict(X_train_kpca) 
    y_test_pred = lr.predict(X_test_kpca)
print("SVM classifier's train accuracy on kPCA:")
print(accuracy_score(y_train,y_train_pred))
print("SVM classifier's test accuracy on kPCA:")
print(accuracy_score(y_test,y_test_pred))

print('\n' * 3)
print("My name is Hongyu Zhang")
print("My NetID is: hongyuz2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")






































