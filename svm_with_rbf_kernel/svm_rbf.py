import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def RBF(X, gamma):
    if gamma == None:
        gamma = 1.0/X.shape[1]

    K = np.exp(-gamma * np.sum((X-X[:, np.newaxis])**2, axis = -1))
    return K

X, y = make_circles(n_samples = 500, noise = 0.06, random_state = 42)
df = pd.DataFrame(dict(x1 = X[:,0], x2 = X[:,1], y = y))
colors = {0:'blue', 1:'yellow'}
fig, ax = plt.subplots()
grouped = df.groupby('y')
for key, group in grouped:
    group.plot(ax = ax, kind = 'scatter', x = 'x1', y = 'x2', label = key, color = colors[key])
plt.show() 

# linear SVM 
clf = SVC(kernel = 'linear')
clf.fit(X, y)
pred = clf.predict(X)
print("Accuracy with linear kernel: ", accuracy_score(pred, y))

# polynomial SVM
clf = SVC(kernel="poly")
clf.fit(X, y)
pred = clf.predict(X)
print("Accuracy with poly kernel: ", accuracy_score(pred, y))

#RBF SVM
X_orig = X
gammas = [None, 1, 0.9,0.8,0.7,0.6,0.5, 0.4, 0.3, 0.2, 0.1, 0.001, 0.0001, 0]
for gamma in gammas:
    X = RBF(X_orig, gamma)
    clf = SVC(kernel = 'linear')
    clf.fit(X,y)
    pred = clf.predict(X)
    print("Accuracy with RBF kernel and gamma =",gamma,": ",accuracy_score(pred, y))

