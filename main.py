import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import os

cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in %r: %s" % (cwd, files))

#Carregar dados
src_dataset = 'dataset.csv'
data = pd.read_csv(open(src_dataset), skiprows=0, delimiter=',')
data.info()


#removendo espaços em branco
sx1 = [sx.strip() for sx in list(data.Symptoms.unique()) if sx[0] == " "]
sx2 = [sx for sx in list(data.Symptoms.unique()) if sx[0] != " "]
sx = sx1 + sx2
sx = [s.lower() for s in sx]
print(sx, len(sx))


# Dividir variáveis
array = data.values
X = array[:, 0:8].astype(float)
Y = array[:, 8]

# Divisão treino/teste
test_size = 0.20
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

models = [
    ("LR", LogisticRegression(solver='newton-cg')),
    ("KNN", KNeighborsClassifier()),
    ("CART", DecisionTreeClassifier()),
    ("NB", GaussianNB()),
    ("SVM", SVC())
]

np.random.seed(7)
num_folds = 10
scoring = 'accuracy'
results = []
names = []
for name, model in models:
    kFold = KFold(n_splits=num_folds)
    cv_results = cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = '%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())
    print(msg)

