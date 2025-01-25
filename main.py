import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, KFold, cross_val_score
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

# Carregar dados
src_dataset = 'dataset.csv'
data = pd.read_csv(open(src_dataset), skiprows=0, delimiter=',')
data.info()

# Removendo espaços em branco nos sintomas
sx1 = [sx.strip() for sx in list(data.Symptoms.unique()) if sx[0] == " "]
sx2 = [sx for sx in list(data.Symptoms.unique()) if sx[0] != " "]
sx = sx1 + sx2
sx = [s.lower() for s in sx]
print(sx, len(sx))

# Dividir variáveis
array = data.values
X = array[:, 0:8].astype(float)  # Características (symptoms)
Y = array[:, 8]  # Alvo (disease)

# Divisão treino/teste
test_size = 0.20
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# Definir os modelos
models = [
    ("LR", LogisticRegression(solver='newton-cg')),
    ("KNN", KNeighborsClassifier()),
    ("CART", DecisionTreeClassifier()),
    ("NB", GaussianNB()),
    ("SVM", SVC(probability=True))  # SVM com probabilidade habilitada para AUC
]

# np.random.seed(7)
# num_folds = 10
# scoring = 'accuracy'
# results = []
# names = []

# Inicializar listas para armazenar os resultados
model_names = []
precision_scores = []
recall_scores = []
f1_scores = []
auc_scores = []

# Avaliar os modelos
for name, model in models:

#     kFold = KFold(n_splits=num_folds)
#    cv_results = cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring)
#    results.append(cv_results)
#    names.append(name)
#    msg = '%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())
#    print(msg)
    
    model.fit(X_train, Y_train)  # Treinamento do modelo
    Y_pred = model.predict(X_test)  # Predição
    precision = precision_score(Y_test, Y_pred, average='weighted')
    recall = recall_score(Y_test, Y_pred, average='weighted')
    f1 = f1_score(Y_test, Y_pred, average='weighted')
    
    # AUC
    try:
        auc = roc_auc_score(Y_test, model.predict_proba(X_test), multi_class='ovr')
    except:
        auc = None

    # Armazenar os resultados
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    auc_scores.append(auc)
    
    print(f'{name}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, AUC={auc:.3f}')

    # Exibir matriz de confusão
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

# Visualizando o desempenho dos modelos com gráficos
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(models))

# Plot de desempenho de cada modelo
ax.bar(x - 0.2, precision_scores, 0.2, label='Precision')
ax.bar(x, recall_scores, 0.2, label='Recall')
ax.bar(x + 0.2, f1_scores, 0.2, label='F1')

# Ajustando o gráfico
plt.xticks(x, [model[0] for model in models])
plt.ylabel('Scores')
plt.title('Comparison of Models')
plt.legend()
plt.show()

# Visualizando AUC
if None not in auc_scores:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, auc_scores, label='AUC', color='green')
    plt.xticks(x, [model[0] for model in models])
    plt.ylabel('AUC')
    plt.title('AUC Comparison')
    plt.legend()
    plt.show()

