import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

training = pd.read_csv("sports_Training.csv", header=0)
scoring = pd.read_csv("sports_scoring.csv", header=0)

training = training[training['CapacidadDecision'] >= 3][training['CapacidadDecision'] <= 100]
scoring = scoring[scoring['CapacidadDecision'] >= 3][scoring['CapacidadDecision'] <= 100]

labels = ['Edad','Fuerza','Velocidad','Lesiones','Vision','Resistencia','Agilidad','CapacidadDecision']
for label in labels:
    le = LabelEncoder()
    le.fit(training[label])
    encodedcolumn = le.transform(training[label])
    training[label] = encodedcolumn

#normalize data
normalized = normalize(training[labels],norm='l2',axis=0)
training[labels]= normalized

normalized_S = normalize(scoring[labels],norm='l2',axis=0)
scoring[labels]= normalized_S

X = training[labels].values
y = training['DeportePrimario'].values


print("==================================== LDA ====================================")
lda = LinearDiscriminantAnalysis()
lda = lda.fit(X, y)

y_pred = lda.predict(scoring)
print("Predictions")
print(y_pred)

import csv 
data = [y_pred] 
file = open('resultadoSCIKITNormalizadoX2.csv', 'w+', newline ='') 
with file:     
    write = csv.writer(file) 
    write.writerows(data) 
file.close()