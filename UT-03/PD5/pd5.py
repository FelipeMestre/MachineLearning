import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

input_file = "sample.csv"
df = pd.read_csv(input_file, header=0)
print(df.values)

colors = ("orange", "blue")
plt.scatter(df['x'], df['y'], s=300, c=df['label'], cmap=matplotlib.colors.ListedColormap(colors))
plt.show()

X = df[['x', 'y']].values
y = df['label'].values

print("==================================== Valores del dataset ====================================")
print(X)
print("==================================== Etiquetas del dataset ====================================")
print(y)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=0, shuffle=True)
train_XLR = train_X.copy()
train_YLR = train_y.copy()
test_XLR = test_X.copy()
test_YLR = test_y.copy()

print("==================================== LDA ====================================")
lda = LinearDiscriminantAnalysis()
lda = lda.fit(train_X, train_y)

y_pred = lda.predict(test_X)
print("Predicted vs Expected")
print(y_pred)
print(test_y)

print(classification_report(test_y, y_pred, digits=3))
print(confusion_matrix(test_y, y_pred))

print("==================================== Regresion Logistica ====================================")
lr = LogisticRegression()
lr = lr.fit(train_XLR, train_YLR)

y_predLR = lr.predict(test_XLR)
print("Predicted vs Expected")
print(y_predLR)
print(test_YLR)

print(classification_report(test_YLR, y_predLR, digits=3))
print(confusion_matrix(test_YLR, y_predLR))

