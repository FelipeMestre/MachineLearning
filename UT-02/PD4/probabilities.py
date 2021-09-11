import pandas as pd 
from IPython.display import display


import warnings
warnings.filterwarnings("ignore")

dataset = pd.read_csv("titanic.csv")
types_train = dataset.dtypes
num_values = types_train[(types_train == float)]

dataset.drop(labels = ["Name","Siblings/Spouses Aboard","Parents/Children Aboard","Fare"], axis = 1, inplace = True)

dataset["Age"].fillna(dataset["Age"].median(), inplace = True)
display(dataset.head())

print("============ PROBABILITY OF SURVIVING FOR WOMEN ============")
for i in range(1,4):
    survived_woman = dataset[dataset["Sex"] == "female"][dataset["Pclass"] == i][dataset["Survived"] == 1]["Age"].count()
    C1_woman = dataset[dataset["Sex"] == "female"][dataset["Pclass"] == i]["Age"].count()
    print("Probability women was in class " + str(i)  +" and survived")
    print(survived_woman/C1_woman)

print("============ PROBABILITY OF SURVIVING FOR MEN ============")
for i in range(1,4):
    survived_men = dataset[dataset["Sex"] == "male"][dataset["Pclass"] == i][dataset["Survived"] == 1]["Age"].count()
    C1_men = dataset[dataset["Sex"] == "male"][dataset["Pclass"] == i]["Age"].count()
    print("Probability men was in class " + str(i)  +" and survived")
    print(survived_men/C1_men)

print("============ PROBABILITY OF SURVIVING FOR A CLASS 3 CHILD ============")
survived_childs = dataset[dataset["Age"] <= 10][dataset["Pclass"] == 3][dataset["Survived"] == 1]["Age"].count()
C3_childs = dataset[dataset["Age"] <= 10][dataset["Pclass"] == 3]["Age"].count()
print("Probability child was in class 3 and survived")
print(survived_childs/C3_childs)