import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
sns.set_style("whitegrid")
from matplotlib import pyplot as plt
from IPython.display import display
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression

training = pd.read_csv("train.csv")
testing = pd.read_csv("test.csv")


print("=================== FIRST ROWS TRAINING ===================")
display(training.head())
print("=================== FIRST ROWS TEST ===================")
display(testing.head())

print("=================== TRAINIGN KEYS ===================")
display(training.keys())
print("=================== TEST KEYS ===================")
display(testing.keys())

types_train = training.dtypes
num_values = types_train[(types_train == float)]

print("=================== These are the numerical features: ===================")
print(num_values)

training.describe()

def null_table(training, testing):
    print("=================== Training Nulls ===================")
    print(pd.isnull(training).sum()) 
    print(" ")
    print("=================== Testing Nulls ===================")
    print(pd.isnull(testing).sum())

null_table(training, testing)

print("=================== Distribution of age ===================")
copy = training.copy()
copy.dropna(inplace = True)
plot1 = plt.figure(1)
sns.distplot(copy["Age"])


# #Substitute Null values
training.drop(labels = ["Cabin", "Ticket"], axis = 1, inplace = True)
testing.drop(labels = ["Cabin", "Ticket"], axis = 1, inplace = True)

training["Age"].fillna(training["Age"].median(), inplace = True)
testing["Age"].fillna(testing["Age"].median(), inplace = True) 
training["Embarked"].fillna("S", inplace = True)
testing["Fare"].fillna(testing["Fare"].median(), inplace = True)

null_table(training, testing)

training.head()
testing.head()

print("=================== There are no nulls !! ===================")
plot2 = plt.figure(2)
sns.barplot(x="Sex", y="Survived", data=training)
plt.title("Distribution of Survival based on Gender")

plot3 = plt.figure(3)
sns.stripplot(x="Survived", y="Age", data=training, jitter=True)
sns.barplot(x="Pclass", y="Survived", data=training)
plt.ylabel("Survival Rate")
plt.title("Distribution of Survival Based on Class")


total_survived_females = training[training.Sex == "female"]["Survived"].sum()
total_survived_males = training[training.Sex == "male"]["Survived"].sum()

print("=================== PROPORTION OF SURVIVORS BASED ON GENDER ===================")
print("Total people survived is: " + str((total_survived_females + total_survived_males)))
print("Proportion of Females who survived:") 
print(total_survived_females/(total_survived_females + total_survived_males))
print("Proportion of Males who survived:")
print(total_survived_males/(total_survived_females + total_survived_males))

plot4 = plt.figure(4)
sns.barplot(x="Pclass", y="Survived", data=training)
plt.ylabel("Survival Rate")
plt.title("Distribution of Survival Based on Class")

total_survived_one = training[training.Pclass == 1]["Survived"].sum()
total_survived_two = training[training.Pclass == 2]["Survived"].sum()
total_survived_three = training[training.Pclass == 3]["Survived"].sum()
total_survived_class = total_survived_one + total_survived_two + total_survived_three

print("=================== PROPORTION OF SURVIVOR BASED ON PASSANGER CLASS ===================")
print("Total people survived is: " + str(total_survived_class))
print("Proportion of Class 1 Passengers who survived:") 
print(total_survived_one/total_survived_class)
print("Proportion of Class 2 Passengers who survived:")
print(total_survived_two/total_survived_class)
print("Proportion of Class 3 Passengers who survived:")
print(total_survived_three/total_survived_class)

print("=================== PROPORTION OF SURVIVOR BASED ON GENDER AND CLASS ===================")
plot5 = plt.figure(5)
sns.barplot(x="Pclass", y="Survived", hue="Sex", data=training)
plt.ylabel("Survival Rate")
plt.title("Survival Rates Based on Gender and Class")

plot6 = plt.figure(6)
sns.barplot(x="Sex", y="Survived", hue="Pclass", data=training)
plt.ylabel("Survival Rate")
plt.title("Survival Rates Based on Gender and Class")

print("=================== PROPORTION OF SURVIVOR BASED ON AGE ===================")
survived_ages = training[training.Survived == 1]["Age"]
not_survived_ages = training[training.Survived == 0]["Age"]
plot7 = plt.figure(7)
plt.subplot(1, 2, 1)
sns.distplot(survived_ages, kde=False)
plt.axis([0, 100, 0, 100])
plt.title("Survived")
plt.ylabel("Proportion")
plt.subplot(1, 2, 2)
sns.distplot(not_survived_ages, kde=False)
plt.axis([0, 100, 0, 100])
plt.title("Didn't Survive")

print("=================== ALL FEATURES GRAPHICS ===================")
plot8 = plt.figure(8)
sns.stripplot(x="Survived", y="Age", data=training, jitter=True)

sns.pairplot(training)
#plt.show()


print ("=================== FEATURE ENGINEERING ===================")
# #Lets make all atributes numerical

le_sex = LabelEncoder()
le_sex.fit(training["Sex"])

encoded_sex_training = le_sex.transform(training["Sex"])
training["Sex"] = encoded_sex_training
encoded_sex_testing = le_sex.transform(testing["Sex"])
testing["Sex"] = encoded_sex_testing

le_embarked = LabelEncoder()
le_embarked.fit(training["Embarked"])

encoded_embarked_training = le_embarked.transform(training["Embarked"])
training["Embarked"] = encoded_embarked_training
encoded_embarked_testing = le_embarked.transform(testing["Embarked"])
testing["Embarked"] = encoded_embarked_testing

training["FamSize"] = training["SibSp"] + training["Parch"] + 1
testing["FamSize"] = testing["SibSp"] + testing["Parch"] + 1
training["IsAlone"] = training.FamSize.apply(lambda x: 1 if x == 1 else 0)
testing["IsAlone"] = testing.FamSize.apply(lambda x: 1 if x == 1 else 0)

for name in training["Name"]:
    training["Title"] = training["Name"].str.extract("([A-Za-z]+)\.",expand=True)
    
for name in testing["Name"]:
    testing["Title"] = testing["Name"].str.extract("([A-Za-z]+)\.",expand=True)

display(training.head())
#CHECKEAR ESTO
titles = set(training["Title"]) #making it a set gets rid of all duplicates
print(titles)

title_list = list(training["Title"])
frequency_titles = []

for i in titles:
    frequency_titles.append(title_list.count(i))
    
print(frequency_titles)

titles = list(titles)

title_dataframe = pd.DataFrame({
    "Titles" : titles,
    "Frequency" : frequency_titles
})

print(title_dataframe)

title_replacements = {"Mlle": "Other", "Major": "Other", "Col": "Other", "Sir": "Other", "Don": "Other", "Mme": "Other",
          "Jonkheer": "Other", "Lady": "Other", "Capt": "Other", "Countess": "Other", "Ms": "Other", "Dona": "Other"}

training.replace({"Title": title_replacements}, inplace=True)
testing.replace({"Title": title_replacements}, inplace=True)

le_title = LabelEncoder()
le_title.fit(training["Title"])

encoded_title_training = le_title.transform(training["Title"])
training["Title"] = encoded_title_training
encoded_title_testing = le_title.transform(testing["Title"])
testing["Title"] = encoded_title_testing

training.drop("Name", axis = 1, inplace = True)
testing.drop("Name", axis = 1, inplace = True)

display(training.sample(5))
print("======================= ALL FEATURES ARE NUMERICAL FAMILY ONES ARE COMBINED AND NAME WAS USED TO MAKE THE TITLE COLUMN =======================")
print("======================= LET'S MAKE LOGISTIC REGRESSION =======================")

from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split #to create validation data set

X_train = training.drop(labels=["PassengerId", "Survived"], axis=1) #define training features set
y_train = training["Survived"] #define training label set
X_test = testing.drop("PassengerId", axis=1) #define testing features set
#we don't have y_test, that is what we're trying to predict with our model
display(X_train.head())


X_training, X_valid, y_training, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0) #X_valid and y_valid are the validation sets

dt_clf = DecisionTreeClassifier()

parameters_dt = {"criterion": ["gini", "entropy"], "splitter": ["best", "random"], "max_features": ["auto", "sqrt", "log2"]}

grid_dt = GridSearchCV(dt_clf, parameters_dt, scoring=make_scorer(accuracy_score))
grid_dt.fit(X_training, y_training)

dt_clf = grid_dt.best_estimator_

dt_clf.fit(X_training, y_training)
pred_dt = dt_clf.predict(X_valid)
acc_dt = accuracy_score(y_valid, pred_dt)

print("The Score for Decision Tree is: " + str(acc_dt))
