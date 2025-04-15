import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.ion()
path = kagglehub.dataset_download("yasserh/titanic-dataset")

def compute_f(x, wt, bias):

    z = np.dot(x,wt) + bias
    sigmoid = 1/(1+np.exp(-z))
    return sigmoid 

def compute_loss(f, y):
    y1 = -y*np.log(f)
    y2 = (1-y)*np.log(1-f)

    return (y1-y2)

def gradient_descent(x, y, alpha):

    m = len(x) # no of training values 
    w = np.zeros(x.shape[1]) # init weight param
    b = 0 # init bias param
    loss = list()

    for i in range(m):

        x_values = np.array(x.iloc[i])
        y_value = y.iloc[i]
        f = compute_f(x_values, w, b)

        loss.append(compute_loss(f, y_value))

        d_w = (alpha*np.dot((f-y_value), x_values))/m
        d_b = (alpha*(f-y_value))/m

        w = w - d_w
        b = b - d_b
    iterations = [i for i in range(m)]
    plt.plot(iterations, loss, color='r')
    return (w,b)

def logistic_regression(data, x, y):
    
    alpha = 0.00001
    w, b = gradient_descent(x, y, alpha)
    m = len(x)
    for i in range(m):
        test_value = x.iloc[i]
        if(compute_f(test_value, w, b, 1)>=0.5):
            print("SURVIVED")
        else:
            print("NOT SURVIVED")

def main():

    data = pd.read_csv("Titanic-Dataset.csv")

    # print(data.describe(include = 'all'))
    # print(data)
    # print(data['Age'].count())

    # drop cabin feature (less data)
    data = data.drop("Cabin", axis=1)

    # drop ticket feature (no usefull data)
    data = data.drop("Ticket", axis=1)

    # drop age feature 
    data = data.drop("Age", axis=1)

    # map sex feature
    map_sex = {"male": 1, "female": 0}
    data["Sex"] = data["Sex"].map(map_sex)

    # fill missing Embarked value with S (majority)
    data = data.fillna({"Embarked": "S"})

    # map embarked feature
    map_embarked = {"S": 1, "C": 2, "Q": 3}
    data["Embarked"] = data["Embarked"].map(map_embarked)

    # extracting title for each Name
    dataset = pd.Series(data["Name"])
    data["Title"] = dataset.str.extract(r" ([A-Za-z]+)\.")

    # print(pd.crosstab(data["Title"], data["Sex"]))

    # replacing the Titles with common Titles like (Rare, Royal, Miss, Mrs)
    data["Title"] = data["Title"].replace(["Capt", "Col", "Don", "Don", "Dr", "Jonkheer", "Major", "Rev"], "Rare")
    data["Title"] = data["Title"].replace(["Countess", "Lady", "Sir"], "Royal")
    data["Title"] = data["Title"].replace(["Mlle", "Ms"], "Miss")
    data["Title"] = data["Title"].replace("Mme", "Mrs")

    # mapping each Title with a number 
    title_mapping = {"Mr":1, "Mrs":2, "Master":3, "Miss":4, "Royal":5, "Rare":6}
    data["Title"] = data["Title"].map(title_mapping)
    data["Title"] = data["Title"].fillna(0)

    # drop Name feature 
    data = data.drop("Name", axis=1)

    # map fare feature to logical groups 
    data["FareBand"] = pd.qcut(data["Fare"], 4, labels=[1,2,3,4])

    # drop fare feature 
    data = data.drop("Fare", axis=1)

    # spliting feature as X & Y
    x = data.drop(["Survived", "PassengerId"], axis=1)
    y = data["Survived"]

    logistic_regression(data, x, y)


if __name__ == "__main__":
    main()

"""
Notes on data set :- 
Pclass - ticket class (1/2/3)
SibSP - number of sibllings or spouses
Parch - number of parents or children
Embarked - S(Southampton), Q(Queenstown), C(Cherbourg)
---
If a features does not have enough values drop it
---
While extract Titles using regular exp 
r'' is used because python treats \. as an escape sequence
"""
