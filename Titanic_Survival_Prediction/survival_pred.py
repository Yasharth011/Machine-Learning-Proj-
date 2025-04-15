import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path = kagglehub.dataset_download("yasserh/titanic-dataset")

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

    print(data)


if __name__ == "__main__":
    main()

"""
Notes on data set :- 
Pclass - ticket class (1/2/3)
SibSP - number of sibllings or spouses
Parch - number of parents or children
---
If a features does not have enough values drop it
---
While extract Titles using regular exp 
r'' is used because python treats \. as an escape sequence
"""
