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

    # map sex feature
    map_sex = {"male": 1, "female": 0}
    data["Sex"].map(map_sex)

    # map embarked feature
    map_embarked = {"S": 1, "C": 2, "Q": 3}
    data["Embarked"] = data["Embarked"].map(map_embarked)

    # extracting title for each Name
    dataset = pd.Series(data["Name"])
    data["Title"] = dataset.str.extract(r" ([A-Za-z]+)\.")

    print(pd.crosstab(data["Title"], data["Sex"]))


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
