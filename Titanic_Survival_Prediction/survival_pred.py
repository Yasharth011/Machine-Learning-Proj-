import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# path = kagglehub.dataset_download("yasserh/titanic-dataset")

def main():
    
    data = pd.read_csv("Titanic-Dataset.csv") 

    # print(data.describe())




if __name__ == "__main__":
    main()

# Notes on data set :- 
# Pclass - ticket class (1/2/3)
# SibSP - number of sibllings or spouses
# Parch - number of parents or children 
