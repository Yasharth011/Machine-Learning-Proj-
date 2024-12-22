import kagglehub
import pandas as pd

path = kagglehub.dataset_download("andonians/random-linear-regression")

print("Path of Data Set : ", path)


def gradient_descent(alpha, data_set, w, b):

    m = data_set[data_set.columns[0]].count()
    for i in range(m):
        x = data_set.iloc[i, 0]
        y = data_set.iloc[i, 1]
        if (pd.isna(x) or pd.isna(y)):
            continue
        f = w*x + b
        w = w - alpha*(1/m)*(f - y)*x
        b = b - alpha*(1/m)*(f - y)
    return (w, b)


def predict_value(x, w, b):
    return w*x + b


def main():
    data = pd.read_csv(path + "/train.csv")

    w = float(input("Enter value for w : "))
    b = float(input("Enter value for b : "))
    # learning rate = 0.1 best for this data set
    w, b = gradient_descent(0.1, data, w, b)
    test_data = pd.read_csv(path+"/test.csv")
    rows = test_data[test_data.columns[0]].count()
    for i in range(rows):
        x = test_data.iloc[i, 0]
        y = predict_value(x, w, b)
        print("Predicted value : ", y)


if __name__ == "__main__":
    main()
