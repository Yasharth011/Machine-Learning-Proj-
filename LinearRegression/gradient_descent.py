import kagglehub
import pandas as pd
import argparse 

# path = kagglehub.dataset_download("andonians/random-linear-regression")

# adding  cmd line arguments 
parser = argparse.ArgumentParser()
parser.add_argument("-w", help="weight for gradient descen", type=float)
parser.add_argument("-b", help="bias for gradient descen", type=float)
parser.add_argument("-alpha", help="learning rate of gradient descen", type=float)
args = parser.parse_args()


def gradient_descent(alpha, data_set, w, b):

    m = data_set[data_set.columns[0]].count()

    for i in range(m):

        x = data_set.iloc[i, 0]
        y = data_set.iloc[i, 1]
        
        # if value is nan ignore 
        if (pd.isna(x) or pd.isna(y)):
            continue

        f = w*x + b 
        w = w - alpha*(1/m)*(f - y)*x
        b = b - alpha*(1/m)*(f - y)
    return (w, b)


def predict_value(x, w, b):
    return (w*x + b)


def main():
    data = pd.read_csv("train.csv")

    # parameters for gardient descent
    w = args.w
    b = args.b 
    alpha = args.alpha

    # performing gradient descent 
    w, b = gradient_descent(0.1, data, w, b)

    # using testing data 
    # test_data = pd.read_csv(path+"/test.csv")
    # rows = test_data[test_data.columns[0]].count()
    # print("test_x | test_y \t train_x | train_y")
    # for i in range(rows):
    #     x = test_data.iloc[i, 0]
    #     y = predict_value(x, w, b)
    #     print(x, ":", y, "|", data.iloc[i, 0], ":", data.iloc[i, 1])

    # using user input 
    input_value = float(input("Enter a number : "))
    
    y = predict_value(input_value, w, b)
    # index = data.index(input_value)
    print(input_value, ":", y)#, "|", data[input_value])


if __name__ == "__main__":
    main()
