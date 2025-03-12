import argparse

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# path = kagglehub.dataset_download("andonians/random-linear-regression")

# adding  cmd line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-type", help="type of gradient algorithm to use")
parser.add_argument("-w", help="weight for gradient descen", type=float)
parser.add_argument("-b", help="bias for gradient descen", type=float)
parser.add_argument("-alpha", help="learning rate of gradient descen", type=float)
parser.add_argument("-i", help="iterations for number of gradients", type=int)
args = parser.parse_args()

plt.ion()


def initialize():
    global cost
    cost = list()
    global iterations
    iterations = list()


def stochastic_gradient_descent(alpha, data_set, w, b):

    m = data_set[data_set.columns[0]].count()

    for i in range(m):

        random_pt = np.random.randint(0, m)
        x = data_set.iloc[random_pt, 0]
        y = data_set.iloc[random_pt, 1]

        # if value is nan ignore
        if pd.isna(x) or pd.isna(y):
            continue

        f = w * x + b
        w = w - alpha * (1 / m) * (f - y) * x
        b = b - alpha * (1 / m) * (f - y)
        j = ((f - y) ** 2) / (2 * m)

        plot_graph(i, j)

    return (w, b)


def batch_gradient_descent(iterations, alpha, data_set, w, b):

    m = data_set[data_set.columns[0]].count()

    for i in range(iterations):

        total_w = 0
        total_b = 0
        cost = 0
        count = 0

        for j in range(m):

            x = data_set.iloc[j, 0]
            y = data_set.iloc[j, 1]

            # if value is nan ignore
            if pd.isna(x) or pd.isna(y):
                continue

            # compute f
            f = x * w + b
            # compute error
            error = f - y
            # compute paramters
            total_w += error * x
            total_b += error
            # compute cost
            cost += error**2
            # data points excluding nan
            count = count + 1

        w = w - (alpha * total_w) / count
        b = b - (alpha * total_b) / count
        cost = cost / (2 * count)
        plot_graph(i, cost)

    return (w, b)

# plot graph of Cost v/s iterations
def plot_graph(i, j):

    # calculating cost
    cost.append(j)
    iterations.append(i)

    plt.plot(iterations, cost, color="red")

    plt.pause(0.00001)


def predict_value(x, w, b):
    return w * x + b


def main():

    initialize()

    data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")

    # parameters for gardient descent
    w = args.w
    b = args.b
    alpha = args.alpha
    i = args.i


    # choosing algorithm to perform 
    if(args.type=="sto"):
        w, b = stochastic_gradient_descent(alpha, data, w, b)
    elif(args.type=="batch"):
        w, b = batch_gradient_descent(i, alpha, data, w, b)
    else: 
        print("Invalid Option")
        exit()

    # using testing data
    # test_data = pd.read_csv(path+"/test.csv")
    # rows = test_data[test_data.columns[0]].count()
    # print("test_x | test_y \t train_x | train_y")
    # for i in range(rows):
    #     x = test_data.iloc[i, 0]
    #     y = predict_value(x, w, b)
    #     print(x, ":", y, "|", data.iloc[i, 0], ":", data.iloc[i, 1])

    # taking user input of x
    input_value = float(input("Enter a number : "))

    # printing corresponding y-values form training set
    print("row | y_value ")
    print(test_data.loc[test_data["x"] == input_value, "y"])

    # prediciting y-value
    y = predict_value(input_value, w, b)

    print(input_value, ":", y)

    plt.pause(10) # hold graph for 10 secs


if __name__ == "__main__":
    main()
