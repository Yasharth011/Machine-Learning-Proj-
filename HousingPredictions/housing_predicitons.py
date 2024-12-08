import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# store file path
file_path = '/home/yasharth/workspace/ML/HousingPredictions/train.csv'

# read csv data
house_data = pd.read_csv(file_path)

# save column corresponding to sales price
y = house_data.SalePrice

# features list
features = ['LotArea',
            'YearBuilt',
            '1stFlrSF',
            '2ndFlrSF',
            'FullBath',
            'BedroomAbvGr',
            'TotRmsAbvGrd']

# to store data corresponding to features list
X = house_data[features]

# split data into training and validation data
train_X, val_x, train_Y, val_y = train_test_split(X, y, random_state=0)


model = DecisionTreeRegressor(random_state=1)

model.fit(train_X, train_Y)

predictions = model.predict(val_x)

error = mean_absolute_error(val_y, predictions)

print(predictions - error)

export_graphviz(model, out_file='tree.dot', feature_names=features)
