import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


def kfold_mean_error(model, input, output, k):
    kf = KFold(n_splits=k)
    error = []
    for train_index, test_index in kf.split(input):
        X_train, X_test, y_train, y_test = input[train_index,:], input[test_index], \
            output[train_index], output[test_index]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        error.append(mean_squared_error(y_test, y_pred))

    return np.sqrt(np.mean(error))


data = pd.read_csv("train_regression.csv")
data.drop(columns=['Alley', 'Id', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'LotFrontage'], inplace=True)
data.fillna(method='ffill', inplace=True)
data.dropna(inplace=True)
column_names = data.columns.to_list()
result = data.applymap(np.isreal)
non_numeric_columns = np.array([])
for i in range(len(column_names)):
    array = np.array(result[column_names[i]])
    if array[0] == False:
        non_numeric_columns = np.append(non_numeric_columns, column_names[i])
for i in range(len(non_numeric_columns)):
    data[[non_numeric_columns[i]]] = data[[non_numeric_columns[i]]].apply(LabelEncoder().fit_transform)

input_x = data.iloc[:, :-1].values
target = data.iloc[:, -1].values

input_train, input_test, target_train, target_test = train_test_split(input_x, target, test_size=0.20, random_state=42)
lin_reg = LinearRegression()

lin_reg.fit(input_train, target_train)
train_predicted_price_lin = lin_reg.predict(input_train)
train_rmse_lin = np.sqrt(mean_squared_error(target_train, train_predicted_price_lin))

test_predicted_price_lin = lin_reg.predict(input_test)
test_rmse_lin = np.sqrt(mean_squared_error(target_test, test_predicted_price_lin))

lasso_reg = Lasso(alpha=1)
lasso_reg.fit(input_train, target_train)
train_predicted_price_lasso = lasso_reg.predict(input_train)
train_rmse_lasso = np.sqrt(mean_squared_error(target_train, train_predicted_price_lasso))

test_predicted_price_lasso = lasso_reg.predict(input_test)
test_rmse_lasso = np.sqrt(mean_squared_error(target_test, test_predicted_price_lasso))

ridge_reg = Ridge(alpha=1)
ridge_reg.fit(input_train, target_train)
train_predicted_price_ridge = ridge_reg.predict(input_train)
train_rmse_ridge = np.sqrt(mean_squared_error(target_train, train_predicted_price_ridge))

test_predicted_price_ridge = ridge_reg.predict(input_test)
test_rmse_ridge = np.sqrt(mean_squared_error(target_test, test_predicted_price_ridge))


labels = ["Linear", "Ridge", "Lasso"]
errors = [test_rmse_lin, test_rmse_ridge, test_rmse_lasso]
plt.bar(labels, errors)
plt.title("Area RMSE Comparison")
plt.ylabel("RMSE")
plt.show()


train_data = pd.read_csv("train_regression.csv")
price = train_data["SalePrice"]
area = train_data["LotArea"]
rooms = train_data["TotRmsAbvGrd"]

price.fillna(method='ffill', inplace=True)
area.fillna(method='ffill', inplace=True)
rooms.fillna(method ='ffill', inplace = True)

price.dropna(inplace=True)
area.dropna(inplace=True)
rooms.dropna(inplace = True)

plt.scatter(area, price)
plt.title("Price vs Area")
plt.ylabel("Price (Dollars)")
plt.xlabel("Area (Square Feet)")
plt.show()

plt.scatter(rooms, price)
plt.title("Price vs Rooms")
plt.ylabel("Price (Dollars)")
plt.xlabel("Rooms NO.")
plt.show()

area_train, area_test, price_train, price_test = train_test_split(area, price, test_size=0.30, random_state=42)



price_train = np.array(price_train).reshape(-1, 1)
area_train = np.array(area_train).reshape(-1, 1)

area_reg = LinearRegression()

area_reg.fit(area_train, price_train)

predicted_price_area = area_reg.predict(area_train)

plt.scatter(area_train, price_train, color='b', label="Real Data")

plt.plot(area_train, predicted_price_area, color='k', label="Linear Model")

plt.title("Price vs Area")
plt.ylabel("Price (Dollars)")
plt.xlabel("Area (Square Feet)")
plt.legend()
plt.show()

price_test = np.array(price_test).reshape(-1, 1)
area_test = np.array(area_test).reshape(-1, 1)
area = np.array(area).reshape(-1, 1)

price = np.array(price).reshape(-1, 1)

area_test_lin_price_pred = area_reg.predict(area_test)
area_rmse_lin = np.sqrt(mean_squared_error(price_test, area_test_lin_price_pred))



max_k = 10
lin_mean_error = np.array([])
for k in range(2, max_k + 1):
    err = kfold_mean_error(lin_reg, input_x, target, k)
    lin_mean_error = np.append(lin_mean_error, err)
plt.plot(range(2, max_k + 1), lin_mean_error)
plt.title("Compare different K in K-Fold For Linear Regression")
plt.xlabel("K")
plt.ylabel("RMSE")
plt.show()

lasso_mean_error = np.array([])
for k in range(2, max_k + 1):
    err = kfold_mean_error(lasso_reg, input_x, target, k)
    lasso_mean_error = np.append(lasso_mean_error, err)
plt.plot(range(2, max_k + 1), lasso_mean_error)
plt.title("Compare different K in K-Fold For Lasso Regression")
plt.xlabel("K")
plt.ylabel("RMSE")
plt.show()

ridge_mean_error = np.array([])
for k in range(2, max_k + 1):
    err = kfold_mean_error(ridge_reg, input_x, target, k)
    ridge_mean_error = np.append(ridge_mean_error, err)
plt.plot(range(2, max_k + 1), ridge_mean_error)
plt.title("Compare different K in K-Fold For Ridge Regression")
plt.xlabel("K")
plt.ylabel("RMSE")
plt.show()
