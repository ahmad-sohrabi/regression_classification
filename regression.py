

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv("train_regression.csv")
data.drop(columns=['Alley', 'Id', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu','LotFrontage'], inplace=True)
data.fillna(method ='ffill', inplace = True)
data.dropna(inplace = True)
column_names = data.columns.to_list()
result = data.applymap(np.isreal)
non_numeric_columns = np.array([])
for i in range(len(column_names)):
   array = np.array(result[column_names[i]])
   if array[0] == False:
      non_numeric_columns = np.append(non_numeric_columns, column_names[i])
for i in range(len( non_numeric_columns )):
   data[[non_numeric_columns[i]]] = data[[non_numeric_columns[i]]].apply(LabelEncoder().fit_transform)

input = data.iloc[:,:-1].values
target = data.iloc[:,-1].values
input_train, input_test, target_train, target_test = train_test_split(input, target,test_size=0.20,random_state=42)
lin_reg = LinearRegression()
lin_reg.fit(input_train, target_train)
train_predicted_price_lin = lin_reg.predict(input_train)
train_rmse_lin = np.sqrt(mean_squared_error(target_train, train_predicted_price_lin))
print(train_rmse_lin)

test_predicted_price_lin = lin_reg.predict(input_test)
test_rmse_lin = np.sqrt(mean_squared_error(target_test, test_predicted_price_lin))
print(test_rmse_lin)

lasso_reg = Lasso(alpha=1000)
lasso_reg.fit(input_train, target_train)
train_predicted_price_lasso = lasso_reg.predict(input_train)
train_rmse_lasso = np.sqrt(mean_squared_error(target_train, train_predicted_price_lasso))
print(train_rmse_lasso)

test_predicted_price_lasso = lasso_reg.predict(input_test)
test_rmse_lasso = np.sqrt(mean_squared_error(target_test, test_predicted_price_lasso))
print(test_rmse_lasso)

ridge_reg = Ridge(alpha=500)
ridge_reg.fit(input_train, target_train)
train_predicted_price_ridge = ridge_reg.predict(input_train)
train_rmse_ridge = np.sqrt(mean_squared_error(target_train, train_predicted_price_ridge))
print(train_rmse_ridge)

test_predicted_price_ridge = ridge_reg.predict(input_test)
test_rmse_ridge = np.sqrt(mean_squared_error(target_test, test_predicted_price_ridge))
print(test_rmse_ridge)

train_data = pd.read_csv("train_regression.csv")
price = train_data["SalePrice"]
area = train_data["LotArea"]
rooms = train_data["TotRmsAbvGrd"]

price.fillna(method ='ffill', inplace = True)
area.fillna(method ='ffill', inplace = True)
rooms.fillna(method ='ffill', inplace = True)

price.dropna(inplace = True)
area.dropna(inplace = True)
rooms.dropna(inplace = True)

area_train, area_test, price_train, price_test = train_test_split(area, price,test_size=0.30,random_state=42)

plt.scatter(area_train, price_train)
plt.title("Area vs Price")
plt.ylabel("Price (Dollars)")
plt.xlabel("Area (Square Feet)")

rooms_train, rooms_test, price_train2, price_test2 = train_test_split(rooms, price,test_size=0.30,random_state=42)

plt.scatter(rooms_train, price_train2)
plt.title("Rooms vs Price")
plt.ylabel("Price (Dollars)")
plt.xlabel("Number of Rooms")

price_train = np.array(price_train).reshape(-1, 1)
price_train2 = np.array(price_train2).reshape(-1, 1)
area_train = np.array(area_train).reshape(-1, 1)
rooms_train = np.array(rooms_train).reshape(-1, 1)

# Separating the data into independent and dependent variables

# Converting each dataframe into a numpy array
# since each dataframe contains only one column

area_reg = LinearRegression()

area_reg.fit(area_train, price_train)

rooms_reg = LinearRegression()

rooms_reg.fit(rooms_train, price_train2)

predicted_price_area = area_reg.predict(area_train)

plt.scatter(area_train, price_train, color ='b', label="Real Data")

plt.plot(area_train, predicted_price_area, color ='k', label="Linear Model")

plt.title("Price vs Area")
plt.ylabel("Price (Dollars)")
plt.xlabel("Area (Square Feet)")
plt.legend()
plt.show()

predicted_prices_rooms = rooms_reg.predict(rooms_train)

plt.scatter(rooms_train, price_train2, color ='b', label="Real Data")

plt.plot(rooms_train, predicted_prices_rooms, color ='k', label="Linear Model")
plt.title("Price vs Rooms")
plt.ylabel("Price (Dollars)")
plt.xlabel("Number of Rooms")
plt.legend()
plt.show()

area_ridge = Ridge(alpha = 3)
area_ridge.fit(area_train, price_train)

rooms_ridge = Ridge(alpha = 1)
rooms_ridge.fit(rooms_train, price_train2)

predicted_area_ridge = area_ridge.predict(area_train)

plt.scatter(area_train, price_train, color ='b', label="Real Data")

plt.plot(area_train, predicted_area_ridge, color ='k', label="Ridge Model")
plt.title("Price vs Area (Ridge Model)")
plt.ylabel("Price (Dollars)")
plt.xlabel("Area (Square Feet)")
plt.legend()
plt.show()

predicted_prices_rooms_ridge = rooms_ridge.predict(rooms_train)

plt.scatter(rooms_train, price_train2, color ='b', label="Real Data")

plt.plot(rooms_train, predicted_prices_rooms_ridge,color ='k', label="Ridge Model")

plt.title("Price vs Rooms(Ridge Model)")
plt.ylabel("Price (Dollars)")
plt.xlabel("Number of Rooms")
plt.legend()
plt.show()

area_lasso = Lasso(alpha = 1)
area_lasso.fit(area_train, price_train)

rooms_lasso = Lasso(alpha = 1)
rooms_lasso.fit(rooms_train, price_train2)

predicted_price_area_lasso = area_lasso.predict(area_train)

plt.scatter(area_train, price_train, color ='b', label="Real Data")

plt.plot(area_train, predicted_price_area_lasso, color ='k', label="Lasso Model")

plt.title("Price vs Area (Lasso Model)")
plt.ylabel("Price (Dollars)")
plt.xlabel("Area (Square Feet)")
plt.legend()
plt.show()

predicted_price_rooms_lasso = rooms_lasso.predict(rooms_train)

plt.scatter(rooms_train, price_train2, color ='b', label="Real Data")

plt.plot(rooms_train, predicted_price_rooms_lasso, color ='k', label="Lasso Model")

plt.title("Rooms vs Price (Lasso Model)")
plt.xlabel("Price (Dollars)")
plt.ylabel("Number of Rooms")
plt.legend()
plt.show()

price_test = np.array(price_test).reshape(-1, 1)
price_test2 = np.array(price_test2).reshape(-1, 1)
area_test = np.array(area_test).reshape(-1, 1)
rooms_test = np.array(rooms_test).reshape(-1, 1)
area = np.array(area).reshape(-1, 1)
rooms = np.array(rooms).reshape(-1, 1)

price = np.array(price).reshape(-1, 1)

area_test_lin_price_pred = area_reg.predict(area_test)
area_rmse_lin = np.sqrt(mean_squared_error(price_test,area_test_lin_price_pred))
print(area_rmse_lin)

rooms_test_lin_price_pred = rooms_reg.predict(rooms_test)
rooms_rmse_lin = np.sqrt(mean_squared_error(price_test2,rooms_test_lin_price_pred))
print(rooms_rmse_lin)


area_test_ridge_price_pred = area_ridge.predict(area_test)
area_rmse_ridge = np.sqrt(mean_squared_error(price_test,area_test_ridge_price_pred))
print(area_rmse_ridge)

rooms_test_ridge_price_pred = rooms_ridge.predict(rooms_test)
rooms_rmse_ridge = np.sqrt(mean_squared_error(price_test2,rooms_test_ridge_price_pred))
print(rooms_rmse_ridge)

area_test_lasso_price_pred = area_lasso.predict(area_test)
area_rmse_lasso = np.sqrt(mean_squared_error(price_test,area_test_lasso_price_pred))
print(area_rmse_lasso)

rooms_test_lasso_price_pred = rooms_lasso.predict(rooms_test)
rooms_rmse_lasso = np.sqrt(mean_squared_error(price_test2,rooms_test_lin_price_pred))
print(rooms_rmse_lasso)

labels = ["Linear", "Ridge", "Lasso"]
errors = [area_rmse_lin, area_rmse_ridge, area_rmse_lasso]
plt.bar(labels, errors)
plt.title("Area RMSE Comparison")
plt.ylabel("RMSE")

def kfold_mean_error(model, input, output, k):
   kf = KFold(n_splits=k)
   error = []
   for train_index, test_index in kf.split(input):
        X_train, X_test, y_train, y_test = input[train_index], input[test_index],\
                                           output[train_index], output[test_index]


        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        error.append(mean_squared_error(y_test, y_pred))

   return np.sqrt(np.mean(error))

max_k = 10
area_lin_mean_error = np.array([])
for k in range(2,max_k+1):
   err =kfold_mean_error(area_reg, area, price, k)
   area_lin_mean_error = np.append(area_lin_mean_error, err)
plt.plot(range(2, max_k+1), area_lin_mean_error)
plt.title("Compare different K in K-Fold")
plt.xlabel("K")
plt.ylabel("RMSE")

area_ridge_mean_error = np.array([])
for k in range(2,max_k+1):
   err =kfold_mean_error(area_ridge, price, area, k)
   area_ridge_mean_error = np.append(area_ridge_mean_error, err)
plt.plot(range(2, max_k+1), area_ridge_mean_error)
plt.title("Compare different K in K-Fold")
plt.xlabel("K")
plt.ylabel("RMSE")

area_lasso_mean_error = np.array([])
for k in range(2,max_k+1):
   err =kfold_mean_error(area_lasso, area, price, k)
   area_lasso_mean_error = np.append(area_lasso_mean_error, err)
plt.plot(range(2, max_k+1), area_lasso_mean_error)
plt.title("Compare different K in K-Fold")
plt.xlabel("K")
plt.ylabel("RMSE")