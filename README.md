# A Regression & A Classification Problem
In This Project, We Have 2 parts.
<br />
The First Part is a Regression & the second part is a classification problem.
## Regression Problem
This part consist of 3 types of regression. Linear Regression, Lasso Regression & Ridge Regression.
<br />
Dataset which is used for is problem is a kaggle competition for house price prediction.
<br />
More detail about the challenge can be found here:
<br />
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview
<br />
<br />
The library which is used for the mentioned algorithm is Scikit-Learn.
### Preprocessing of dataset
For this purpose, some columns that we did not need and also column with lots of missing data were dropped and also missing data filled with last valid value.
So 73 input feature is remained as the input of algorithms.
### Scatter Plots of Price vs Area & also Price vs Rooms
![Price Area Scatter](https://github.com/ahmad-sohrabi/regression_classification/blob/main/regression_result/price_area_scatter.png?raw=true)
![Price Rooms Scatter](https://github.com/ahmad-sohrabi/regression_classification/blob/main/regression_result/price_rooms_scatter.png?raw=true)
### Regressions Result
The Result which is obtained from 3 mentioned algorithms are compared with each other.
<br />
Comparison of 3 algorithms in a bar plot is like this:
<br />
![RMSE Bar](https://github.com/ahmad-sohrabi/regression_classification/blob/main/regression_result/rmse_bar_comparison_dummy.png?raw=true)
<br />
![R2 Bar](https://github.com/ahmad-sohrabi/regression_classification/blob/main/regression_result/r2_bar_comparison.png?raw=true)

### K-Fold Cross validation of models
For K from 2 to 10, K-Fold cross validation Mean R2 Score is calculated and plotted. As a sample you can see Linear regression K-Fold plot:
<br />
![Linear Regression K-Fold](https://github.com/ahmad-sohrabi/regression_classification/blob/main/regression_result/R2KFoldLinear.png?raw=true)

## Classification Problem
This part consist of a logistic regression.
<br />
Dataset which is used for is problem is a kaggle competition for titanic ship survival prediction.
<br />
More detail about the challenge can be found here:
<br />
https://www.kaggle.com/competitions/titanic/overview
<br />
<br />
The library which is used for the mentioned algorithm is Scikit-Learn.

### Preprocessing of dataset
For this purpose, some columns that we did not need and also column with lots of missing data were dropped and also missing data filled with last valid value.

### Regressions Result
The Result which is obtained from logistic algorithm is displayed as a confusion matrix
<br />

![Confusion Matrix](https://github.com/ahmad-sohrabi/regression_classification/blob/main/classification_result/Confusion%20Matrix.png?raw=true)

### K-Fold Cross validation of logistic regression model
For K from 2 to 10, K-Fold cross validation Mean Accuracy score is calculated and plotted.
<br />
![Logistic Regression K-Fold](https://github.com/ahmad-sohrabi/regression_classification/blob/main/classification_result/KFold.png?raw=true)


