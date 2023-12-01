import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import cross_val_score

data = pd.read_csv("train_classification.csv")
data.drop(columns=['PassengerId', 'Cabin', 'Name', 'Ticket'], inplace=True)

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

input_x = data.iloc[:, 1:].values
target = data.iloc[:, 0].values
input_train, input_test, target_train, target_test = train_test_split(input_x, target, test_size=0.20, random_state=42)
logistic_reg = LogisticRegression(max_iter=10000)
logistic_reg.fit(input_train, target_train)
predictions = logistic_reg.predict(input_test)
score = logistic_reg.score(input_test, target_test)

print(f"Accuracy Score For Test Data is: {score}")

confusion_matrix = metrics.confusion_matrix(target_test, predictions)
sns.heatmap(confusion_matrix, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=13)
plt.show()

max_k = 10
log_reg_mean_scores = np.array([])
for k in range(2, max_k + 1):
    cv = KFold(n_splits=k)
    model = LogisticRegression(max_iter=10000)
    scores = cross_val_score(model, input_x, target, scoring='accuracy', cv=cv, n_jobs=-1)
    log_reg_mean_scores = np.append(log_reg_mean_scores, np.mean(scores))
plt.plot(range(2, max_k + 1), log_reg_mean_scores, linewidth=5)
plt.title("Compare different K in K-Fold")
plt.xlabel("K")
plt.ylabel("Mean of Scores")
plt.show()
