![WhatsApp Image 2025-12-19 at 08 36 10_3af858d3](https://github.com/user-attachments/assets/4038c954-2b5a-40e7-8a5f-725d3cd4fd2f)# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
### Step1
import numpy required libraries such as Numpy,matplotlib,scikit learn for data handeling,visualization,and model creation.


### Step2

load the dataset and seperate it into independent variables(x)
and dependent variables(y)
### Step3

split the dataset into trainig data and testing data using the train-test split method
### Step4
create a multivariant linear regression model and train it using the training dataset



## Program:
```
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear model, metrics
from sklearn.model_selection import train_test_split
housing = datasets.fetch_california_housing()
X = housing.data
Y = housing. target

X_train, x_test, y_train, y_test = train_test_split(x, y, test_size-0.4, random_state=1)

reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

print('Coefficients:', reg.coef_)
print('Variance score: ({}'.format(reg.score(x_test, y_test)))

plt.style.use('fivethirtyeight')
plt.scatter(reg.predict(x_train), reg.predict(X_train) - y_train, color="green”, 5=10, label='Train data')
plt.scatter(reg.predict(X_test),reg.predict(x_test)- y_test, color="blue", 5=10, label='Test data')
plt.hlines(y=0,xmin=0，xmax=5,linewidth=2)
plt.legend(1oc='upper right')
plt.title('Residual Errors')
plt.show()
```
## Output:
![WhatsApp Image 2025-12-19 at 08 36 10_3af858d3](https://github.com/user-attachments/assets/371d15e0-b14c-49ed-a8f8-e12867efce17)

## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
