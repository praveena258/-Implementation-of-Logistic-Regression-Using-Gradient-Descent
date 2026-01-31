# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.  START
2. Import the data file and import numpy, matplotlib and scipy.
3. Visulaize the data and define the sigmoid function, cost function and gradient descent.
4. Plot the decision boundary .
5. Calculate the y-prediction.
6.  STOP

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: PRAVEENA D
RegisterNumber:  212224040248
*/
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: PAVITHRA S
RegisterNumber:  212223220072
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("Placement_Data.csv")
dataset

dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset

x=dataset.iloc[:, :-1].values
y=dataset.iloc[: ,-1].values
y

theta=np.random.randn(x.shape[1])
Y=y
def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,x,Y):
      h=sigmoid(x.dot(theta))
      return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,x,Y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(x.dot(theta))
        gradient=x.T.dot(h-y)/m
        theta-=alpha * gradient
    return theta

theta=gradient_descent(theta,x,Y,alpha=0.01,num_iterations=1000)

def predict(theta,x):
    h=sigmoid(x.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred
y_pred=predict(theta,x)

accuracy=np.mean(y_pred.flatten()==Y)
print("Accuracy:",accuracy)

print(y_pred)
print(y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
*/
```

## Output:

## dataset

<img width="957" height="337" alt="image" src="https://github.com/user-attachments/assets/0fa58842-2cdd-4e77-a9d6-9b3d02197c75" />

## dtypes

<img width="333" height="239" alt="image" src="https://github.com/user-attachments/assets/e60b75ef-4045-4724-8ae9-c3b83d988d6d" />

<img width="821" height="363" alt="image" src="https://github.com/user-attachments/assets/d4e5c886-5bb2-45fa-93c1-eafc3116822d" />

## y

<img width="674" height="187" alt="image" src="https://github.com/user-attachments/assets/7a530437-997a-418b-ae7e-424d222d988b" />

## Accuracy

<img width="286" height="50" alt="image" src="https://github.com/user-attachments/assets/5d255571-d8fe-4300-9dcb-54777cb247d4" />

## Predicted

<img width="861" height="571" alt="image" src="https://github.com/user-attachments/assets/602da764-e318-49b8-9edc-9a8e8ea1f8d3" />





## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

