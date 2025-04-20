# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
5.Display the results.
```
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: G SANJAY
RegisterNumber:  212224230243
*/
```PY
import pandas as pd
data=pd.read_csv("C:\\Users\\admin\\Downloads\\Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#removes the specified row or column
data1.head()

data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")# A Library for Large Linear Classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion=(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
```

## Output:
![image](https://github.com/user-attachments/assets/da66d623-d233-4e2b-b4e0-eceb027d7620)
![image](https://github.com/user-attachments/assets/4a525564-c232-4c71-9b71-770412789501)
![image](https://github.com/user-attachments/assets/1e1bf37e-06cf-4f33-9c0c-ad68c6253c6a)
![image](https://github.com/user-attachments/assets/4623478a-c801-40e4-957e-6bb7d9165f21)
![image](https://github.com/user-attachments/assets/764a1c16-fd0c-4200-b6fb-42e25130d77e)
![image](https://github.com/user-attachments/assets/e29779fa-4ecb-425b-8593-5affac4c797b)
![image](https://github.com/user-attachments/assets/8464dddc-423e-47de-ae4e-20fd0fe48a34)
![image](https://github.com/user-attachments/assets/0c2c24d4-7062-4477-aa08-8f35233c6979)
![image](https://github.com/user-attachments/assets/14eef1d5-8a27-48b0-a95c-d65ed3e901de)
![image](https://github.com/user-attachments/assets/953b4207-53ae-4610-a880-75a446a49c13)
![image](https://github.com/user-attachments/assets/b47ac50c-49d4-485b-a68d-dc96a77c191b)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
