#2_Multiple_Linear_Regression_Profit_Predictor
import pandas as pd 
dataset=pd.read_csv("50_Startups.csv")
dataset

#preprocessing 
#Categorical data preprocessing 
#4th column is State-NewYork,California, Florida --these are nominal dataset all the remaining data numerical
#so we need to perform "one-hot-coding"
#means we need to convert that into numbering format 
dataset=pd.get_dummies(dataset)
dataset
#Now after One-Hot-encoding- the state column turns into several columns with its unique values and marked with zeros 0s and ones 1s
#like State_California, State_Florida,State_NewYork.. Column_UniqueValueName1..Column_UniqueValueName_n in alphabetical order of UniqueValues
#which ever row has that state will be marked with 1 and remaining Column_UniqueValueNames will be marked as zeros
dataset=pd.get_dummies(dataset,drop_first=True)
dataset
#drop_first=True means we are dropping one column out of three to save memory
#to understand if Column_UniqueValueName1 and Column_UniqueValueName2 has zero by default this means Column_UniqueValueName3 has one
#all three columns cannot be Zero or ones
#so if two columns has zero values, by default that means remaing column has value as one
#Quiz
#How do you handle Categorical Data?
#If it is Nominal, we can use One_Hot_Encoding using get_dummies method in pandas
#If it is Ordinal, we can use Label_Encoder

#Finally now we will have 5 input columns and 1 output column which is profit column
dataset.colums
#output
Index(['R&D Spend', 'Administration', 'Marketing Spend', 'Profit','State_California', 'State_Florida', 'State_New York'],dtype='object')

#input
independent_variable = dataset[['R&D Spend', 'Administration', 'Marketing Spend','State_California', 'State_Florida', 'State_New York']
independent_variable

#output
dependent_variable = dataset[["Profit"]]
dependent_variable

#Splitting dataset for training and Testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(independent_variable, dependent_variable, test_size=0.30, random_state=0)

x_train

x_test

y_train

y_test

#Model Creation
from sklearn.linear_model import LinearRegression
regressor_variable=LinearRegression()
regressor_variable.fit(x_train,y_train) 

#we got weight and bias
weight=regressor_variable.coef_
bais=regressor_variable.intercept_

#Predition test
y_predict=regressor_variable.predict(x_test)

#Evaluate Predition metrics
from sklearn.metrics import r2_score
r_score= r2_score(y_test,y_predict)   


import pickle as pk
filename = "2_Multiple_Linear_Regression_Profit_Predictor.pkl"

pk.dump(regressor_variable,open(filename,'wb'))

loaded_model=pk.load(open(filename,'rb'))
#pass five inputs
#inputs=input1,...input5
inputs=[1234,345,4565,1,0]
result=loaded_model.predict([inputs])

result
#array([[43994.79745873]]




#Deployment phase
import pickle
filename = "2_Multiple_Linear_Regression_Profit_Predictor.pkl"
loaded_model=pickle.load(open(filename,'rb'))
result=loaded_model.predict([[10]]) #Testing what should be the salary for 10 Years of experiance
result
