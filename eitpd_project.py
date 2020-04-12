# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:13:52 2019

@author: Dipto
"""
#Loading Important Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Reading the Dataset
df = pd.read_csv("records_backup.csv")

#Statistics
print("Descriptive Statistics About the Dataset")
df.head()
df.shape
df.info()
df.describe(include='all')

#Maximum Concentration Recorded
print("Maximum CO2 PPM Recorded: ","\n",df.loc[df['CO2_PPM'].idxmax()])

#Minimum Concentration Recorded
print("Minimum CO2 PPM Recorded: ","\n",df.loc[df['CO2_PPM'].idxmin()])

#Distribution of CO2 PPM
sns.set(style='darkgrid')
plt.figure(figsize=(20,10))
plt.title(label="Distribution of Carbon Dixoide in Parts Per Million",loc="center")
plt.ylabel(ylabel="Frequency")
sns.distplot(df['CO2_PPM'], kde=False,bins=10)

#Distribution of CO2 PPM in Days of the Week
sns.set(style='darkgrid')
plt.figure(figsize=(20,10))
plt.title(label="Concentration of Carbon Dioxide During the Days")
sns.barplot(y="Day",x="CO2_PPM",data=df)

#Changing the Conditions for the "Time" feature for Visualisation and Machine Learning
# slice first and second string from time column
df['Hour'] = df['Time'].str[0:2]
df['Hour'] = df.Hour.str.replace(':','')

# convert new column to numeric datetype
df['Hour'] = pd.to_numeric(df['Hour'])

# cast to integer values
df['Hour'] = df['Hour'].astype('int')

# define a function that turns the hours into daytime groups
def when_was_it(hour):
    if hour >= 0 and hour < 12:
        return "1"
    elif hour >= 12 and hour < 18:
        return "2"
    elif hour >= 18 and hour < 24:
        return "3"

# create a little dictionary to later look up the groups I created
daytime_groups = {1: 'Morning: Between 9:00 and 12:00', 
                  2: 'Afternoon: Between 12:01 and 18:01', 
                  3: 'Evening: Between 18:01 and 24:00'}

# apply this function to the temporary "Hour" column
df['Daytime'] = df['Hour'].apply(when_was_it)
df['Daytime'] = df['Daytime'].astype('int')


#CO2 PPM during Daytimes
plt.title(label="CO2 Concentrations During Different Daytimes")
sns.pointplot(x='Daytime',y='CO2_PPM',data=df)
plt.xlabel(xlabel="1: Morning, 2: Afternoon, 3: Evening")
plt.show()


#Creating a Decimal Date Column to convert the Date
df['Date']= pd.to_datetime(df['Date'])

#Importing Required Libraries
from datetime import datetime as dt
import time

#Function for Convertion to Decimal Date
def toYearFraction(date):
    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch

    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction


#Creation of Decimal_Date column
df["Decimal_Date"] = df['Date'].apply(toYearFraction)

#Levels of CO2 During the Month
plt.title(label="Levels of CO2 During the Month")
sns.lineplot(x="Decimal_Date",y="CO2_PPM",data=df)
plt.show()

#Creating a Dataframe for Machine Learning
ml_df = df.drop(columns=["Date","Hour","Time"])

#Dummy Encoding
ml_df = pd.get_dummies(data=ml_df,drop_first=True)

# Defining the features 
x = ml_df.drop(['CO2_PPM'], axis=1)

# Defining the target Variable
y = ml_df[['CO2_PPM']]

#Splitting the Training and Test Set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)


#Making Predictions
from sklearn import svm
clf = svm.SVR(kernel="rbf",gamma="scale")
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

#Evaluating Model Performance
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))












