#importing all necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#load the dataset
data=pd.read_csv("Mall_Customers.csv")
#explore rows and columns of it
print(data.head())
print(data.info())
print(data['Gender'].unique())#prints the labels of this Gender column

#Converting gender values into 0 for Male and 1 for female
data['Gender']=data['Gender'].map({"Male":0,"Female":1})
#Considering all the columns that will contribute to the segmentation
X=data[['Gender','Age','Annual Income (k$)','Spending Score (1-100)']]
#All the numeric columns have values belonging to diffrent ranges, for the model to learn well giving importance to every feature,StandardScaler converts the values of all columns into values belonging to one small range.(especially mean=0 and std dev(spread of values)=1)
#fit_transform() scales X dataset and transforms into that form
X_scaled=StandardScaler().fit_transform(X) 

#wcss- within clusters sum of squares measures how close each data point is to its cluster's centroid
#.inertia_ gives the sum of squared distances of all points from their cluster center

