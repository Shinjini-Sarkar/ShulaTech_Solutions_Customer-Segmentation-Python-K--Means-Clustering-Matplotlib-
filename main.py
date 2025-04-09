#importing all necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


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

#wcss- within clusters sum of squares 
#.inertia_ calculates the wcss sum of squared distances of all points from their cluster center
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

#visualization
plt.plot(range(1,11),wcss,marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("wcss-Within Cluster Sum of Squares")
plt.show()

#after checking the graph the elbow point is observed to be 5
kmeans=KMeans(n_clusters=6,random_state=42)

#fit -- Find the clusters (centroids) from the data
#predict -- Assign each data point to the nearest cluster
clusters=kmeans.fit_predict(X_scaled)

#add a new column 'Cluster' to the data
data['Clusters']=clusters
data.info()

#PCA reduces the data into 2 colums(principal components) or 2D graph
pca=PCA(n_components=2) 
X_pca=pca.fit_transform(X_scaled)

#figsize determines the height and width of the figure
plt.figure(figsize=(8,6))
#x=X_pca[:,0]: Use the first PCA component for x-axis
#y=X_pca[:,1]: Use the second PCA component for y-axis
#hue=clusters: Color the dots based on the cluster each point belongs to
#palette='Set1': Choose a colorful color set
#s=100: Size of each point (bigger dots)
sns.scatterplot(x=X_pca[:,0],y=X_pca[:,1],hue=clusters,palette="Set1",s=100)
plt.title("Customer Segments(PCA+KMeans)")
plt.xlabel('PCA Component 1')
plt.ylabel("PCA Component 2")
plt.savefig("images/2d_customer_segments.png")#save the output in the images folder
plt.show()

#drawing a 3D plot
#add_subplot(111):1 row, 1 column, 1 plot
pca_3d=PCA(n_components=3)
X_3d=pca_3d.fit_transform(X_scaled)
fig=plt.figure(figsize=(10,10))
plot_3d=fig.add_subplot(111,projection='3d')

#X_3d[:, 0] → x-axis (1st principal component)
#X_3d[:, 1] → y-axis (2nd component)
#X_3d[:, 2] → z-axis (3rd component)
#c=clusters → color dots by their cluster label
#cmap='Set1' → use a bright, clean color palette
#s=100 → dot size (100 = medium-large)
plot_3d.scatter(X_3d[:,0],X_3d[:,1],X_3d[:,2],c=clusters,cmap="Set1",s=100)
plot_3d.set_title("3D Visualization of Customer Segmentation")
plot_3d.set_xlabel("Component 1")
plot_3d.set_ylabel("Component 2")
plot_3d.set_zlabel("Component 3")
#display the 3d plot

plt.savefig("images/3d_customer_segments.png")#save the image of the 3d plot in images folder
plt.show()



