# libraries 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import style
from sklearn.datasets import load_iris
import pandas as pd
style.use('ggplot')
# data 

#x=np.array([[1,0.2],[1.3,3],[1.3,4],[4.2,8],[9,10]])
#iris=load_iris()
#dataset=iris.data
dataset=pd.read_csv('iris.csv')
x=dataset.iloc[:,[1,2,3,4]].values

#cluster

clf=KMeans(n_clusters=3)
y_kmeans=clf.fit_predict(x)
#training

#clf.fit(x)
#centroid

centroid=clf.cluster_centers_

#label
labels=clf.labels_
#color set for each features
colors=["g.","r.","y.","b."]

#prediction
'''
for i in range(len(x)):
	if i==0:
		p='versicolour'
		colo='pink'
	if i==1:
		p='setosa'
		colo='yellow'
	if i==2:
		p='virginica'
		colo='green'''
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],c='yellow',s=100,label='versicolour')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],c='red',s=100,label='setosa')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],c='pink',s=100,label='virginica')

plt.scatter(centroid[:,0],centroid[:,1],linewidths=5,s=100,marker='*', c = 'green', label = 'Centroids')
#plt.xlabel('Feature_1')
#plt.ylabel('Feature_2')
plt.legend()
plt.show()
'''
#Visualising the clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'green', label = 'versicolour')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'yellow', label = 'setosa')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'blue', label = 'virginica')

#Plotting the centroids of the clusters
plt.scatter(clf.cluster_centers_[:, 0], clf.cluster_centers_[:,1], s = 100, c = 'red', label = 'Centroids')
plt.legend()
plt.show()
print(y_kmeans)
'''