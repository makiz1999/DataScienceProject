import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time

#Read data from iris.csv and store in the data frame (only virginica and versicolor)
df = pd.read_csv('iris.csv',
                 usecols=['sepal_length','sepal_width','petal_length','petal_width','species'],
                 index_col='species',
                 skiprows=[i for i in range(1,51)])

species = pd.read_csv('iris.csv',
                      usecols=['species'],
                      skiprows=[i for i in range(1,51)]
                      ).T.values.tolist()[0]

#Running time for PCA

begin = time.time()

#Applying PCA
scaled_data = preprocessing.scale(df)
pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)

#Finding PCA components
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]

plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Screen Plot')
plt.show()

pca_df = pd.DataFrame(pca_data, index=species, columns=labels)

#Plotting PCA data
plt.figure(2)
plt.title('My PCA Graph')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))

i = 0
flower = ["versicolor","virginica"]
color = ["orange","blue"]
for sample in pca_df.index:
    if sample == flower[0]:
        curColor = color[0]
    if sample == flower[1]:
        curColor = color[1]

    plt.scatter(pca_df.PC1[i], pca_df.PC2[i], c = curColor)
    i = i+1

orange_patch = mpatches.Patch(color='orange', label='Versicolor')
blue_patch = mpatches.Patch(color='blue', label='Virginica')
plt.legend(handles=[blue_patch, orange_patch ])

# Determine which grades had the biggest influence on PC1
loading_scores = pd.Series(pca.components_[0], index=['sepal_length','sepal_width','petal_length','petal_width'])

## Sort the scores
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)

# Show the names of the grades
top_4_features = sorted_loading_scores[0:10].index.values

print("---PCA---")
print("Execution time for PCA:", time.time() - begin,"seconds")
print("Most important principal components:\n"
      "PC1:", per_var[0],
      "PC2:", per_var[1])
print("Biggest influence features:")
print(loading_scores[top_4_features])
plt.show()
#SVM

#Prepare data for training
pc1 = pca_df['PC1'].to_numpy()
pc2 = pca_df['PC2'].to_numpy()

#Running time for SVM

begin = time.time()

#Training
training_X = np.vstack((pc1, pc2)).T
# preparing the labeling
training_Y = np.array(species)

clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(training_X, training_Y)

# weights
w = clf.coef_[0]

# offset
a = -w[0] / w[1]

XX = np.linspace(-4, 4)
yy = a * XX - clf.intercept_[0] / w[1]

#Plotting data
plt.figure(3)
plt.plot(XX, yy, 'k-')
plt.scatter(pc1[0:50], pc2[0:50], color='orange')
plt.scatter(pc1[50:100], pc2[50:100], color='blue')
plt.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], marker='s',edgecolors='black', facecolors='none')

orange_patch = mpatches.Patch(color='orange', label='Versicolor')
blue_patch = mpatches.Patch(color='blue', label='Virginica')
plt.legend(handles=[blue_patch, orange_patch ])
plt.title('SVM')

#Testing accuracy
index = 0
correct = 0
incorrect = 0
for item in training_X:
    #print(clf.predict([[item[0],item[1]]]))
    #print(training_y[index])
    if clf.predict([[item[0],item[1]]]) == training_Y[index]:
        correct += 1
    else:
        incorrect += 1
    index += 1


print("\n---SVM---")
print("Execution time for SVM:", time.time() - begin, "seconds")
print("Accuracy of prediction of SVM:", (correct/(correct+incorrect))*100, "%")
plt.show()

#Neural Network

#Running time for Neural Network

begin = time.time()

#Training
nn = MLPClassifier(solver='sgd', learning_rate='constant',learning_rate_init=0.0005, hidden_layer_sizes=(160,140), max_iter =2000, random_state=1)
nn.fit(training_X, training_Y )

# Plotting Neural Network results
plt.figure(4)
for item in training_X:
    ans = nn.predict([[item[0],item[1]]])
    if ans=='versicolor':
        plt.scatter(item[0], item[1], color='orange')
    else:
        plt.scatter(item[0], item[1], color='blue')
plt.title('Neural Network')
orange_patch = mpatches.Patch(color='orange', label='Versicolor')
blue_patch = mpatches.Patch(color='blue', label='Virginica')
plt.legend(handles=[blue_patch, orange_patch ])
plt.show()


#Comparing the predictions against the actual observations
yp = nn.predict(training_X)

#check how many of them are predicted well
count  = 0;
for i in range(len(training_Y)):
    if yp[i] == training_Y[i]:
        count +=1

accuracy = count/len(training_Y)*100

#Printing the accuracy
print("\n---Neural Network---")
print("Execution time for Neural Network:", time.time() - begin, "seconds")
print('Accuracy of MLPClassifier(in percentage) :', accuracy)
print('Final Loss Value :', nn.loss_)
print('Final Iterations :', nn.n_iter_)

#K-Means

#Running time for K-Means

begin = time.time()

#Need only 2 clusters
kmeans = KMeans(n_clusters=2, random_state=0, max_iter=200)
kmeans.fit(training_X)


#Plotting K-Means data
plt.figure(5)
i = 0
for item in training_X:
    if kmeans.labels_[i]==0:
        plt.scatter(item[0], item[1], color='orange')
    else:
        plt.scatter(item[0], item[1], color='blue')
    i += 1
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, marker='x', c='green')
orange_patch = mpatches.Patch(color='orange', label='Versicolor')
blue_patch = mpatches.Patch(color='blue', label='Virginica')
plt.legend(handles=[blue_patch, orange_patch ])
plt.title('K-Means')
plt.show()

#Testing accuracy
index = 0
correct = 0
incorrect = 0
for item in training_X:
    if kmeans.predict([[item[0],item[1]]]) == 0:
        correct += 1
    else:
        incorrect += 1
    index += 1

# results
print("\n---K-Means---")
print("Execution time for PCA:", time.time() - begin, "seconds")
print("K-Means score:",kmeans.score(training_X))
print("Accuracy of prediction of K-Means:", (correct/(correct+incorrect))*100, "%")