"""
Some examples on ensembles
"""

# Bagging witj kNN
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

#Call function to generate the dataset
X, y = create_artificial_dataset('moons', 100)

#Perform train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    random_state=42)

#Creating a BaggingClassifier() classification model object
#We specify that the base learner is kNN (i.e. KNeighborsClassifier())
#& that ensemble will consist of 10 base models (n_estimators = 10)
clf = BaggingClassifier(KNeighborsClassifier(), n_estimators = 10)

model_name = 'Bagging Ensemble of 10 kNN Classifiers'
print(model_name)

#Train the model on the training set
start_tr = timer()
clf.fit(X_train, y_train)
end_tr = timer()
print('Model training time (sec): '+str(end_tr - start_tr)+
      '\n(i.e. '+str((end_tr - start_tr)/len(y_train))+' per datapoint)')

#Evaluate model on the test set:
start_te = timer()
accuracy = clf.score(X_test, y_test)
end_te = timer()
print('Model inference time (sec): '+str(end_te - start_te)+
      '\n(i.e. '+str((end_te - start_te)/len(y_test))+' per datapoint)')

print('The accuracy of the classifier on the test set is: '+str(accuracy))


#Visualize the decison boundary & decision regions along with the training & test data
plt.figure(figsize = (12,9))

#Show the decison boundary & decision regions along with the training data
plot_decision_regions(X = X_train, y=y_train, clf=clf, legend=2)

#Adding test data to the figure:
X_test_class0 = X_test[y_test == 0]
plt.scatter(X_test_class0[:, 0], X_test_class0[:, 1], c='#1f77b4', marker = 's',
            edgecolors="k", alpha=0.5)
X_test_class1 = X_test[y_test == 1]
plt.scatter(X_test_class1[:, 0], X_test_class1[:, 1], c='#ff7f0e', marker = '^',
            edgecolors="k", alpha=0.5)

#Adding axes annotations
plt.xlabel('Feature X1')
plt.ylabel('Feature X2')
plt.title('Decision Boundary of '+model_name)
plt.show()