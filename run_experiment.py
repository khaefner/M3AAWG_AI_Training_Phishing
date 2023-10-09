#This hides some of the warnings we get in MLP
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd  #Pandas is a data manipulation library
import numpy as np   # numpy is computing library that uses C libraries in the backend
from sklearn.model_selection import StratifiedKFold   #This gives us nice 'slices' of examples for training and testing

from sklearn.neighbors import KNeighborsClassifier  # K Nearest Neighbors
from sklearn.tree import DecisionTreeClassifier  # Decision Trees
from sklearn.ensemble import RandomForestClassifier  # Random Forrest Classifier
from sklearn.neural_network import MLPClassifier   #Neural Network Classifier

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score  #Libraries for calculating scores.

import lib.url_features
import lib.content_features
import lib.external_features
import lib.feature_extractor as fextract2

phishing_data = pd.read_csv("https://raw.githubusercontent.com/khaefner/M3AAWG_AI_Training_Phishing/main/dataset_phishing.csv")

print(phishing_data)

#Get rid of the first column:
phishing_data = phishing_data.iloc[:, 1:]
#Print the result
print(phishing_data)

#Change the label classes to a one or a zero
phishing_data['status'] = phishing_data['status'].replace({'legitimate': 0, 'phishing': 1})
#Print the result
print(phishing_data)

#First the Labels
y = phishing_data["status"].values
print(y)
#Second the example data
X = phishing_data.drop("status", axis=1).values
print(X)

correlation_matrix = phishing_data.corr()
sorted_corr = correlation_matrix.sort_values(by='status',ascending=False)

print(sorted_corr['status'].head(50))

def KNN(X,y):
  skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
  accuracy = []
  precision = []
  recall = []
  f1 = []


  # we are going to run the model 10 times 'n_splits=10' each time we shuffle the data randomly.
  #This helps prevent our model from overfitting.
  for train, test in skf.split(X,y):
     X_train, y_train = X[train], y[train] #training
     X_test, y_test = X[test], y[test] #testing

     knn = KNeighborsClassifier(n_neighbors=3)
     knn.fit(X_train, y_train)

     y_pred = knn.predict(X_test)

     accuracy.append(accuracy_score(y_test, y_pred))
     recall.append(recall_score(y_test, y_pred, average='macro'))
     precision.append(precision_score(y_test, y_pred, average='macro'))
     f1.append(f1_score(y_test, y_pred, average='macro'))


  average_accuracy = np.mean(accuracy)
  average_recall = np.mean(recall)
  average_precision = np.mean(precision)
  average_f1 = np.mean(f1)

  print(f"Acurracy: {average_accuracy}")
  print(f"Recall: {average_recall}")
  print(f"Precision:{average_precision}")
  print(f"F1 Score:{average_f1}")

  return knn

def DT(X,y):
  skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
  accuracy = []
  precision = []
  recall = []
  f1 = []


  for train, test in skf.split(X,y):
     X_train, y_train = X[train], y[train] #training
     X_test, y_test = X[test], y[test] #testing

     dt = DecisionTreeClassifier(criterion='gini') #Gini is a measure of statistical dispersion that quantifies the inequality or impurity within a set of values,
     dt.fit(X_train, y_train)

     y_pred = dt.predict(X_test)

     accuracy.append(accuracy_score(y_test, y_pred))
     recall.append(recall_score(y_test, y_pred, average='macro'))
     precision.append(precision_score(y_test, y_pred, average='macro'))
     f1.append(f1_score(y_test, y_pred, average='macro'))


  average_accuracy = np.mean(accuracy)
  average_recall = np.mean(recall)
  average_precision = np.mean(precision)
  average_f1 = np.mean(f1)

  print(f"Acurracy: {average_accuracy}")
  print(f"Recall: {average_recall}")
  print(f"Precision:{average_precision}")
  print(f"F1 Score:{average_f1}")

  return dt

#Random Forrest
def RF(X,y):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accuracy = []
    precision = []
    recall = []
    f1 = []


    # we are going to run the model 10 times 'n_splits=10' each time we shuffle the data randomly.
    #This helps prevent our model from overfitting.
    for train, test in skf.split(X,y):
      X_train, y_train = X[train], y[train] #training
      X_test, y_test = X[test], y[test] #testing


      # Create a Random Forest classifier
      rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

      # Fit the classifier to the training data
      rf_classifier.fit(X_train, y_train)

      # Make predictions on the test data
      y_pred = rf_classifier.predict(X_test)

      accuracy.append(accuracy_score(y_test, y_pred))
      recall.append(recall_score(y_test, y_pred, average='macro'))
      precision.append(precision_score(y_test, y_pred, average='macro'))
      f1.append(f1_score(y_test, y_pred, average='macro'))

    average_accuracy = np.mean(accuracy)
    average_recall = np.mean(recall)
    average_precision = np.mean(precision)
    average_f1 = np.mean(f1)

    print(f"Acurracy: {average_accuracy}")
    print(f"Recall: {average_recall}")
    print(f"Precision:{average_precision}")
    print(f"F1 Score:{average_f1}")

    return rf_classifier

#Neural Nets  We'll use the Multi-Layer Perceptron MLP
#Important hyper-paramters are Learning_rate  This affects how fast the algorihms converges (minimizes errors).
#A learning rate that is to high will lead to sub-optimal solutions, too low and it will take forever.
def MLP(X,y):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accuracy = []
    precision = []
    recall = []
    f1 = []


    # we are going to run the model 10 times 'n_splits=10' each time we shuffle the data randomly.
    #This helps prevent our model from overfitting.
    for train, test in skf.split(X,y):
      X_train, y_train = X[train], y[train] #training
      X_test, y_test = X[test], y[test] #testing

      mlp = MLPClassifier(
            solver='adam',            #‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba
            hidden_layer_sizes=(20,),
            activation='tanh',
            max_iter=20,
            validation_fraction=0.2,
            learning_rate_init=0.01,   #The amount that the weights are updated a small positive value, often in the range between 0.0 and 1.0.
        )

      # Fit (train) the classifier
      mlp.fit(X_train,y_train)

      #Predict the results using the set-aside test data.
      y_pred = mlp.predict(X_test)

      accuracy.append(accuracy_score(y_test, y_pred))
      recall.append(recall_score(y_test, y_pred, average='macro'))
      precision.append(precision_score(y_test, y_pred, average='macro'))
      f1.append(f1_score(y_test, y_pred, average='macro'))

    average_accuracy = np.mean(accuracy)
    average_recall = np.mean(recall)
    average_precision = np.mean(precision)
    average_f1 = np.mean(f1)

    print(f"Acurracy: {average_accuracy}")
    print(f"Recall: {average_recall}")
    print(f"Precision:{average_precision}")
    print(f"F1 Score:{average_f1}")

    return mlp


def feature_selection(num_top_correlated=10):
    top_list = sorted_corr['status'].head(num_top_correlated)
    print(top_list)
    X_orig = phishing_data.drop("status", axis=1)
    top_features=sorted_corr[1:num_top_correlated+1].index
    print(top_features)
    return X_orig[top_features].values


knn = KNN(X,y)
dt = DT(X,y)
rf = RF(X,y)
mlp = MLP(X,y)

X_prime = feature_selection()

print("----------------")
print("KNN All Features")
print("________________")
KNN(X,y)
print("----------------")
print("KNN Selected Features")
print("________________")
KNN(X_prime,y)
print("\n")

print("----------------")
print("DT All Features")
print("________________")
DT(X,y)
print("----------------")
print("DT Selected Features")
print("________________")
DT(X_prime,y)
print("\n")

print("----------------")
print("RF All Features")
print("________________")
RF(X,y)
print("----------------")
print("RF Selected Features")
print("________________")
RF(X_prime,y)
print("\n")

print("----------------")
print("MLP All Features")
print("________________")
MLP(X,y)
print("----------------")
print("MLP Selected Features")
print("________________")
MLP(X_prime,y)



url="https://cnn.com"
site_data = fextract2.extract_features(url)
print(site_data)

site_data.pop() #we don't need the label...this is what we want to predict
site_data = site_data[1:]   #slice off the url
print(site_data)
site_data_array = np.array(site_data, dtype=float) # convert to numpy array
reshaped_site_data_array = site_data_array.reshape(1,-1)
print(reshaped_site_data_array)
print(reshaped_site_data_array.shape)
result = knn.predict(reshaped_site_data_array)
print(f"KNN: {result}")
print("Phishing" if result == 1 else "Not Phishing")
result = dt.predict(reshaped_site_data_array)
print(f"DT: {result}")
print("Phishing" if result == 1 else "Not Phishing")
result =rf.predict(reshaped_site_data_array)
print(f"RF: {result}")
print("Phishing" if result == 1 else "Not Phishing")
result = mlp.predict(reshaped_site_data_array)
print(f"MLP: {result}")
print("Phishing" if result == 1 else "Not Phishing")

"""
print(f"KNN: {result}")
result = dt.predict(reshaped_site_data_array)
probabilities =  dt.predict_proba(reshaped_site_data_array)
positive_class_probability = probabilities[0, 1]
print(f"DT: {result}, {positive_class_probability}")
result =rf.predict(reshaped_site_data_array)
probabilities =  rf.predict_proba(reshaped_site_data_array)
positive_class_probability = probabilities[0, 1]
print(f"RF: {result}, {positive_class_probability}")
result = mlp.predict(reshaped_site_data_array)
print(f"MLP: {result}")
"""





