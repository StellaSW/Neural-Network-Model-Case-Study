# import the function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,f1_score,recall_score,precision_score,accuracy_score

# load the data
data = pd.read_table("abalone.data",sep=",",header=None)
# Data preprocessing
sex_dict = {"M":0,"F":1,"I":2}
data[0] = data[0].apply(lambda x:sex_dict[x])
class_number = []
for i in data[8].values:
    if 0 <= i <= 7:
        class_number.append(1)
    elif 8 <= i <= 10:
        class_number.append(2)
    elif 11 <= i <= 15:
        class_number.append(3)
    else:
        class_number.append(4)
data[8] = class_number

# the distribution of class
plt.figure(figsize=(8,6))
sns.countplot(x=data[8])
plt.xlabel("The age")
plt.title("The reporting the distribution of class")
plt.show()
# Correlation analysis between features and labels
plt.figure(figsize=(8,6))
sns.heatmap(data.corr())
plt.title("Correlation analysis between features and labels")
plt.show()
# Divide training set and test set
y = data[8]
x = data.loc[:,data.columns != 8]
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.4,random_state=0)
"""
Investigate the effect of the number of hidden neurons
"""
hidden_neurons = [5,10,15,20,25,30]
performance = []
for neuron in hidden_neurons:
    neural_network = MLPClassifier(hidden_layer_sizes=(neuron),max_iter=1000)
    neural_network.fit(X_train,y_train)
    pred_test = neural_network.predict(X_test)
    performance.append(f1_score(y_test,pred_test,average="macro"))
plt.figure(figsize=(8,6))
plt.plot(hidden_neurons,performance)
plt.xlabel("The number hidden_neurons")
plt.ylabel("The f1 score")
plt.title("Classification performance of different neurons in test data")
plt.show()
"""
Investigate the effect of learning rate (in case of SGD) 
"""
learning_rates = [0.005,0.01,0.05,0.1,0.5]
performance = []
for rate in learning_rates:
    neural_network = MLPClassifier(hidden_layer_sizes=(20,),learning_rate_init=rate)
    neural_network.fit(X_train,y_train)
    pred_test = neural_network.predict(X_test)
    performance.append(f1_score(y_test,pred_test,average="macro"))
plt.figure(figsize=(8,6))
plt.plot(learning_rates,performance)
plt.xlabel("The different learning_rates")
plt.ylabel("The f1 score")
plt.title("Classification performance of different learning rates in test data")
plt.show()
"""
Investigate the effect on a different number of hidden layers 
"""
neural_network = MLPClassifier(hidden_layer_sizes=(20),max_iter=1000)
neural_network.fit(X_train,y_train)
pred_test = neural_network.predict(X_test)
f1_1 = f1_score(y_test,pred_test,average="macro")
print("The F1 score of single-layer neural network is: ",f1_1)

neural_network = MLPClassifier(hidden_layer_sizes=(20,20),max_iter=1000)
neural_network.fit(X_train,y_train)
pred_test = neural_network.predict(X_test)
f1_2 = f1_score(y_test,pred_test,average="macro")
print("The F1 score of double-layer neural network is: ",f1_2)

"""
Investigate the effect of Adam and SGD on training and test performance.
"""
neural_network = MLPClassifier(hidden_layer_sizes=(20,20),solver="adam",max_iter=1000)
neural_network.fit(X_train,y_train)
pred_test = neural_network.predict(X_test)
f1_1 = f1_score(y_test,pred_test,average="macro")
print("The F1 score of  Adam  is: ",f1_1)

neural_network = MLPClassifier(hidden_layer_sizes=(20,20),solver="sgd",max_iter=1000)
neural_network.fit(X_train,y_train)
pred_test = neural_network.predict(X_test)
f1_2 = f1_score(y_test,pred_test,average="macro")
print("The F1 score of SGD is: ",f1_2)

"""
Evaluate the best* model using a confusion matrix and show ROC and AUC for the
classiffcation problems.
"""
final_model = MLPClassifier(hidden_layer_sizes=(20,20),learning_rate_init=0.05,solver="adam")
final_model.fit(X_train,y_train)
pred_test = final_model.predict(X_test)
pred_test_prob = final_model.predict_proba(X_test)
print("The confusion matrix is : \n\n",confusion_matrix(y_test,pred_test))

print("The precision score is :",precision_score(y_test,pred_test,average="macro"))
print("The accuracy score is :",accuracy_score(y_test,pred_test))
print("The recall score is :",recall_score(y_test,pred_test,average="macro"))
print("The f1_score is :",f1_score(y_test,pred_test,average="macro"))

from sklearn.metrics import auc,roc_curve
def plot_roc_curve(label):
    index_list = []
    temp_data = np.where(y_test.values == label,1,0)
    for index,value in enumerate(temp_data):
        if value == 1:
            index_list.append(index)
    fpr, tpr, thresholds = roc_curve(temp_data,pred_test_prob.max(axis=1))
    plt.figure(figsize=(9,6))
    plt.plot(fpr,tpr)
    plt.title("ROC curve about class{}  AUC:{}".format(label,auc(fpr,tpr)))
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.show()

plot_roc_curve(1)
plot_roc_curve(2)
plot_roc_curve(3)
plot_roc_curve(4)