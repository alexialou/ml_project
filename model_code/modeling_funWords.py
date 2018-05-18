import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#for server
#plt.switch_backend('agg')
import collections as co
from operator import itemgetter

f = open('model_output.txt','w')
# load data
data = pd.read_csv("cleaned.csv")

# data size
print ('The data has {0} rows and {1} columns\n'.format(data.shape[0],data.shape[1]))
f.write('The data has {0} rows and {1} columns\n'.format(data.shape[0],data.shape[1]))

# info about the law schoos in the dataset
print("The info about the law schools:\n")
print(data['clerk_school'].describe())


# top 10 law schools in the whole dataset
schCnt = co.Counter(data['clerk_school'])
top10 = schCnt.most_common(10)
print("The top 10 law schools in the whole data set:\n",top10)
f.write("The top 10 law schools in the whole data set:\n {} \n".format(top10))

# plot the distribution of the law school for the whole dataset
proDict = {}
for sch, cnt in schCnt.items():
    proDict.update({sch: cnt/ data.shape[0]})

y_pos = np.arange(len(proDict.items())) 
plt.bar(y_pos,proDict.values() , align='center', alpha=0.5)
plt.xticks(y_pos, proDict.keys() , fontsize = 7)
plt.ylabel('Probability')
plt.title('Histogram of law school')
plt.show()


# Model Training and Evaluation
y = data['clerk_school']
X = data.drop(['clerk_school','judge'], axis=1)
#split trainng dataset and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=44)

# using the school distribution to do the prediction
train_sch_cnt = co.Counter(y_train)
train_top10 = train_sch_cnt.most_common(10)
print("The top 10 law schools in the training data set:\n {}".format(train_top10))
f.write("The top 10 law schools in the training data set:\n {} \n".format(train_top10))

# plot the distribution of top 10 law schools for the training dataset
train_top10Sch = []
train_top10Prob = []
for i in range(len(train_top10)):
    train_top10Sch.append(train_top10[i][0])
    train_top10Prob.append(train_top10[i][1]/len(y_train))
y_pos = np.arange(len(train_top10Sch))
plt.bar(y_pos,train_top10Prob , align='center', alpha=0.5)
plt.xticks(y_pos, train_top10Sch , fontsize = 7)
plt.ylabel('Probability')
plt.title('Histogram of top10 law schools in training dataset')
plt.show()

# using the top1 school in training set (Harvard) as the predicted school for all the casese in test set
train_cnt = 0
for y in y_train:
    if y == 'Harvard Law School':
        train_cnt += 1
train_score = train_cnt / len(y_train)
print("The accuracy on training set by using the school distribution to do the prediction: \n", train_score)
f.write("The accuracy on training set by using the school distribution to do the prediction: \n {} \n".format(train_score))

test_cnt = 0
for y in y_test:
    if y == 'Harvard Law School':
        test_cnt += 1
test_score = test_cnt / len(y_test)
print("The accuracy on test set by using the school distribution to do the prediction: \n", test_score)
f.write("The accuracy on test set by using the school distribution to do the prediction: \n {} \n".format(test_score))

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_score_train = nb.score(X_train, y_train)
print("Naive Bayes accuracy on training set: ", nb_score_train)
f.write("Naive Bayes accuracy on training set: {} \n".format(nb_score_train))
nb_score_test = nb.score(X_test, y_test)
print("Naive Bayes accuracy on test set: ", nb_score_test)
f.write("Naive Bayes accuracy on test set: {} \n".format(nb_score_test))

# Logit
from sklearn import linear_model
logreg = linear_model.LogisticRegression(C=1e8)
logreg.fit(X_train, y_train)
logreg_train = logreg.score(X_train, y_train)
print("Logit accuracy on training set: ", logreg_train)
f.write("Logit accuracy on training set: {} \n".format(logreg_train))
logreg_test = logreg.score(X_test, y_test)
print("Logit accuracy on test set: ", logreg_test)
f.write("Logit accuracy on test set: {} \n".format(logreg_test))


# Logit wich PCA
# PCA
from sklearn import linear_model
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
X_r = pca.fit(X).transform(X)
X_r.shape
y = data['clerk_school']
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_r, y, test_size=0.25, random_state=44)
logreg_PCA = linear_model.LogisticRegression(C=1e8)
logreg_PCA.fit(Xr_train, yr_train)
logreg_PCA_train = logreg_PCA.score(Xr_train, yr_train)
print("Logit with PCA accuracy on training set: ", logreg_PCA_train)
f.write("Logit with PCA accuracy on training set: {} \n".format(logreg_PCA_train))
logreg_PCA_test = logreg_PCA.score(Xr_test, yr_test)
print("Logit with PCA accuracy on test set: ", logreg_PCA_test)
f.write("Logit with PCA accuracy on test set: {} \n".format(logreg_PCA_test))


# boosting
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
gbst = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=44)
gbst.fit(X_train, y_train)
gbst_train = gbst.score(X_train, y_train)
print("boosting accuracy on training set: ", gbst_train)
f.write("boosting accuracy on training set: {} \n".format(gbst_train))
gbst_test = gbst.score(X_test, y_test)
print("boosting accuracy on test set: ", gbst_test)
f.write("boosting accuracy on test set: {} \n".format(gbst_test))

# Random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators= 100, random_state=44)
rf.fit(X_train, y_train)
rf_train = rf.score(X_train, y_train)
print("Random forest (with 100 trees) accuracy on training set: ", rf_train)
f.write("Random forest (with 100 trees) accuracy on training set: {} \n".format(rf_train))
rf_test = rf.score(X_test, y_test)
print("Random forest (with 100 trees) accuracy on test set: ", rf_test)
f.write("Random forest (with 100 trees) accuracy on test set: {} \n".format(rf_test))

rf1 = RandomForestClassifier(n_estimators= 500, random_state=1)
rf1.fit(X_train, y_train)
rf1_train = rf1.score(X_train, y_train)
print("Random forest (with 500 trees) accuracy on training set: ", rf1_train)
f.write("Random forest (with 500 trees) accuracy on training set: {} \n".format(rf1_train))
rf1_test = rf1.score(X_test, y_test)
print("Random forest (with 500 trees) accuracy on training set: ", rf1_test)
f.write("Random forest (with 500 trees) accuracy on training set: {} \n".format(rf1_test))


# nnet
from sklearn.neural_network import MLPClassifier
nnet = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(3, 300), random_state=44)
nnet.fit(X_train, y_train)
nnet_train = nnet.score(X_train, y_train)
print("Neural network accuracy on training set: ", nnet_train)
f.write("Neural network accuracy on training set: {} \n".format(nnet_train))
nnet_test = nnet.score(X_test, y_test)
print("Neural network accuracy on training set: ", nnet_test)
f.write("Neural network accuracy on training set: {} \n".format(nnet_test))

f.close()



