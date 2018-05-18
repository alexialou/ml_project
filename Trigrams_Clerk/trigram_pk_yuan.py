import os
#os.chdir("/scratch/cg3015/clerks/output_bigrams")
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import decomposition
import nltk
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import numpy as np
import pickle
from collections import Counter
import glob
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from ast import literal_eval
import sys
from sklearn.model_selection import train_test_split
import scipy


file_name = "1995_trigrams_judge_clerk.csv"
file_name = "14year_trigrams_judge_clerk.csv"
file_name = "1996_parsed_2grams_judge_clerk.csv"
file_name = "trigrams_judge_clerk/all_trigrams.csv"
#file_name = "18_year.csv"
#file_name = "1996_parsed_trigrams_96.csv"
file_name = "all_trigrams.csv"
do_pca = 0
def PCA(features):
    tsvd = decomposition.TruncatedSVD(n_components=1000)
    tsvd.fit(features)
    new_features = tsvd.transform(features)
    return new_features

def convert(doc):
    dict = {}
    if doc == '{}':
        return dict
    doc = doc.replace('{', '')
    doc = doc.replace('}', '')
    doc = doc.split(',')
    for i in range(len(doc)):
        doc[i] = doc[i].split(':')
        doc[i][1] = int(doc[i][1])
        doc[i][0] = doc[i][0].replace('\'', '')
        doc[i][0] = doc[i][0].replace(' ', '')
    for i in range(len(doc)):
        dict[int(doc[i][0])] = doc[i][1]
    return dict

def loadData(file_name):
    data = pd.read_csv(file_name, sep = ',')
    print("len of data is %d " % len(data))
    feature_map = []
    label = []
    for i in range(len(data)):
        feature = data.loc[i, 'fw_count']
        feature_map.append(convert(feature))
        #print(convert(feature))
        #break
        label.append(data.loc[i, 'clerk_school'])
    return feature_map, label

def vectorizeFeature(features):
    vec = DictVectorizer()
    transformed_features = vec.fit_transform(features)
    print(type(transformed_features))
    print(transformed_features.nnz)
    print(transformed_features.dtype.itemsize)
    X_scaled = preprocessing.scale(transformed_features.toarray(),with_mean=False)
    return X_scaled

def feature_norm(cleaned_csv, isFolder=False):
    if isFolder:
        allFiles = glob.glob(cleaned_csv + '*.csv')
        parsed = pd.DataFrame()
        list_ = []
        for file_ in allFiles:
            df = pd.read_csv(file_)
            list_.append(df)
            parsed = pd.concat(list_, ignore_index=True)
    else:
        parsed = pd.read_csv(cleaned_csv)
    fw_count = parsed['fw_count'].tolist()
    fw_count = list(map(lambda fw: literal_eval(fw), fw_count))
    v = DictVectorizer(sparse=False)
    fw_vec = v.fit_transform(fw_count)
    fw_nomed = normalize(fw_vec, axis=0, norm='max')
    fw_df = pd.DataFrame(data = fw_nomed, columns = v.feature_names_)
    expanded_df = pd.concat([parsed, fw_df], axis=1)
    expanded_df.drop('fw_count', axis=1, inplace=True)
    expanded_df.to_csv('cleaned.csv', index=False)

def model(features, labels):

#    feature_norm(file_name)
#    normed = pd.read_csv('cleaned.csv')
#    normed = normed.loc[normed['clerk_school'].notnull()]
#    normed = normed.drop(['judge'], axis=1)
#    normed['clerk_school'] = pd.Categorical(normed.clerk_school).codes
#    y_data = normed['clerk_school'].as_matrix()
#    X_data = normed.drop(['clerk_school'], axis=1).as_matrix()
#
#    uniq_sch = len(np.unique(y_data))
#    print(X_data.shape)
#    feature_num = X_data.shape[1]


    #features = X_data
    #labels = y_data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25)

    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X_train, y_train)
    train_score = logreg.score(X_train, y_train)
    test_score = logreg.score(X_test, y_test)
    print("train score is %f " % train_score)
    print("test score is %f " % test_score)

#    from sklearn.naive_bayes import GaussianNB
#    logreg = GaussianNB()
#    logreg.fit(X_train, y_train)
#    train_score = logreg.score(X_train, y_train)
#    test_score = logreg.score(X_test, y_test)
#    print("train score is %f " % train_score)
#    print("test score is %f " % test_score)
#
#    from sklearn.ensemble import RandomForestClassifier
#    logreg = RandomForestClassifier(n_estimators= 100, random_state=1)
#
#    logreg.fit(X_train, y_train)
#    train_score = logreg.score(X_train, y_train)
#    test_score = logreg.score(X_test, y_test)
#    print("train score is %f " % train_score)
#    print("test score is %f " % test_score)
#
#
#    from sklearn.neural_network import MLPClassifier
#    logreg = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(3, 300), random_state=1)
#
#    logreg.fit(X_train, y_train)
#    train_score = logreg.score(X_train, y_train)
#    test_score = logreg.score(X_test, y_test)
#    print("train score is %f " % train_score)
#    print("test score is %f " % test_score)
    return train_score, test_score

def main():
    features, labels = loadData(file_name)
    print("data loaded ....")
    print("len of feature %d " % len(features))
    print("len of labels %d " % len(labels))
    features = vectorizeFeature(features)
    #print("feature dim %d %d " % (features[0].shape[0], features[0].shape[1]))
    if do_pca == 1:
        features = PCA(features)
        print("PCA is finished")
    train_score, test_score = model(features, labels)
    #print("train score is %f " % train_score)
    #print("test score is %f " % test_score)

print("start training, good luck!...............")
main()
