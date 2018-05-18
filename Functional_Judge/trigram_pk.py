import os
os.chdir("/scratch/cg3015/clerks/output_bigrams")
import pickle
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import decomposition

file_name = "1995_article_judge.csv"
file_name = "18_year.csv"
#file_name = "1996_parsed_trigrams_96.csv"
file_name = "19_year_trigrams.csv"
file_name = "all_functional.csv"
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
        dict[doc[i][0]] = doc[i][1]
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
        label.append(data.loc[i, 'judge'])
    return feature_map, label

def vectorizeFeature(features):
    vec = DictVectorizer()
    transformed_features = vec.fit_transform(features)
    print(type(transformed_features))
    print(transformed_features.nnz)
    print(transformed_features.dtype.itemsize)
    X_scaled = preprocessing.scale(transformed_features.toarray(),with_mean=False)
    return X_scaled

def model(features, labels):
    logreg = linear_model.LogisticRegression(C=1e5)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25)
    #logreg.fit(X_train, y_train)
    #train_score = logreg.score(X_train, y_train)
    #test_score = logreg.score(X_test, y_test)
    #print("train score is %f " % train_score)
    #print("test score is %f " % test_score)

    from sklearn.naive_bayes import GaussianNB
    logreg = GaussianNB()
    logreg.fit(X_train, y_train)
    train_score = logreg.score(X_train, y_train)
    test_score = logreg.score(X_test, y_test)
    print("train score is %f " % train_score)
    print("test score is %f " % test_score)

    from sklearn.ensemble import RandomForestClassifier
    logreg = RandomForestClassifier(n_estimators= 100, random_state=1)

    logreg.fit(X_train, y_train)
    train_score = logreg.score(X_train, y_train)
    test_score = logreg.score(X_test, y_test)
    print("train score is %f " % train_score)
    print("test score is %f " % test_score)


    from sklearn.neural_network import MLPClassifier
    logreg = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(3, 300), random_state=1)

    logreg.fit(X_train, y_train)
    train_score = logreg.score(X_train, y_train)
    test_score = logreg.score(X_test, y_test)
    print("train score is %f " % train_score)
    print("test score is %f " % test_score)

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
