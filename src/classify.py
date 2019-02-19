import os
import StringIO
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC, LinearSVC

def read_embeddings(fname):
    with open(fname) as f:
        f.readline()
        content = StringIO.StringIO(f.read())
        names = []
        def fn(name):
            names.append(name)
            return 0
        data = np.loadtxt(content, converters={0: fn})[:, 1:]
        return names, data
    
def read_labels(basedir):
    classes = os.listdir(basedir)
    graph2label = {}
    for clss in classes:
        path = os.path.join(basedir, clss)
        graphs = os.listdir(path)
        for graph in graphs:
            graph2label[graph.split('.')[0]] = clss
    return graph2label

def organize_data(graph2label, embed_vnames, embed_vecs):
    N = len(graph2label)
    data = [0 for _ in range(N)]
    labels = [0 for _ in range(N)]
    for i in range(len(embed_vnames)):
        if "dt" in embed_vnames[i]:
            _, gname = embed_vnames[i].split("_")
            if gname == '0': continue
            data[int(gname)-1] = embed_vecs[i]
            labels[int(gname)-1] = int(graph2label[gname])
    return np.array(data), np.array(labels)

def load_data(folder, emb_fname):
    label_dir = "{f}/graphs".format(f=folder)
    names, data = read_embeddings(emb_fname)
    g2l = read_labels(label_dir)
    X, y = organize_data(g2l, names, data)
    return X, y

def learn(folder, emb):
    X, y = load_data(folder, emb)
    print("Dataset {}".format(folder))
    print("Shapes {} {}".format(X.shape, y.shape))

    params = {'C':[0.01,0.1,1,10,100,1000]}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='f1')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print("Score {}".format(score))
    print("Best params {}".format(classifier.best_params_))
    print(classification_report(y_test, y_pred))
    return score, classifier

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run classifier")
    parser.add_argument("dataset", help="The dataset folder to use")
    parser.add_argument("embedding", help="Which embedding file to use")
    args = parser.parse_args()

    score, classifier = learn(args.dataset, args.embedding)
