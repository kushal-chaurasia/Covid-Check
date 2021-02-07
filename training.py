import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression


def data_split(data,ratio):
    np.random.seed(42)
    shuffeled= np.random.permutation(len(data))
    test_set_size= int(len(data)*ratio)
    test_indices=shuffeled[:test_set_size]
    train_indices=shuffeled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

if __name__ == '__main__':
    
    df = pd.read_csv('data.csv')
    train, test = data_split(df,0.2)
    X_train= train[['fever','bodypain','age','RunnyNose','diffbreathe']].to_numpy()
    X_test= test[['fever','bodypain','age','RunnyNose','diffbreathe']].to_numpy()
    Y_train= train[['infection']].to_numpy().reshape(2060,)
    Y_test= test[['infection']].to_numpy().reshape(515,)
    clf = LogisticRegression()
    clf.fit(X_train, Y_train)

    file = ('model.pkl', 'wb')
    pickle.dump(clf, file)

    inputFeature= [100,1,22,-1,1]
    infProb=clf.predict_proba([inputFeature])[0][1]