from sklearn.cluster import DBSCAN

def make_cluster_indices(X, thr=10):
    db = DBSCAN(eps=thr, min_samples=1, metric='euclidean')
    y_db = db.fit_predict(X)
    return y_db

if __name__ == "__main__":
    pass