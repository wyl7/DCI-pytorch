import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans

def load_data_cluster(datasets, num_cluster, num_folds):
    # load the adjacency
    adj = np.loadtxt('./data/'+datasets+'.txt')
    adj = adj[:, 0: 2]
    num_user = len(set(adj[:, 0]))
    num_object = len(set(adj[:, 1]))
    adj[:, 1] += num_user
    adj = adj.astype('int')
    edge_index = adj.T
    print('Load the edge_index down!')
    
    # load the user label
    label = np.load('./data/'+datasets+'_label.npy')
    y = label[:, 1]
    print('Ratio of fraudsters: ', np.sum(y) / len(y))
    print('Number of edges: ', edge_index.shape[1])
    print('Number of users: ', num_user)
    print('Number of objects: ', num_object)

    # split the train_set and validation_set
    split_idx = []
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=0)
    for (train_idx, test_idx) in skf.split(y, y):
        split_idx.append((train_idx, test_idx))
   
    # load initial features
    feats = np.load('../visual/data/features_adj/'+datasets+'_feature64.npy')

    # pre-clustering
    kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(feats)
    ss_label = kmeans.labels_
    loc = np.zeros((feats.shape[0], num_cluster))
    for i in range(feats.shape[0]):
        loc[i, ss_label[i]] = 1
    print("Number of clusters: ", loc.shape)

    return edge_index, feats, split_idx, label, num_user, num_object, loc

