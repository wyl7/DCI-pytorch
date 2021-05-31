import numpy as np
import scipy
import sklearn.preprocessing as preprocessing
from scipy.sparse import linalg
import scipy.sparse as sp
import sys

np.random.seed(0)

def eigen_decomposision(n, k, laplacian, hidden_size, retry):
    if k <= 0:
        return torch.zeros(n, hidden_size)
    laplacian = laplacian.astype("float64")
    #print(type(laplacian))
    ncv = min(n, max(2 * k + 1, 20))
    v0 = np.random.rand(n).astype("float64")
    for i in range(retry):
        try:
            s, u = linalg.eigsh(laplacian, k=k, which="LA", ncv=ncv, v0=v0)
           
        except sparse.linalg.eigen.arpack.ArpackError:
            # print("arpack error, retry=", i)
            ncv = min(ncv * 2, n)
            if i + 1 == retry:
                sparse.save_npz("arpack_error_sparse_matrix.npz", laplacian)
                u = torch.zeros(n, k)
        else:
            break
    x = preprocessing.normalize(u, norm="l2")
    x = x.astype("float64")
    return x

def intial_embedding(n, adj, in_degree,hidden_size, retry=10):
    in_degree = in_degree.clip(1) ** -0.5
    norm = sp.diags(in_degree, 0, dtype=float)
    laplacian = norm * adj * norm
    k = min(n - 2, hidden_size)
    x = eigen_decomposision(n, k, laplacian, hidden_size, retry)
    
    return x

def process_adj(dataSetName):
    f = open(dataSetName)
    userId = []
    itemId = []
    for line in f.readlines():
        curLine = line.strip().split(" ")
        userId.append(int(curLine[0]))
        itemId.append(int(curLine[1]))
    f.close()
    itemId= list(map(lambda x:x+len(list(set(userId))),itemId))
    user_num = len(list(set(userId)))
    item_num = len(list(set(itemId)))
    node_num = user_num + item_num
    adj = np.zeros([node_num,node_num], dtype = np.float64)
    i = 0
    while i<len(userId):
        adj[userId[i],itemId[i]] =1
        adj[itemId[i],userId[i]] =1
        i = i+1
    adj = sp.csr_matrix(adj)
    return adj,user_num,item_num

datasetName = sys.argv[1]
adj, user_num, item_num = process_adj('../data/'+datasetName+'.txt')
n = user_num +item_num
hidden_size = 64
i = 0
in_degree = []
while i < n:
    in_degree.append(np.sum(adj.data[adj.indptr[i]:adj.indptr[i+1]]))
    i = i+1
in_degree = np.array(in_degree)
x = intial_embedding(n, adj, in_degree, hidden_size, retry=10)
np.save(datasetName+'_feature64.npy', x)
