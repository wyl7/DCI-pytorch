import numpy as np
import csv

# Set random seed
seed = 0
np.random.seed(seed)
net = []
# Load data
print("Loading dataset")
# load adjacency matrix and the ground_truth
with open('wikipedia.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    for d in data[1:]:
        net.append([int(d[0]), int(d[1]), int(d[3])])

productID = {}
userID = {}
for key_word in net:
    if key_word[0] not in userID:
        userID[key_word[0]] = len(userID.keys())
    if key_word[1] not in productID:
        productID[key_word[1]] = len(productID.keys())

label = np.zeros(len(userID))#[0 for _ in range(len(userID))]
G = {}
for key_word in net:
    if (userID[key_word[0]], productID[key_word[1]]) not in G.keys():
        G[(userID[key_word[0]], productID[key_word[1]])] = 1
    else:
        G[(userID[key_word[0]], productID[key_word[1]])] += 1
    if key_word[2] == 1:
        label[userID[key_word[0]]] = 1

f = open('wiki.txt','a')
for key in G.keys():
    f.write(str(key[0])+' '+str(key[1])+'\n')
f.close()
print(sum(label))
label = np.reshape(label, (-1, 1)).astype('int')
node_id = np.reshape(np.array(range(len(userID)), dtype='int'), (-1, 1))
label = np.concatenate([node_id, label], 1)
np.save('wiki_label.npy', label)
