# DCI-pytorch
The pytorch implementation of decoupling representation learning and classification for GNN-based anomaly detection. We run on a DELL server with 2 * Intel(R) Xeon(R) Silver 4210, 4 * NVIDIA TITAN V (12G), 10 * 32GB DDR4 RAM and 1 * 8TB hard disk.
See our [paper](https://xiaojingzi.github.io/publications/SIGIR21-Wang-et-al-decoupled-GNN.pdf) for details on the algorithm.

Abstract
====
GNN-based anomaly detection has recently attracted considerable attention. 
Existing attempts have thus far focused on jointly learning the node representations and the classifier for detecting the anomalies.
Inspired by the recent advances of self-supervised learning (SSL) on graphs, we explore another possibility of decoupling the node representation learning and the classification for anomaly detection. 
Decoupled training can alleviate the negative effects caused by the inconsistency between user’s behavior patterns and their label semantics. 
The proposed SSL scheme, called Deep cluster Infomax (DCI), can contribute to the decoupled training. 
In effect, the idea of decoupled training is not restricted to the anomaly detection.

<img src="https://github.com/wyl7/DCI-pytorch/blob/main/jointVsDecoupled.PNG" with="10">

Requirements
====
You can create a virtual environment first via:
```
conda create -n your_env_name python=3.8.5
```

You can install all the required tools using the following command:
```
# CUDA 10.2
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch
$ pip install -r requirements.txt
```

Overview
====
Here we provide the implementation of different training schemes (i.e., joint training and decoupled training) in PyTorch, 
along with an execution example (on the Wiki dataset). For the decoupled training, we provide different instantiations of the SSL loss function, 
including DGI and DCI. Specifically, the repository is organized as follows:
* `data/` contains the files for Wiki dataset. `dataName.txt` stores the edges in the graph. Format is: node_id, node_id.
`dataName_label.txt` stores the user labels. Format is: node_id, label, where the user label is a binary value which takes value 1 if the user is abnormal and 0 otherwise.
We have processed the node_id, so that the top node_ids all correspond to the users.

* `features/` contains the initial node features. You can run:
```$python init_feats.py dataName```
to generate the initial node features. 
Specially, we perform eigen-decomposition on the normalized adjacency matrix for the feature initialization.
For the large-scale graphs, you can also adopt other algorithms, such as [DeepWalk](https://github.com/phanein/deepwalk)[1] and [Node2Vec](https://github.com/aditya-grover/node2vec)[2], to generate the initial node features.

* `models/` contains the implementation of the DGI loss (`dgi.py`), DCI loss (`dci.py`) and the binary classifier (`clf_model.py`).

* `layers/` contains the implementation of the GIN layer (`graphcnn.py`), the MLP layer (`mlp.py`), the averaging readout (`readout.py`), and the bi-linear discriminator (`discriminator.py`). `readout.py` and `discriminator.py` are copied from the source code of [Deep Graph Infomax](https://github.com/PetarV-/DGI)[4]. `mlp.py` is copied from the source code of [GIN](https://github.com/weihua916/powerful-gnns)[3]. `graphcnn.py` is revised based on the corresponding implemention in [GIN](https://github.com/weihua916/powerful-gnns).

* `util.py` is used for loading and pre-processing the dataset.

Running the code
====
To run the joint training scheme, execute:
```
$ python main_dci.py --training_scheme joint --dataset wiki
```

To run the decoupled training scheme with DGI, execute:
```
$ python main_dgi.py --dataset wiki
```

To run the decoupled training scheme with DCI, execute:
```
$ python main_dci.py --dataset wiki --training_scheme decoupled --num_cluster <number of clusters>
```

Notes: the optimal \<number of clusters\> could be somewhat different under different environments (e.g., different versions of PyTorch), you can use the suggested method introduced in our paper to determine a proper \<number of clusters\> for your dataset. Besides DGI and DCI, you can try other graph SSL algorithms. Even though the SSL objective does not rely on the task-specific label information, it should be related to your classification task.

Reference
====
[1] Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. 2014. DeepWalk: online learning of social representations. In KDD. 701–710.

[2] Aditya Grover and Jure Leskovec. 2016. node2vec: Scalable Feature Learning for Networks. In KDD. 855–864.

[3] Petar Velickovic, William Fedus, William L. Hamilton, Pietro Liò, Yoshua Bengio, and R. Devon Hjelm. 2019. Deep Graph Infomax. In ICLR.

[4] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. 2019. How Powerful are Graph Neural Networks?. In ICLR.
