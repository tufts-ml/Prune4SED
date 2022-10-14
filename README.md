This repo includes the implementation of TMLR submission **"Towards Accurate Subgraph Similarity Computation via Neural Graph Pruning"**

# Requirements
The model is tested on a Linux server (32 cores and a A100 GPU) with the following packages,
* pytorch (1.9.1)
* torch-geometric (2.0.2)
* scipy (1.7.1)

# Run examples
We provide both data and pretrained models on three datasets (AIDS, CiteSeer, and Protein) at https://anonymous.4open.science/r/Prune4SED-EF60. We refer to NeuroSED repo (https://github.com/idea-iitd/NeuroSED) for the rest datasets. Before running examples, data and pretrained models (optional) should be downloaded to the current folder `./`.

The file `experiment.py` provides both training and prediction procedures. Below is an example of running on AIDS dataset.
Training (The training requires creating a folder `./runlogs/dataset_name/` to save training logs):
```
$ python experiment.py --dataset_name AIDS --input_dim 38
```
Prediction using pretrained model:
```
$ python experiment.py --dataset_name AIDS --input_dim 38 --test
```
Training and prediction on CiteSeer and Protein datasets follow the same protocol, with `dataset_name` as `CiteSeer/Protein` and `input_dim` as `6/3`, respectively.

