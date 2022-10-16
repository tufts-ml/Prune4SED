This repo includes the implementation of TMLR submission ["Towards Accurate Subgraph Similarity Computation via Neural Graph Pruning"](https://openreview.net/forum?id=CfzIsWWBlo).

# Requirements
The model is tested on a Linux server (32 cores and a A100 GPU) with the following packages,
* Python (3.8.13)
* pytorch (1.12.1)
* torch-geometric (2.1.0)

# Run examples
To download the full datasets reported in Table 2, we refer to [NeuroSED repo](https://github.com/idea-iitd/NeuroSED). In this repo, we share an example to evaluate our model on CiteeSeer dataset.  The data can be found under `data/CiteSeer/`, which contains a small set of graph pairs that are derived from CiteSeer. Our pretrained model is under `saved_model/`.

The file `experiment.py` provides both train and predict procedures. Below is an example:
Train procedure:
```
$ python experiment.py --dataset_name CiteSeer --input_dim 6
```
Predict procedure using the pretrained model:
```
$ python experiment.py --dataset_name CiteSeer --input_dim 6 --test
```