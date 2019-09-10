# Densely Connected Graph Convolutional Networks for Graph-to-Sequence Learning

This paper/code introduces the Densely Connected Graph Convolutional Networks (DCGCNs) for the graph-to-sequence learning task. We evaluate our model on two tasks including **AMR-to-Text Generation** (AMR2015 and AMR2017) and **Syntax-Based Machine Translation** (EN2DE and EN2CS).

You can find the latest version of the TACL paper [here](http://www.statnlp.org/wp-content/uploads/2019/03/DCGCN.pdf).

This paper is presented in ACL 2019, you can find the video [here](http://www.acl2019.org/EN/program.xhtml) in Session 6F: Machine Learning 4. Slides are also available (under the images directory) for brief introduction of this work.

See below for an overview of the encoder (DCGCNs) architecture: Each block **has two sub-blocks**. Both of them are densely connected graph convolutional layers with different numbers (**n** & **m**) of layers. For these two tasks we use **n=6** and **m=3**. These are hyper-parameters. 

![Densely Connected Graph Convolutional Layers](images/encoder.png "Densely connected layers")


## Dependencies
The model requires:
- Python3
- [MXNet 1.3.0](https://github.com/apache/incubator-mxnet/tree/1.3.0)
- [Sockeye 1.18.56 (NMT framework based on MXNet)](https://github.com/awslabs/sockeye)
- CUDA (we tested on CUDA 9.2)

## Installation
#### GPU

If you want to run sockeye on a GPU you need to make sure your version of Apache MXNet
Incubating contains the GPU bindings. Depending on your version of CUDA you can do this by running the following:

```bash
> pip install -r requirements/requirements.gpu-cu${CUDA_VERSION}.txt
> pip install .
```
where `${CUDA_VERSION}` can be `75` (7.5), `80` (8.0), `90` (9.0), `91` (9.1), or `92` (9.2).

## Preprocessing

We need to convert the dataset into extended Levi graphs for training. For details please refer to the paper.

Here, for AMR-to-text, get the AMR Sembank (LDC2017T10) first and put the folder called abstract_meaning_representation_amr_2.0 inside the data folder. Then run:

```
./gen_amr.sh
```

For NMT, you can download the raw dataset from [here](https://drive.google.com/drive/folders/0BxGk3yrG1HHVMy1aYTNld3BIN2s) first and change the data folder inside nmt_preprocess.py. Then run:

```
python nmt_preprocess.py
```

Or you can download our preprocessed dataset (en2cs and en2de) for DCGCN model from [here](https://drive.google.com/drive/folders/1QTWRTnQjDnnREeS1DCxMg46yTadl0A-e?usp=sharing). For AMR corpus, it has LDC license so we cannot distribute the preprocessed data. If you have the license, feel free to drop us an email to get the preprocessed data.

## Training

To train the DCGCN model, run (here we use AMR2015 as an example):

```
./train_amr15.sh
```

Model checkpoints and logs will be saved to `./sockeye/amr2015_model`.

## Decoding

When we finish the training, we can use the trained model to decode on the test set, run:

```
./decode_amr15.sh
```

This will use the last checkpoint (84th for AMR2015) by default. Use `--checkpoints` to specify a model checkpoint file.

## Postprocessing

For AMR-to-Text generation, we also use the scope markers as in [Konstas et al. (2017)](https://arxiv.org/pdf/1704.08381.pdf) and [Beck et al. (2018)](https://arxiv.org/pdf/1806.09835.pdf). Basically, they conduct named entity anonymization and named entity clustering in the preprocessing stage.
In the postprocessing state, we need to substitute the anonymized entities, run:

```
./postprocess_amr.sh
```

For Syntax-Based Machine Translation, we use BPE in the decoder side. In the postprocessing stage, we need to merge them into natural language sequence for evaluation, run:

```
./merge.sh
```


## Evaluation

For BLEU score evaluation, run:

```
./eval_bleu.sh
```

## Citation
```
@article{guo-etal-2019-densely,
    title = "Densely Connected Graph Convolutional Networks for Graph-to-Sequence Learning",
    author = "Guo, Zhijiang and Zhang, Yan and Teng, Zhiyang and Lu, Wei",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "7",
    month = mar,
    year = "2019",
    url = "https://www.aclweb.org/anthology/Q19-1019",
    pages = "297--312"
}
```

## Pretrained Models

For pretrained models, please download the models [here](https://drive.google.com/drive/folders/1QTWRTnQjDnnREeS1DCxMg46yTadl0A-e?usp=sharing)

After the download, unzip the file and put it under the sockeye directory.

For the AMR2015 dataset, the pretrained model achieves 25.9 BLEU score, while for the AMR2017 dataset, it achieves 27.9 BLEU score. You can train the model by yourself, the hyperparameters are given. The results should be the same.

## Related Repo

This repo is built based on [Graph-to-Sequence Learning using Gated Graph Neural Networks](https://github.com/beckdaniel/acl2018_graph2seq).
DCGCNs can also be applied on other NLP tasks. For example, relation extraction: [Attention Guided Graph Convolutional Networks for Relation Extraction](https://github.com/Cartus/AGGCN_TACRED).


## Results

We also release the output of our model. Please refer to the **results** directory.

