# DCGCN
Code for the TACL paper [Densely Connected Graph Convolutional Network for Graph-to-Sequence Learning](http://www.statnlp.org/wp-content/uploads/2019/03/DCGCN.pdf)

# Citation
```
@article{dcgcnforgraph2seq19guo, 
title={Densely Connected Graph Convolutional Networks for Graph-to-Sequence Learning}, 
author={Zhijiang Guo and Yan Zhang and Zhiyang Teng and Wei Lu}, 
journal={Transactions of the Association of Computational Linguistics}, 
year={2019}, 
}
```

## Dependencies
The model requires:
- **Python3**
- [MXNet 1.3.0](https://github.com/apache/incubator-mxnet/tree/1.3.0)
- numpy
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

## Training

The remaining new options are related to the new encoder, a Dense Graph Convolutional Network, which was abbreviated it as `gcn` in the API. Here is a list of options to train our model:

`--source`: path to the file of the source sentences of the training set.

`--target`: path to the file of the target sentences of the training set.

`--source-graphs`: path to the graph information file of the training set.

`--validation-source`: path to the file of the source sentences of the validation set.

`--validation-target`: path to the file of the target sentences of the validation set.

`--val-source-graphs`: path to the graph information file of the validation set.

`--edge-vocab`: path to the file of edge vocabulary.

`--batch-size`: for AMR generation task, we usually set it to 16. For NMT task, we usually set it to 24.

`--batch-type`: sentence.

`--word-min-count`: for AMR generation task, we usually set it to 2:2. For NMT task, it should be 3:3. This is the minimum frequency of words to be included in vocabularies.

`--num-embed`: for both tasks, we set it to 360:360. It is the embedding size for source and target tokens.

`--gcn-pos-embed`: for both tasks, we set it to 300. It is the dimensionality of positional embeddings. We concat it with the word embeddings.

`--embed-dropout`: for both tasks, we set it to 0.5:0.5. It is the dropout probability for source and target embeddings.

`--shared-vocab`: for both task, we share the source and target vocabulary.

`--max-seq-len`: for AMR generation task, we set it to 199:199 (the Sockeye framework needs to take BOS and EOS symbols into consideration, the totol length should be 200:200). It is the maximum sequence length in terms of number of tokens.

`--encoder`: we use gcn here, which is our densely connected graph convolutional network.

`--gcn-activation`: type activation to use for each graph convolutional layer. For both tasks, we use relu.

`--gcn-num-hidden`: for both tasks, we set it to 360. It is the number of DCGCN hidden units.

`--gcn-dropout`: for both tasks, we set it to 0.1. Dropout rate on the DCGCN output vectors.

`--gcn-adj-norm`: this flag enables normalisation when updating each node hidden state.

`--num-layers`: for AMR generation task, we set it to 4:1, which means we stacked four densely connected graph convolutional layers for the encoder side and one LSTM layer for the decoder side. For NMT task, we set it to 3:2.

`--decoder`: we used LSTM for both tasks.

`--rnn-attention-type`: we use coverage attention for both tasks. 

`--rnn-num-hidden`: we set it to 300 for both tasks.

`--rnn-decoder-hidden-dropout`: we set it to 0.2 for both tasks.

`--weight-tying`: turn on weight tying for both tasks.

`--weight-tying-type`: we used src_trg here.

`--weight-init-xavier-factor-type`: we used in as the default type.

`--weight-init-scale`: weight initialization scale for xavier initialization. We set it to 2.34.

`--initial-learning-rate`: for AMR generation task, we set it to 0.0003.

`--learning-rate-reduce-factor`: we use plateau-reduce as the default learning rate scheduler. Here we set the factor to multiply learning rate with to be 0.7.

`--learning-rate-reduce-num-not-improved`: for plateau-reduce learning rate scheduler. Adjust learning rate if perplexity did not improve for 5 checkpoints.

`--checkpoint-frequency`: checkpoint and evaluate every 1000 updates/batches for AMR generation, 1500 for NMT task.

`--max-num-checkpoint-not-improved`: maximum number of checkpoints the model is allowed to not improve in perplexity on validation data before training is stopped. Here we set it to 28.

`--decode-and-evaluate`: decode certain number of (-1 means all) sampled sentences from validation data and compute evaluation metrics.

`--output`: file to write parameters.

`--overwrite-output`: delete all contents of the model directory if it already exists.

`--device-ids`: List or number of GPUs ids to use.

`--keep-last-params`: keep only the last 30 params files.

`--fixed-param-names`: we fix the embedding while finetuning the model on the gold training data (for external dataset pretraining).


## Decoding

Here is a list of options to use the saved model to decode:

`-m`: path to the folder of the saved model.

`--edge-vocab`: path to the file of edge vocabulary.

`-o`: path to the file of the input sentences.

`--beam-size`: size of the beam search, we use 10 for both task.

`--checkpoints`: you can select a specific checkpoint for decoding. Otherwise, the system will choose the best checkpoint (in terms of perplexity on validation set) for decoding.


