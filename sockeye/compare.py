import mxnet as mx

import sockeye.constants as C
from sockeye import utils
from sockeye.config import Config


import logging
logger = logging.getLogger(__name__)


def get_gcn(config, prefix):
    gcn = GCNCell(input_dim=config.input_dim,
                  output_dim=config.output_dim,
                  directions=config.directions,
                  num_layers=config.num_layers,
                  adj_norm=config.adj_norm,
                  dropout=config.dropout,
                  prefix=prefix)
    return gcn


class GCNConfig(Config):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 directions: int,
                 num_layers: int,
                 adj_norm: bool = True,
                 activation: str = 'relu',
                 dropout: float = 0.0,
                 dtype: str = C.DTYPE_FP32) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.directions = directions
        self.num_layers = num_layers
        self.adj_norm = adj_norm
        self.activation = activation
        self.dropout = dropout
        self.dtype = dtype


class GCNCell(object):
    def __init__(self,
                 input_dim,
                 output_dim,
                 directions,
                 num_layers,
                 adj_norm=True,
                 prefix='gcn_',
                 activation='relu',
                 dropout=0.0):

        self._input_dim = input_dim
        self._output_dim = output_dim
        self._directions = directions
        self._num_layers = 4
        self._layers = []
        self._prefix = prefix

        self._activation = activation
        self._dropout = dropout
        self._norm = adj_norm

        self.reset()
        self._modified = False
        self._own_params = True

        if self._input_dim != self._output_dim:
            self._input_W = mx.symbol.Variable(self._prefix + '_input_weight',
                                               shape=(input_dim, output_dim))
            self._input_b = mx.symbol.Variable(self._prefix + '_input_bias',
                                               shape=(output_dim,))

        for i in range(self._num_layers):
            self._layers.append(GraphConvolution(prefix="%s%d_6_" % (self._prefix, i),
                                                 heads=6,
                                                 output_dim=self._output_dim,
                                                 directions=self._directions,
                                                 dropout=self._dropout,
                                                 norm=self._norm,
                                                 activation=self._activation))
            self._layers.append(GraphConvolution(prefix="%s%d_3_" % (self._prefix, i),
                                                 heads=3,
                                                 output_dim=self._output_dim,
                                                 directions=self._directions,
                                                 dropout=self._dropout,
                                                 norm=self._norm,
                                                 activation=self._activation))

        # Layer Aggregation Params
        self._aggregate_W = mx.symbol.Variable(self._prefix + '_aggregate_weight',
                                               shape=(self._num_layers * 2 * self._output_dim, self._output_dim))
        self._aggregate_b = mx.symbol.Variable(self._prefix + '_aggregate_bias',
                                               shape=(self._output_dim,))

    def convolve(self, adj, inputs, seq_len):
        layer_list = []

        if self._input_dim != self._output_dim:
            inputs = mx.sym.dot(inputs, self._input_W)
            inputs = mx.sym.broadcast_add(inputs, self._input_b)

        outputs = inputs
        for i in range(len(self._layers)):
            outputs = self._layers[i](adj=adj, inputs=outputs)
            layer_list.append(outputs)

        aggregate_output = mx.sym.concat(*layer_list, dim=2)
        aggregate_output = mx.sym.dot(aggregate_output, self._aggregate_W)
        aggregate_output = mx.sym.broadcast_add(aggregate_output, self._aggregate_b)

        return aggregate_output

    def reset(self):
        pass


class GraphConvolution:

    def __init__(self,
                 prefix: str,
                 heads: int,
                 output_dim: int,
                 directions: int,
                 dropout: float,
                 norm: bool,
                 activation: str = 'relu'):

        self._prefix = prefix
        self._heads = heads
        self._output_dim = output_dim
        self._directions = directions
        utils.check_condition(output_dim % heads == 0,
                              "Number of heads (%d) must divide attention depth (%d)" % (heads, output_dim))
        self._hidden_dim = self._output_dim // self._heads
        self._dropout = dropout
        self._norm = norm
        self._activation = activation
        self._weight_list = []
        self._bias_list = []

        # Graph Convolution Params
        for i in range(heads):
            self._weight_list.append([mx.symbol.Variable(self._prefix + "_dense_" + str(i) + "_" + str(j) + "_weight",
                                                         shape=(self._output_dim + self._hidden_dim * i, self._hidden_dim))
                                      for j in range(self._directions)])
            self._bias_list.append([mx.symbol.Variable(self._prefix + "_dense_" + str(i) + "_" + str(j) + "_bias",
                                                       shape=(self._hidden_dim,))
                                    for j in range(self._directions)])

        # Attention Params
        self._att_1_W = [mx.symbol.Variable(self._prefix + str(i) + '_att_1_weight',
                                            shape=(self._hidden_dim, 1))
                         for i in range(self._heads * self._directions)]
        self._att_2_W = [mx.symbol.Variable(self._prefix + str(i) + '_att_2_weight',
                                            shape=(self._hidden_dim, 1))
                         for i in range(self._heads * self._directions)]

        # Direction Params
        self._direct_W = [mx.symbol.Variable(self._prefix + str(i) + '_direct_weight',
                                             shape=(self._directions * self._hidden_dim, self._hidden_dim))
                          for i in range(self._heads)]
        self._direct_b = [mx.symbol.Variable(self._prefix + str(i) + '_direct_bias',
                                             shape=(self._hidden_dim,))
                          for i in range(self._heads)]

        # Linear Transform Params
        self._linear_W = mx.symbol.Variable(self._prefix + '_linear_weight',
                                            shape=(self._output_dim, self._output_dim))
        self._linear_b = mx.symbol.Variable(self._prefix + '_linear_bias',
                                            shape=(self._output_dim,))

    def __call__(self, adj, inputs):
        outputs = inputs
        cache_list = [outputs]
        output_list = []
        for i in range(self._heads):
            convolved = self._convolve(adj, outputs, i)
            cache_list.append(convolved)
            outputs = mx.sym.concat(*cache_list, dim=2)
            output_list.append(convolved)

        outputs = mx.sym.concat(*output_list, dim=2)
        if self._dropout != 0.0:
            outputs = mx.sym.Dropout(outputs, p=self._dropout)

        outputs = mx.sym.broadcast_add(outputs, inputs)
        outputs = mx.sym.dot(outputs, self._linear_W)
        outputs = mx.sym.broadcast_add(outputs, self._linear_b)

        return outputs

    def _convolve(self, adj, inputs, i):
        direct_list = []
        for j in range(self._directions):
            k = i * self._directions + j

            weight = self._weight_list[i][j]
            bias = self._bias_list[i][j]

            output = mx.sym.dot(inputs, weight)
            output = mx.sym.broadcast_add(output, bias)

            a1 = self._att_1_W[k]
            a2 = self._att_2_W[k]

            f_1 = mx.sym.dot(output, a1)
            f_2 = mx.sym.dot(output, a2)

            f_2 = mx.sym.transpose(f_2, axes=(0, 2, 1))
            f = mx.sym.broadcast_add(f_1, f_2)

            e = mx.sym.LeakyReLU(f)

            label_id = j + 1
            mask = mx.sym.ones_like(adj) * label_id
            adji = (mask == adj)

            zero_vec = mx.sym.ones_like(e) * (-9e15)

            attention = mx.sym.where(adji > 0, e, zero_vec)
            attention = mx.sym.softmax(attention, axis=-1)

            output = mx.sym.batch_dot(attention, output)
            direct_list.append(output)

        outputs = mx.sym.concat(*direct_list, dim=2)
        direct_W = self._direct_W[i]
        direct_b = self._direct_b[i]

        outputs = mx.sym.dot(outputs, direct_W)
        outputs = mx.sym.broadcast_add(outputs, direct_b)

        if self._norm:
            norm_adj = mx.sym.broadcast_not_equal(adj, mx.sym.zeros_like(adj))
            norm_factor = mx.sym.sum(norm_adj, axis=2, keepdims=True)
            outputs = mx.sym.broadcast_div(outputs, norm_factor)

        final_output = mx.sym.Activation(outputs, act_type=self._activation)
        return final_output
