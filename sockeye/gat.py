import logging

import mxnet as mx
import sockeye.constants as C
from sockeye.config import Config

logger = logging.getLogger(__name__)


class GATConfig(Config):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 tensor_dim: int,
                 activation: str = 'relu',
                 dropout: float = 0.0,
                 dtype: str = C.DTYPE_FP32) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tensor_dim = tensor_dim
        self.activation = activation
        self.dropout = dropout
        self.dtype = dtype


def get_gat(config, prefix):
    gat = GATCell(config.input_dim,
                  config.output_dim,
                  config.tensor_dim,
                  dropout=config.dropout,
                  prefix=prefix)
    return gat


class GATCell(object):
    """GCN cell
    """
    def __init__(self, input_dim, output_dim, tensor_dim,
                 prefix='gcn_', params=None,
                 activation='relu',
                 dropout=0.0):
        self._own_params = True
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._tensor_dim = tensor_dim
        self._prefix = prefix
        self._params = params
        self._modified = False
        self.reset()
        self._activation = activation
        self._dropout = dropout
        self.nheads = 4
        self._norm = False
        if self._input_dim != self._output_dim:
            self._first_W = mx.symbol.Variable(self._prefix + '_first_weight',
                                               shape=(input_dim, output_dim))
            self._first_b = mx.symbol.Variable(self._prefix + '_first_bias',
                                               shape=(output_dim,))
        # the first GAT layer
        self._W = [mx.symbol.Variable(self._prefix + str(i) + '_weight',
                                      shape=(output_dim, output_dim))
                                      for i in range(self.nheads*self._tensor_dim)]
        self._b = [mx.symbol.Variable(self._prefix + str(i) + '_bias',
                                      shape=(output_dim,))
                                      for i in range(self.nheads*self._tensor_dim)]

        # Attention parameters
        self._att_1_W = [mx.symbol.Variable(self._prefix + str(i) + '_att_1_weight',
                                           shape=(output_dim, 1))
                                           for i in range(self.nheads*self._tensor_dim)]
        self._att_2_W = [mx.symbol.Variable(self._prefix + str(i) + '_att_2_weight',
                                           shape=(output_dim, 1))
                                           for i in range(self.nheads*self._tensor_dim)]

        # the second GAT layer
        self._2_W = [mx.symbol.Variable(self._prefix + str(i) + '_2_weight',
                                      shape=(output_dim, output_dim))
                                      for i in range(self._tensor_dim)]
        self._2_b = [mx.symbol.Variable(self._prefix + str(i) + '_2_bias',
                                      shape=(output_dim,))
                                      for i in range(self._tensor_dim)]

        # Attention parameters
        self._2_att_1_W = [mx.symbol.Variable(self._prefix + str(i) + '_2_att_1_weight',
                                           shape=(output_dim, 1))
                                           for i in range(self._tensor_dim)]
        self._2_att_2_W = [mx.symbol.Variable(self._prefix + str(i) + '_2_att_2_weight',
                                           shape=(output_dim, 1))
                                           for i in range(self._tensor_dim)]

        # the third GAT layer
        self._3_W = [mx.symbol.Variable(self._prefix + str(i) + '_3_weight',
                                      shape=(output_dim, output_dim))
                                      for i in range(self.nheads*self._tensor_dim)]
        self._3_b = [mx.symbol.Variable(self._prefix + str(i) + '_3_bias',
                                      shape=(output_dim,))
                                      for i in range(self.nheads*self._tensor_dim)]

        # Attention parameters
        self._3_att_1_W = [mx.symbol.Variable(self._prefix + str(i) + '_3_att_1_weight',
                                           shape=(output_dim, 1))
                                           for i in range(self.nheads*self._tensor_dim)]
        self._3_att_2_W = [mx.symbol.Variable(self._prefix + str(i) + '_3_att_2_weight',
                                           shape=(output_dim, 1))
                                           for i in range(self.nheads*self._tensor_dim)]

        # the fourth GAT layer
        self._4_W = [mx.symbol.Variable(self._prefix + str(i) + '_4_weight',
                                      shape=(output_dim, output_dim))
                                      for i in range(self._tensor_dim)]
        self._4_b = [mx.symbol.Variable(self._prefix + str(i) + '_4_bias',
                                      shape=(output_dim,))
                                      for i in range(self._tensor_dim)]

        # Attention parameters
        self._4_att_1_W = [mx.symbol.Variable(self._prefix + str(i) + '_4_att_1_weight',
                                           shape=(output_dim, 1))
                                           for i in range(self._tensor_dim)]
        self._4_att_2_W = [mx.symbol.Variable(self._prefix + str(i) + '_4_att_2_weight',
                                           shape=(output_dim, 1))
                                           for i in range(self._tensor_dim)]

        # the fifth GAT layer
        self._5_W = [mx.symbol.Variable(self._prefix + str(i) + '_5_weight',
                                      shape=(output_dim, output_dim))
                                      for i in range(self.nheads*self._tensor_dim)]
        self._5_b = [mx.symbol.Variable(self._prefix + str(i) + '_5_bias',
                                      shape=(output_dim,))
                                      for i in range(self.nheads*self._tensor_dim)]

        # Attention parameters
        self._5_att_1_W = [mx.symbol.Variable(self._prefix + str(i) + '_5_att_1_weight',
                                           shape=(output_dim, 1))
                                           for i in range(self.nheads*self._tensor_dim)]
        self._5_att_2_W = [mx.symbol.Variable(self._prefix + str(i) + '_5_att_2_weight',
                                           shape=(output_dim, 1))
                                           for i in range(self.nheads*self._tensor_dim)]

        # the sixth GAT layer
        self._6_W = [mx.symbol.Variable(self._prefix + str(i) + '_6_weight',
                                      shape=(output_dim, output_dim))
                                      for i in range(self._tensor_dim)]
        self._6_b = [mx.symbol.Variable(self._prefix + str(i) + '_6_bias',
                                      shape=(output_dim,))
                                      for i in range(self._tensor_dim)]

        # Attention parameters
        self._6_att_1_W = [mx.symbol.Variable(self._prefix + str(i) + '_6_att_1_weight',
                                           shape=(output_dim, 1))
                                           for i in range(self._tensor_dim)]
        self._6_att_2_W = [mx.symbol.Variable(self._prefix + str(i) + '_6_att_2_weight',
                                           shape=(output_dim, 1))
                                           for i in range(self._tensor_dim)]

        self._second_W = mx.symbol.Variable(self._prefix + '_second_weight',
                                      shape=(self.nheads*output_dim, output_dim))
        self._second_b = mx.symbol.Variable(self._prefix + '_second_bias',
                                      shape=(output_dim,))

        self._third_W = mx.symbol.Variable(self._prefix + '_third_weight',
                                      shape=(self.nheads*output_dim, output_dim))
        self._third_b = mx.symbol.Variable(self._prefix + '_third_bias',
                                      shape=(output_dim,))

        self._fourth_W = mx.symbol.Variable(self._prefix + '_fourth_weight',
                                      shape=(self.nheads*output_dim, output_dim))
        self._fourth_b = mx.symbol.Variable(self._prefix + '_fourth_bias',
                                      shape=(output_dim,))

        #layer-wise aggregation
        self._final_W = mx.symbol.Variable(self._prefix + '_final_weight',
                                      shape=(6*output_dim, output_dim))
        self._final_b = mx.symbol.Variable(self._prefix + '_final_bias',
                                      shape=(output_dim,))

    def convolve(self, adj, inputs, seq_len):
        output_list = []
        output_list_3 = []
        output_list_5 = []
        layer_list = []

        # if self._dropout != 0.0:
        # 	print("DROPOUT: %f" % self._dropout)
        # 	inputs = mx.sym.Dropout(inputs, p=self._dropout)

        if self._input_dim != self._output_dim:
            inputs = mx.symbol.dot(inputs, self._first_W)
            inputs = mx.symbol.broadcast_add(inputs, self._first_b)
            # inputs = mx.symbol.Activation(inputs, act_type=self._activation)

        #embed_layer = mx.symbol.expand_dims(inputs, axis=1)
        #layer_list.append(embed_layer)

        for i in range(self.nheads):
            convolved = self._first_convolve(adj, inputs, seq_len, i)
            output_list.append(convolved)
        outputs = mx.symbol.concat(*output_list, dim=2)

        outputs = mx.symbol.dot(outputs, self._second_W)
        outputs = mx.symbol.broadcast_add(outputs, self._second_b)

        outputs_2 = self._second_convolve(adj, outputs, seq_len)
        outputs_2 = mx.symbol.LayerNorm(outputs_2)

        # the second layer
        for i in range(self.nheads):
            convolved = self._third_convolve(adj, outputs_2, seq_len, i)
            output_list_3.append(convolved)
        outputs_3 = mx.symbol.concat(*output_list_3, dim=2)

        outputs_3 = mx.symbol.dot(outputs_3, self._third_W)
        outputs_3 = mx.symbol.broadcast_add(outputs_3, self._third_b)

        outputs_4 = self._fourth_convolve(adj, outputs_3, seq_len)
        outputs_4 = mx.symbol.LayerNorm(outputs_4)

        # the third layer
        for i in range(self.nheads):
            convolved = self._fifth_convolve(adj, outputs_4, seq_len, i)
            output_list_5.append(convolved)
        outputs_5 = mx.symbol.concat(*output_list_5, dim=2)

        # outputs_cache_5 = outputs_5
        outputs_5 = mx.symbol.dot(outputs_5, self._fourth_W)
        outputs_5 = mx.symbol.broadcast_add(outputs_5, self._fourth_b)

        outputs_6 = self._sixth_convolve(adj, outputs_5, seq_len)
        outputs_6 = mx.symbol.LayerNorm(outputs_6)

        outputs = mx.symbol.expand_dims(outputs, axis=1)
        outputs_2 = mx.symbol.expand_dims(outputs_2, axis=1)
        outputs_3 = mx.symbol.expand_dims(outputs_3, axis=1)
        outputs_4 = mx.symbol.expand_dims(outputs_4, axis=1)
        outputs_5 = mx.symbol.expand_dims(outputs_5, axis=1)
        outputs_6 = mx.symbol.expand_dims(outputs_6, axis=1)
        layer_list.append(outputs_6)
        layer_list.append(outputs_5)
        layer_list.append(outputs_4)
        layer_list.append(outputs_3)
        layer_list.append(outputs_2)
        layer_list.append(outputs)

        final_output = mx.symbol.concat(*layer_list, dim=1)
        final_output = mx.symbol.sum(final_output, axis=1) * (1/6)
        # final_output = mx.symbol.Dropout(final_output, p=self._dropout)
        # final_output = mx.symbol.dot(final_output, self._final_W)
        # final_output = mx.symbol.broadcast_add(final_output, self._final_b)
        return final_output

    def reset(self):
        pass

    def _first_convolve(self, adj, inputs, seq_len, i):
        """
        IMPORTANT: when retrieving the original adj matrix for an
        edge label we add one to "i" because the edge ids stored
        in the matrix start at 1. 0 corresponds to lack of edges.
        """
        output_list = []
        for j in range(self._tensor_dim):
            # linear transformation
            k = i * self._tensor_dim + j
            Wi = self._W[k]
            bi = self._b[k]
            output = mx.symbol.dot(inputs, Wi)
            output = mx.symbol.broadcast_add(output, bi)

            #attention
            a1 = self._att_1_W[k]
            a2 = self._att_2_W[k]
            f_1 = mx.symbol.dot(output, a1)
            f_2 = mx.symbol.dot(output, a2)
            f_2 = mx.symbol.transpose(f_2, axes=(0,2,1))
            f = mx.symbol.broadcast_add(f_1, f_2)
            # h = mx.symbol.broadcast_add(f_1, gate_val.transpose)
            e = mx.symbol.LeakyReLU(f)
            # convolution
            label_id = j + 1
            mask = mx.symbol.ones_like(adj) * label_id
            adji = (mask == adj)
            zero_vec = mx.symbol.ones_like(e) * (-9e15)
            attention = mx.symbol.where(adji > 0, e, zero_vec)
            attention = mx.symbol.softmax(attention, axis=-1)
            #adji = mx.symbol.slice_axis(adj, axis=1, begin=i, end=i+1)
            #adji = mx.symbol.reshape(adji, shape=(-1, seq_len, seq_len))
            output = mx.symbol.batch_dot(attention, output)
            output = mx.symbol.expand_dims(output, axis=1)
            #output = mx.symbol.LeakyReLU(output, act_type='elu')
            output_list.append(output)
        outputs = mx.symbol.concat(*output_list, dim=1)
        outputs = mx.symbol.sum(outputs, axis=1)
        if self._norm:
            norm_adj = mx.symbol.broadcast_not_equal(adj, mx.symbol.zeros_like(adj))
            norm_factor = mx.symbol.sum(norm_adj, axis=2, keepdims=True)
            outputs = mx.symbol.broadcast_div(outputs, norm_factor)
        final_output = mx.symbol.Activation(outputs, act_type=self._activation)
        final_output = mx.symbol.Dropout(final_output, p=self._dropout)
        final_output = mx.symbol.broadcast_add(final_output, inputs)
        final_output = mx.symbol.LayerNorm(final_output)

        return final_output

    def _second_convolve(self, adj, inputs, seq_len):
        """
        IMPORTANT: when retrieving the original adj matrix for an
        edge label we add one to "i" because the edge ids stored
        in the matrix start at 1. 0 corresponds to lack of edges.
        """
        output_list = []

        for i in range(self._tensor_dim):
            # linear transformation
            Wi = self._2_W[i]
            bi = self._2_b[i]
            output = mx.symbol.dot(inputs, Wi)
            output = mx.symbol.broadcast_add(output, bi)

            #attention
            a1 = self._2_att_1_W[i]
            a2 = self._2_att_2_W[i]
            f_1 = mx.symbol.dot(output, a1)
            f_2 = mx.symbol.dot(output, a2)
            f_2 = mx.symbol.transpose(f_2, axes=(0,2,1))
            f = mx.symbol.broadcast_add(f_1, f_2)
            # h = mx.symbol.broadcast_add(f_1, gate_val.transpose)
            e = mx.symbol.LeakyReLU(f)
            # convolution
            label_id = i + 1
            mask = mx.symbol.ones_like(adj) * label_id
            adji = (mask == adj)
            zero_vec = mx.symbol.ones_like(e) * (-9e15)
            attention = mx.symbol.where(adji > 0, e, zero_vec)
            attention = mx.symbol.softmax(attention, axis=-1)
            #adji = mx.symbol.slice_axis(adj, axis=1, begin=i, end=i+1)
            #adji = mx.symbol.reshape(adji, shape=(-1, seq_len, seq_len))
            output = mx.symbol.batch_dot(attention, output)
            output = mx.symbol.expand_dims(output, axis=1)
            output_list.append(output)
        outputs = mx.symbol.concat(*output_list, dim=1)
        outputs = mx.symbol.sum(outputs, axis=1)
        if self._norm:
            norm_adj = mx.symbol.broadcast_not_equal(adj, mx.symbol.zeros_like(adj))
            norm_factor = mx.symbol.sum(norm_adj, axis=2, keepdims=True)
            outputs = mx.symbol.broadcast_div(outputs, norm_factor)
        final_output = mx.symbol.Activation(outputs, act_type=self._activation)
        final_output = mx.symbol.Dropout(final_output, p=self._dropout)
        final_output = mx.symbol.broadcast_add(final_output, inputs)

        return final_output

    def _third_convolve(self, adj, inputs, seq_len, i):
        """
        IMPORTANT: when retrieving the original adj matrix for an
        edge label we add one to "i" because the edge ids stored
        in the matrix start at 1. 0 corresponds to lack of edges.
        """
        output_list = []
        for j in range(self._tensor_dim):
            # linear transformation
            k = i * self._tensor_dim + j
            Wi = self._3_W[k]
            bi = self._3_b[k]
            output = mx.symbol.dot(inputs, Wi)
            output = mx.symbol.broadcast_add(output, bi)
            #attention
            a1 = self._3_att_1_W[k]
            a2 = self._3_att_2_W[k]
            f_1 = mx.symbol.dot(output, a1)
            f_2 = mx.symbol.dot(output, a2)
            f_2 = mx.symbol.transpose(f_2, axes=(0,2,1))
            f = mx.symbol.broadcast_add(f_1, f_2)
            # h = mx.symbol.broadcast_add(f_1, gate_val.transpose)
            e = mx.symbol.LeakyReLU(f)
            # convolution
            label_id = j + 1
            mask = mx.symbol.ones_like(adj) * label_id
            adji = (mask == adj)
            zero_vec = mx.symbol.ones_like(e) * (-9e15)
            attention = mx.symbol.where(adji > 0, e, zero_vec)
            attention = mx.symbol.softmax(attention, axis=-1)
            #adji = mx.symbol.slice_axis(adj, axis=1, begin=i, end=i+1)
            #adji = mx.symbol.reshape(adji, shape=(-1, seq_len, seq_len))
            output = mx.symbol.batch_dot(attention, output)
            output = mx.symbol.expand_dims(output, axis=1)
            #output = mx.symbol.LeakyReLU(output, act_type='elu')
            output_list.append(output)
        outputs = mx.symbol.concat(*output_list, dim=1)
        outputs = mx.symbol.sum(outputs, axis=1)
        if self._norm:
            norm_adj = mx.symbol.broadcast_not_equal(adj, mx.symbol.zeros_like(adj))
            norm_factor = mx.symbol.sum(norm_adj, axis=2, keepdims=True)
            outputs = mx.symbol.broadcast_div(outputs, norm_factor)
        final_output = mx.symbol.Activation(outputs, act_type=self._activation)
        final_output = mx.symbol.Dropout(final_output, p=self._dropout)
        final_output = mx.symbol.broadcast_add(final_output, inputs)
        final_output = mx.symbol.LayerNorm(final_output)

        return final_output

    def _fourth_convolve(self, adj, inputs, seq_len):
        """
        IMPORTANT: when retrieving the original adj matrix for an
        edge label we add one to "i" because the edge ids stored
        in the matrix start at 1. 0 corresponds to lack of edges.
        """
        output_list = []

        for i in range(self._tensor_dim):
            # linear transformation
            Wi = self._4_W[i]
            bi = self._4_b[i]
            output = mx.symbol.dot(inputs, Wi)
            output = mx.symbol.broadcast_add(output, bi)

            #attention
            a1 = self._4_att_1_W[i]
            a2 = self._4_att_2_W[i]
            f_1 = mx.symbol.dot(output, a1)
            f_2 = mx.symbol.dot(output, a2)
            f_2 = mx.symbol.transpose(f_2, axes=(0,2,1))
            f = mx.symbol.broadcast_add(f_1, f_2)
            # h = mx.symbol.broadcast_add(f_1, gate_val.transpose)
            e = mx.symbol.LeakyReLU(f)
            # convolution
            label_id = i + 1
            mask = mx.symbol.ones_like(adj) * label_id
            adji = (mask == adj)
            zero_vec = mx.symbol.ones_like(e) * (-9e15)
            attention = mx.symbol.where(adji > 0, e, zero_vec)
            attention = mx.symbol.softmax(attention, axis=-1)
            #adji = mx.symbol.slice_axis(adj, axis=1, begin=i, end=i+1)
            #adji = mx.symbol.reshape(adji, shape=(-1, seq_len, seq_len))
            output = mx.symbol.batch_dot(attention, output)
            output = mx.symbol.expand_dims(output, axis=1)
            output_list.append(output)
        outputs = mx.symbol.concat(*output_list, dim=1)
        outputs = mx.symbol.sum(outputs, axis=1)
        if self._norm:
            norm_adj = mx.symbol.broadcast_not_equal(adj, mx.symbol.zeros_like(adj))
            norm_factor = mx.symbol.sum(norm_adj, axis=2, keepdims=True)
            outputs = mx.symbol.broadcast_div(outputs, norm_factor)
        final_output = mx.symbol.Activation(outputs, act_type=self._activation)
        final_output = mx.symbol.Dropout(final_output, p=self._dropout)
        final_output = mx.symbol.broadcast_add(final_output, inputs)

        return final_output

    def _fifth_convolve(self, adj, inputs, seq_len, i):
        """
        IMPORTANT: when retrieving the original adj matrix for an
        edge label we add one to "i" because the edge ids stored
        in the matrix start at 1. 0 corresponds to lack of edges.
        """
        output_list = []
        for j in range(self._tensor_dim):
            # linear transformation
            k = i * self._tensor_dim + j
            Wi = self._5_W[k]
            bi = self._5_b[k]
            output = mx.symbol.dot(inputs, Wi)
            output = mx.symbol.broadcast_add(output, bi)
            #attention
            a1 = self._5_att_1_W[k]
            a2 = self._5_att_2_W[k]
            f_1 = mx.symbol.dot(output, a1)
            f_2 = mx.symbol.dot(output, a2)
            f_2 = mx.symbol.transpose(f_2, axes=(0,2,1))
            f = mx.symbol.broadcast_add(f_1, f_2)
            # h = mx.symbol.broadcast_add(f_1, gate_val.transpose)
            e = mx.symbol.LeakyReLU(f)
            # convolution
            label_id = j + 1
            mask = mx.symbol.ones_like(adj) * label_id
            adji = (mask == adj)
            zero_vec = mx.symbol.ones_like(e) * (-9e15)
            attention = mx.symbol.where(adji > 0, e, zero_vec)
            attention = mx.symbol.softmax(attention, axis=-1)
            #adji = mx.symbol.slice_axis(adj, axis=1, begin=i, end=i+1)
            #adji = mx.symbol.reshape(adji, shape=(-1, seq_len, seq_len))
            output = mx.symbol.batch_dot(attention, output)
            output = mx.symbol.expand_dims(output, axis=1)
            #output = mx.symbol.LeakyReLU(output, act_type='elu')
            output_list.append(output)
        outputs = mx.symbol.concat(*output_list, dim=1)
        outputs = mx.symbol.sum(outputs, axis=1)
        if self._norm:
            norm_adj = mx.symbol.broadcast_not_equal(adj, mx.symbol.zeros_like(adj))
            norm_factor = mx.symbol.sum(norm_adj, axis=2, keepdims=True)
            outputs = mx.symbol.broadcast_div(outputs, norm_factor)
        final_output = mx.symbol.Activation(outputs, act_type=self._activation)
        final_output = mx.symbol.Dropout(final_output, p=self._dropout)
        final_output = mx.symbol.broadcast_add(final_output, inputs)
        final_output = mx.symbol.LayerNorm(final_output)

        return final_output

    def _sixth_convolve(self, adj, inputs, seq_len):
        """
        IMPORTANT: when retrieving the original adj matrix for an
        edge label we add one to "i" because the edge ids stored
        in the matrix start at 1. 0 corresponds to lack of edges.
        """
        output_list = []

        for i in range(self._tensor_dim):
            # linear transformation
            Wi = self._6_W[i]
            bi = self._6_b[i]
            output = mx.symbol.dot(inputs, Wi)
            output = mx.symbol.broadcast_add(output, bi)

            #attention
            a1 = self._6_att_1_W[i]
            a2 = self._6_att_2_W[i]
            f_1 = mx.symbol.dot(output, a1)
            f_2 = mx.symbol.dot(output, a2)
            f_2 = mx.symbol.transpose(f_2, axes=(0,2,1))
            f = mx.symbol.broadcast_add(f_1, f_2)
            # h = mx.symbol.broadcast_add(f_1, gate_val.transpose)
            e = mx.symbol.LeakyReLU(f)
            # convolution
            label_id = i + 1
            mask = mx.symbol.ones_like(adj) * label_id
            adji = (mask == adj)
            zero_vec = mx.symbol.ones_like(e) * (-9e15)
            attention = mx.symbol.where(adji > 0, e, zero_vec)
            attention = mx.symbol.softmax(attention, axis=-1)
            #adji = mx.symbol.slice_axis(adj, axis=1, begin=i, end=i+1)
            #adji = mx.symbol.reshape(adji, shape=(-1, seq_len, seq_len))
            output = mx.symbol.batch_dot(attention, output)
            output = mx.symbol.expand_dims(output, axis=1)
            output_list.append(output)
        outputs = mx.symbol.concat(*output_list, dim=1)
        outputs = mx.symbol.sum(outputs, axis=1)
        if self._norm:
            norm_adj = mx.symbol.broadcast_not_equal(adj, mx.symbol.zeros_like(adj))
            norm_factor = mx.symbol.sum(norm_adj, axis=2, keepdims=True)
            outputs = mx.symbol.broadcast_div(outputs, norm_factor)
        final_output = mx.symbol.Activation(outputs, act_type=self._activation)
        final_output = mx.symbol.Dropout(final_output, p=self._dropout)
        final_output = mx.symbol.broadcast_add(final_output, inputs)

        return final_output
