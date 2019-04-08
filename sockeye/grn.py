"""
Implementation of a range of Graph Recurrent Networks.
Trying to follow the structure of rnn_cell.py in the mxnet code.
"""

import mxnet as mx

import sockeye.constants as C
from sockeye.config import Config


import logging
logger = logging.getLogger(__name__)


def get_gatedgrn(config, prefix):
    gatedgrn = GatedGRNCell(config.input_dim,
                            config.output_dim,
                            config.tensor_dim,
                            config.num_layers,
                            activation=config.activation,
                            add_gate=config.add_gate,
                            dropout=config.dropout,
                            norm=config.norm,
                            prefix=prefix)
    return gatedgrn


class GatedGRNConfig(Config):
    """
    GCN configuration.

    :param input_dim: Dimensionality for input vectors.
    :param output_dim: Dimensionality for output vectors.
    :param tensor_dim: Edge label space dimensionality.
    :param layers: Number of layers / unrolled timesteps.
    :param no_residual: skip residual connections
    :param activation: Non-linear function used inside the GGRN updates.
    :param add_gate: Add edge-wise gating (Marcheggiani & Titov, 2017).
    :param dropout: Dropout between layers.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 tensor_dim: int,
                 num_layers: int,
                 activation: str = 'relu',
                 add_gate: bool = False,
                 dropout: float = 0.0,
                 norm: bool = False,
                 dtype: str = C.DTYPE_FP32) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tensor_dim = tensor_dim
        self.num_layers = num_layers
        self.activation = activation
        self.add_gate = add_gate
        self.dropout = dropout
        self.norm = norm
        self.dtype = dtype
        
                 
class GatedGRNCell(object):
    """Gated GRN cell
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 tensor_dim,
                 num_layers,
                 activation='relu',
                 add_gate=False,
                 prefix='gatedgrn_',
                 params=None, 
                 dropout=0.0,
                 norm=False):
        self._prefix = prefix
        self._params = params
        self._modified = False
        
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._tensor_dim = tensor_dim
        self._num_layers = num_layers
        self._activation = activation        
        self._add_edge_gate = add_gate
        self._dropout = dropout
        self._dropout_mask = None
        self._norm = norm

        # Linear transformation for the first layer in case input vectors
        # are of a different dimensionality from the output vectors
        if self._input_dim != self._output_dim:
            self._first_W = mx.symbol.Variable(self._prefix + '_first_weight',
                                               shape=(input_dim, output_dim))
            self._first_b = mx.symbol.Variable(self._prefix + '_first_bias',
                                               shape=(output_dim,))

        # Main transformation, using label-wise parameters
        self._Wl = [mx.symbol.Variable(self._prefix + str(i) + '_edge_weight',
                                       shape=(output_dim, output_dim))
                    for i in range(tensor_dim)]

        self._bl = [mx.symbol.Variable(self._prefix + str(i) + '_edge_bias',
                                       shape=(output_dim,))
                    for i in range(tensor_dim)]

        # Reset gate
        self._reset_Wl = [mx.symbol.Variable(self._prefix + str(i) + '_reset_weight',
                                             shape=(output_dim, output_dim))
                          for i in range(tensor_dim)]
        
        self._reset_bl = [mx.symbol.Variable(self._prefix + str(i) + '_reset_bias',
                                             shape=(output_dim,))
                          for i in range(tensor_dim)]

        # Update gate
        self._update_Wl = [mx.symbol.Variable(self._prefix + str(i) + '_update_weight',
                                              shape=(output_dim, output_dim))
                           for i in range(tensor_dim)]
        
        self._update_bl = [mx.symbol.Variable(self._prefix + str(i) + '_update_bias',
                                               shape=(output_dim,))
                            for i in range(tensor_dim)]
        
        # Edge gate parameters
        if self._add_edge_gate:
            self._edge_gate_W = [mx.symbol.Variable(self._prefix + str(i) + '_edge_gate_weight',
                                                    shape=(output_dim, 1))
                                 for i in range(tensor_dim)]
            self._edge_gate_b = [mx.symbol.Variable(self._prefix + str(i) + '_edge_gate_bias',
                                                    shape=(1, 1))
                                 for i in range(tensor_dim)]

    def convolve(self, adj, inputs, seq_len):
        """
        Apply one convolution per layer. This is where we apply the gates
        A linear transformation is required in case the input dimensionality is
        different from GRN output dimensionality.
        """
        # Dropout is applied on inputs
        if self._dropout != 0.0:
            print("DROPOUT: %f" % self._dropout)
            inputs = mx.sym.Dropout(inputs, p=self._dropout)
        
        # Transformation to match dims
        if self._input_dim != self._output_dim:
            outputs = mx.symbol.dot(inputs, self._first_W)
            outputs = mx.symbol.broadcast_add(outputs, self._first_b)
            # Sounded like a sensible idea but didn't really work...
            # I guess because ReLU?
            #outputs = mx.symbol.Activation(outputs, act_type=self._activation)
        else:
            outputs = inputs

        # Variational/Bayesian Dropout mask. Mask does not change between layers.
        #if self._dropout_mask is None:
        #self._dropout_mask = mx.sym.Dropout(data=mx.sym.ones_like(outputs), p=self._dropout)

        # Convolutions
        for i in range(self._num_layers):
            reset_outputs = self._reset(adj, outputs, seq_len)
            convolved = self._single_convolve(adj, reset_outputs, seq_len)
            outputs = self._update(adj, outputs, convolved, seq_len)
            #if self._dropout != 0.0:
            #    outputs = mx.symbol.Dropout(outputs, p=self._dropout)
            #outputs = outputs * self._dropout_mask
            #outputs = outputs * mx.sym.Dropout(data=mx.sym.ones_like(outputs), p=self._dropout)
            #outputs = outputs * mx.sym.ones_like(outputs)
        return outputs

    def _reset(self, adj, inputs, seq_len):
        """
        Apply reset gate to the inputs.
        IMPORTANT: when retrieving the original adj matrix for an
        edge label we add one to "i" because the edge ids stored
        in the matrix start at 1. 0 corresponds to lack of edges.
        """
        output_list = []
        for i in range(self._tensor_dim):
            # linear transformation
            reset_Wi = self._reset_Wl[i]
            reset_bi = self._reset_bl[i]            
            output = mx.symbol.dot(inputs, reset_Wi)
            output = mx.symbol.broadcast_add(output, reset_bi)
            # convolution
            label_id = i + 1
            mask = mx.symbol.ones_like(adj) * label_id
            adji = (mask == adj)
            output = mx.symbol.batch_dot(adji, output)
            output = mx.symbol.expand_dims(output, axis=1)
            output_list.append(output)
        outputs = mx.symbol.concat(*output_list, dim=1)
        outputs = mx.symbol.sum(outputs, axis=1)
        reset_gate = mx.symbol.Activation(outputs, act_type='sigmoid')
        final_outputs = mx.symbol.broadcast_mul(reset_gate, inputs)
        return final_outputs

    def _update(self, adj, inputs, convolved, seq_len):
        """
        Apply update gate to the inputs.
        IMPORTANT: when retrieving the original adj matrix for an
        edge label we add one to "i" because the edge ids stored
        in the matrix start at 1. 0 corresponds to lack of edges.
        """
        output_list = []
        for i in range(self._tensor_dim):
            # linear transformation
            update_Wi = self._update_Wl[i]
            update_bi = self._update_bl[i]            
            output = mx.symbol.dot(convolved, update_Wi)
            output = mx.symbol.broadcast_add(output, update_bi)
            # convolution
            label_id = i + 1
            mask = mx.symbol.ones_like(adj) * label_id
            adji = (mask == adj)
            output = mx.symbol.batch_dot(adji, output)
            output = mx.symbol.expand_dims(output, axis=1)
            output_list.append(output)
        outputs = mx.symbol.concat(*output_list, dim=1)
        outputs = mx.symbol.sum(outputs, axis=1)
        update_gate = mx.symbol.Activation(outputs, act_type='sigmoid')
        final_outputs = (mx.symbol.broadcast_mul(update_gate, convolved) +
                         mx.symbol.broadcast_mul((1 - update_gate), inputs))
        return final_outputs
    
    def _single_convolve(self, adj, inputs, seq_len):
        """
        IMPORTANT: when retrieving the original adj matrix for an
        edge label we add one to "i" because the edge ids stored
        in the matrix start at 1. 0 corresponds to lack of edges.
        """
        output_list = []
        for i in range(self._tensor_dim):
            # linear transformation
            Wi = self._Wl[i]
            #Wi = mx.symbol.dot(self._W, Wi)
            bi = self._bl[i]            
            output = mx.symbol.dot(inputs, Wi)
            output = mx.symbol.broadcast_add(output, bi)
            # optional edge gating
            if self._add_edge_gate:
                edge_gate_Wi = self._edge_gate_W[i]
                edge_gate_bi = self._edge_gate_b[i]
                edge_gate_val = mx.symbol.dot(inputs, edge_gate_Wi)
                edge_gate_val = mx.symbol.broadcast_add(edge_gate_val, edge_gate_bi)
                edge_gate_val = mx.symbol.Activation(edge_gate_val, act_type='sigmoid')
                output = mx.symbol.broadcast_mul(output, edge_gate_val)
            # convolution
            label_id = i + 1
            mask = mx.symbol.ones_like(adj) * label_id
            #adji = (mask == adj)
            adji = mx.symbol.broadcast_equal(mask, adj)
            #adji = mx.symbol.slice_axis(adj, axis=1, begin=i, end=i+1)
            #adji = mx.symbol.reshape(adji, shape=(-1, seq_len, seq_len))
            output = mx.symbol.batch_dot(adji, output)
            output = mx.symbol.expand_dims(output, axis=1)
            output_list.append(output)
        outputs = mx.symbol.concat(*output_list, dim=1)
        outputs = mx.symbol.sum(outputs, axis=1)
        if self._norm:
            norm_adj = mx.symbol.broadcast_not_equal(adj, mx.symbol.zeros_like(adj))
            norm_factor = mx.symbol.sum(norm_adj, axis=2, keepdims=True)
            outputs = mx.symbol.broadcast_div(outputs, norm_factor)
        final_output = mx.symbol.Activation(outputs, act_type=self._activation)
        #final_output = mx.symbol.Dropout(final_output, p=self._dropout)
        return final_output

    def reset(self):
        logger.info("GRN DROPOUT MASK RESET")
        self._dropout_mask = None
