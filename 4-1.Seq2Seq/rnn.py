import math
import mindspore
import numpy as np
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import Tensor, Parameter

class RNNCell(nn.Cell):
    """
    An Elman RNN cell with tanh or ReLU non-linearity.
    
    Args:
        input_size:  The number of expected features in the input 'x'
        hidden_size: The number of features in the hidden state 'h'
        bias: If 'False', then the layer does not use bias weights b_ih and b_hh. Default: 'True'
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'
    
    Inputs:
        input: Tensor, (batch, input_size)
        hidden: Tensor, (batch, hidden_size)
    Outputs:
        h: Tensor, (batch, hidden_size)
    """
    nonlinearity_dict = {
        'tanh': nn.Tanh(),
        'relu': nn.ReLU()
    }
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, nonlinearity: str = 'tanh'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        stdv = 1 / math.sqrt(hidden_size)
        self.weight_ih = Parameter(Tensor(np.random.uniform(-stdv, stdv, (input_size, hidden_size)).astype(np.float32)))
        self.weight_hh = Parameter(Tensor(np.random.uniform(-stdv, stdv, (hidden_size, hidden_size)).astype(np.float32)))
        if bias:
            self.bias_ih = Parameter(Tensor(np.random.uniform(-stdv, stdv, (hidden_size)).astype(np.float32)))
            self.bias_hh = Parameter(Tensor(np.random.uniform(-stdv, stdv, (hidden_size)).astype(np.float32)))
        
        self.nonlinearity = self.nonlinearity_dict[nonlinearity]
        self.mm = P.MatMul()

    def construct(self, input: Tensor, hx: Tensor) -> Tensor:
        if self.bias:
            i_gates = self.mm(input, self.weight_ih) + self.bias_ih
            h_gates = self.mm(hx, self.weight_hh) + self.bias_hh
        else:
            i_gates = self.mm(input, self.weight_ih)
            h_gates = self.mm(hx, self.weight_hh)
        h = self.nonlinearity(i_gates + h_gates)
        return h
    
class RNN(nn.Cell):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 nonlinearity: str = 'tanh', 
                 bias: bool = True, 
                 dropout: float = 0.,
                 batch_first: bool = False,
                ):
        super().__init__()
        self.rnn_cell = RNNCell(input_size, hidden_size, bias, nonlinearity)
        
        self.batch_first = batch_first
        self.transpose = P.Transpose()
        self.pack = P.Pack()
        self.dropout = nn.Dropout(1 - dropout)
    def construct(self, input: Tensor, h_0: Tensor):
        if self.batch_first:
            input = self.transpose(input, (1, 0, 2))
        input_shape = input.shape
        time_steps = input_shape[0]
        h_t = h_0
        output = []
        for t in range(time_steps):
            h_t = self.rnn_cell(input[t], h_t)
            output.append(h_t)
        output = self.pack(output)
        h_t = self.dropout(h_t)
        output = self.dropout(output)
        return output, h_t
    
class WithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn
        self.transpose = P.Transpose()
    def construct(self, *args):
        out = self._backbone(*args[:-1])
        out = self.transpose(out, (1, 0, 2))
        return self._loss_fn(out, args[-1])