import math
import mindspore
import mindspore.nn as nn
from mindspore import Parameter, Tensor
from mindspore.common.initializer import initializer, HeUniform, Uniform, Normal, _calculate_fan_in_and_fan_out

class Conv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad_mode='same', padding=0, dilation=1, group=1, has_bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, pad_mode, padding, dilation, group, has_bias, weight_init='normal', bias_init='zeros')
        self.reset_parameters()
        
    def reset_parameters(self):
        self.weight = Parameter(initializer(HeUniform(math.sqrt(5)), self.weight.shape), name='weight')
        if self.has_bias:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight.shape)
            bound = 1 / math.sqrt(fan_in)
            self.bias = Parameter(initializer(Uniform(bound), [self.out_channels]), name='bias')

class Dense(nn.Dense):
    def __init__(self, in_channels, out_channels, has_bias=True, activation=None):
        super().__init__(in_channels, out_channels, weight_init='normal', bias_init='zeros', has_bias=has_bias, activation=activation)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.weight = Parameter(initializer(HeUniform(math.sqrt(5)), self.weight.shape), name='weight')
        if self.has_bias:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight.shape)
            bound = 1 / math.sqrt(fan_in)
            self.bias = Parameter(initializer(Uniform(bound), [self.out_channels]), name='bias')

class Embedding(nn.Embedding):
    def __init__(self, vocab_size, embedding_size, use_one_hot=False, embedding_table='normal', dtype=mindspore.float32, padding_idx=None):
        if embedding_table == 'normal':
            embedding_table = Normal(1.0)
        super().__init__(vocab_size, embedding_size, use_one_hot, embedding_table, dtype, padding_idx)
    @classmethod
    def from_pretrained_embedding(cls, embeddings:Tensor, freeze=True, padding_idx=None):
        rows, cols = embeddings.shape
        embedding = cls(rows, cols, embedding_table=embeddings, padding_idx=padding_idx)
        embedding.embedding_table.requires_grad = not freeze
        return embedding