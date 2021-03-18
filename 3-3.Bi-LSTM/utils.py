import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.ops import constexpr 

@constexpr
def Range(length_input, reverse=False):
    if reverse:
        return Tensor(list(reversed(range(length_input))), mindspore.int32)
    return Tensor(list(range(length_input)), mindspore.int32)

class Reverse(nn.Cell):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def construct(self,input_x):
        shape = input_x.shape
        dim_size = shape[self.dim]
        reversed_indexes = Range(dim_size, True)
        output = ops.Gather()(input_x, reversed_indexes, self.dim)
        return output
    
class ReverseSequence(nn.Cell):
    def __init__(self, seq_dim, batch_dim=0):
        super().__init__()
        self.seq_dim = seq_dim
        self.batch_dim = batch_dim
        
    def construct(self, x, seq_lengths):
        batch_size = x.shape[self.batch_dim]
        max_seq_len = x.shape[self.seq_dim]
        seq_lens_type = seq_lengths.dtype
        
        # Create [batch, sequence, 2] tensor that contains the indices where the
        # real data belongs
        back = ops.Sub()(seq_lengths, ops.OnesLike()(seq_lengths))
        
        batch_idx = self.make_shape((batch_size, max_seq_len), seq_lens_type, 0)
        forward_idx= self.make_shape((batch_size, max_seq_len), seq_lens_type, 1)
        
        back = back.view(-1, 1)
        reverse_idx = ops.Sub()(back, forward_idx)
        
        condition = ops.Less()(reverse_idx, ops.ZerosLike()(reverse_idx))
        reverse_idx = ops.Select()(condition, forward_idx, reverse_idx)
        
        reverse_idx = ops.ExpandDims()(reverse_idx, 2)
        batch_idx = ops.ExpandDims()(batch_idx, 2)
        
        if self.batch_dim > self.seq_dim:
            batch_idx = ops.Transpose()(batch_idx, (1, 0, 2))
            reverse_idx = ops.Transpose()(reverse_idx, (1, 0, 2))
            x = ops.Transpose()(x, (1, 0, 2))
        start_indices = ops.Concat(2)((batch_idx, reverse_idx))
        
        output = ops.GatherNd()(x, start_indices)
        
        return output
    
    def make_shape(self, shape, dtype, range_dim):
        output = ops.Ones()(shape, mindspore.float32)
        output = ops.CumSum()(output, range_dim)
        output = ops.Cast()(output, dtype)
        output = output - 1
        return output