### [TextRNN](3-1.TextRNN) - **Predict Next Step**

Since MindSpore didn't provide the  official RNN layer, we used python level operations to build a simple RNN layer to use. The implementation was following paper [Finding Structure in Time(1990)](http://psych.colorado.edu/~kimlab/Elman1990.pdf) and [Pytorch code](https://pytorch.org/docs/stable/generated/torch.nn.RNNCell.html?highlight=rnncell#torch.nn.RNNCell).

##### RNNCell:

![1](https://latex.codecogs.com/svg.latex?h%27%20=%20\tanh(W_{ih}%20x%20+%20b_{ih}%20%20+%20%20W_{hh}%20h%20+%20b_{hh}))

- Args:
    - **input_size**:  The number of expected features in the input '*x*'
    - **hidden_size**: The number of features in the hidden state '*h*'
    - **bias**: If '*False*', then the layer does not use bias weights b_ih and b_hh. Default: '*True*'
    - **nonlinearity**: The non-linearity to use. Can be either '*tanh*' or '*relu*'. Default: '*tanh*'
    
- Inputs:
    - **input**: Tensor, (batch, input_size)
    - **hidden**: Tensor, (batch, hidden_size)
- Outputs:
    - **h**: Tensor, (batch, hidden_size)