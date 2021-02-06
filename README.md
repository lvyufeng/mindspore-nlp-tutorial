# mindspore-nlp-tutorial

<p align="center"><img width="400" src="https://gitee.com/mindspore/mindspore/raw/master/docs/MindSpore-logo.png" /></p>

`mindspore-nlp-tutorial` is a tutorial for who is studying NLP(Natural Language Processing) using **MindSpore**. This repository is migrated from [nlp-tutorial](https://github.com/graykode/nlp-tutorial). Most of the models in NLP were migrated from Pytorch version with less than **100 lines** of code.(except comments or blank lines)

- **Notice**: All models are implemented in GPU version, but not test on Ascend platform.

## Curriculum - (Example Purpose)

#### 1. Basic Embedding Model

- 1-1. [NNLM(Neural Network Language Model)](1-1.NNLM) - **Predict Next Word**
  - Paper -  [A Neural Probabilistic Language Model(2003)](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- 1-2. [Word2Vec(Skip-gram)](1-2.Word2Vec) - **Embedding Words and Show Graph**
  - Paper - [Distributed Representations of Words and Phrases
    and their Compositionality(2013)](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
<!-- - 1-3. [FastText(Application Level)](1-3.FastText) - **Sentence Classification**
  - Paper - [Bag of Tricks for Efficient Text Classification(2016)](https://arxiv.org/pdf/1607.01759.pdf)
  - Colab - [FastText.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/1-3.FastText/FastText.ipynb)  -->



#### 2. CNN(Convolutional Neural Network)

- 2-1. [TextCNN](2-1.TextCNN) - **Binary Sentiment Classification**
  - Paper - [Convolutional Neural Networks for Sentence Classification(2014)](http://www.aclweb.org/anthology/D14-1181)



#### 3. RNN(Recurrent Neural Network)

- 3-1. [TextRNN](3-1.TextRNN) - **Predict Next Step**
  - Paper - [Finding Structure in Time(1990)](http://psych.colorado.edu/~kimlab/Elman1990.pdf)
- 3-2. [TextLSTM](3-2.TextLSTM) - **Autocomplete**
  - Paper - [LONG SHORT-TERM MEMORY(1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)
- 3-3. [Bi-LSTM](3-3.Bi-LSTM) - **Predict Next Word in Long Sentence**


#### 4. Attention Mechanism

- 4-1. [Seq2Seq](4-1.Seq2Seq) - **Change Word**
  - Paper - [Learning Phrase Representations using RNN Encoder–Decoder
    for Statistical Machine Translation(2014)](https://arxiv.org/pdf/1406.1078.pdf)
<!-- 
- 4-2. [Seq2Seq with Attention](4-2.Seq2Seq(Attention)) - **Translate**
  - Paper - [Neural Machine Translation by Jointly Learning to Align and Translate(2014)](https://arxiv.org/abs/1409.0473)
  - Colab - [Seq2Seq(Attention).ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/4-2.Seq2Seq(Attention)/Seq2Seq(Attention).ipynb)
- 4-3. [Bi-LSTM with Attention](4-3.Bi-LSTM(Attention)) - **Binary Sentiment Classification**
  <!-- - Colab - [Bi_LSTM(Attention).ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/4-3.Bi-LSTM(Attention)/Bi_LSTM(Attention).ipynb) -->



#### 5. Model based on Transformer

- 5-1.  [The Transformer](5-1.Transformer) - **Translate**
  - Paper - [Attention Is All You Need(2017)](https://arxiv.org/abs/1706.03762)
  <!-- - Colab - [Transformer.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/5-1.Transformer/Transformer.ipynb), [Transformer(Greedy_decoder).ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/5-1.Transformer/Transformer(Greedy_decoder).ipynb) -->
<!-- - 5-2. [BERT](5-2.BERT) - **Classification Next Sentence & Predict Masked Tokens**
  - Paper - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding(2018)](https://arxiv.org/abs/1810.04805)
  - Colab - [BERT.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/5-2.BERT/BERT.ipynb) -->



## Dependencies

- Python 3.7.5
- MindSpore 1.1.1(GPU version)
- Pytorch 1.7.1(for comparation)

## Author

- Yufeng Lyu
- Author Email : lvyufeng2007@hotmail.com
- Acknowledgements to [graykode](https://github.com/graykode) who opensource the Pytorch and Tensorflow version.