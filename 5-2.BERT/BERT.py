#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
from random import *
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from layers import Dense, Embedding


# In[2]:


# sample IsNext and NotNext to be same in small batch size
def make_batch():
    batch = []
    positive = negative = 0
    while positive != batch_size/2 or negative != batch_size/2:
        tokens_a_index, tokens_b_index= randrange(len(sentences)), randrange(len(sentences)) # sample random index in sentences
        tokens_a, tokens_b= token_list[tokens_a_index], token_list[tokens_b_index]
        input_ids = [word_dict['[CLS]']] + tokens_a + [word_dict['[SEP]']] + tokens_b + [word_dict['[SEP]']]
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        # MASK LM
        n_pred =  min(max_pred, max(1, int(round(len(input_ids) * 0.15)))) # 15 % of tokens in one sentence
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != word_dict['[CLS]'] and token != word_dict['[SEP]']]
        shuffle(cand_maked_pos)
        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random() < 0.8:  # 80%
                input_ids[pos] = word_dict['[MASK]'] # make mask
            elif random() < 0.5:  # 10%
                index = randint(0, vocab_size - 1) # random index in vocabulary
                input_ids[pos] = word_dict[number_dict[index]] # replace

        # Zero Paddings
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        # Zero Padding (100% - 15%) tokens
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        if tokens_a_index + 1 == tokens_b_index and positive < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, 1]) # IsNext
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, 0]) # NotNext
            negative += 1
    return batch
# Proprecessing Finished


# In[3]:


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.shape
    batch_size, len_k = seq_k.shape
    
    pad_attn_mask = ops.equal(seq_k, 0)
    pad_attn_mask = pad_attn_mask.expand_dims(1) # batch_size x 1 x len_k(=len_q), one is masking

    return ops.broadcast_to(pad_attn_mask, (batch_size, len_q, len_k)) # batch_size x len_q x len_k


# In[4]:


class BertEmbedding(nn.Cell):
    def __init__(self):
        super(BertEmbedding, self).__init__()
        self.tok_embed = Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = Embedding(maxlen, d_model)  # position embedding
        self.seg_embed = Embedding(n_segments, d_model)  # segment(token type) embedding
        self.norm = nn.LayerNorm([d_model,])

    def construct(self, x, seg):
        seq_len = x.shape[1]
        pos = ops.arange(seq_len, dtype=mindspore.int64)
        pos = pos.expand_dims(0).expand_as(x)  # (seq_len,) -> (batch_size, seq_len)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)


# In[5]:


class ScaledDotProductAttention(nn.Cell):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, Q, K, V, attn_mask):
        scores = ops.matmul(Q, K.swapaxes(-1, -2)) / ops.sqrt(ops.scalar_to_tensor(d_k)) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores = scores.masked_fill(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = self.softmax(scores)
        context = ops.matmul(attn, V)
        return context, attn


# In[6]:


class MultiHeadAttention(nn.Cell):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = Dense(d_model, d_k * n_heads)
        self.W_K = Dense(d_model, d_k * n_heads)
        self.W_V = Dense(d_model, d_v * n_heads)
        self.attn = ScaledDotProductAttention()
        self.out_fc = Dense(n_heads * d_v, d_model)
        self.norm = nn.LayerNorm([d_model,])

    def construct(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.shape[0]
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).swapaxes(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).swapaxes(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).swapaxes(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = attn_mask.expand_dims(1)
        attn_mask = ops.tile(attn_mask, (1, n_heads, 1, 1))
        
        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = self.attn(q_s, k_s, v_s, attn_mask)
        context = context.swapaxes(1, 2).view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = self.out_fc(context)
        return self.norm(output + residual), attn # output: [batch_size x len_q x d_model]


# In[7]:


class PoswiseFeedForwardNet(nn.Cell):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = Dense(d_model, d_ff)
        self.fc2 = Dense(d_ff, d_model)
        self.activation = nn.GELU(False)

    def construct(self, x):
        # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
        return self.fc2(self.activation(self.fc1(x)))


# In[8]:


class EncoderLayer(nn.Cell):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def construct(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn


# In[9]:


class BERT(nn.Cell):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = BertEmbedding()
        self.layers = nn.CellList([EncoderLayer() for _ in range(n_layers)])
        self.fc = Dense(d_model, d_model)
        self.activ1 = nn.Tanh()
        self.linear = Dense(d_model, d_model)
        self.activ2 = nn.GELU(False)
        self.norm = nn.LayerNorm([d_model,])
        self.classifier = Dense(d_model, 2)
        # decoder is shared with embedding layer
        embed_weight = self.embedding.tok_embed.embedding_table
        n_vocab, n_dim = embed_weight.shape
        self.decoder = Dense(n_dim, n_vocab, has_bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = mindspore.Parameter(ops.zeros(n_vocab), 'decoder_bias')

    def construct(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)
        # output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_mode, d_model]
        # it will be decided by first token(CLS)
        h_pooled = self.activ1(self.fc(output[:, 0])) # [batch_size, d_model]
        logits_clsf = self.classifier(h_pooled) # [batch_size, 2]

        masked_pos = ops.tile(masked_pos[:, :, None], (1, 1, output.shape[-1])) # [batch_size, max_pred, d_model]
        # get masked position from final output of transformer.
        h_masked = ops.gather_d(output, 1, masked_pos) # masking position [batch_size, max_pred, d_model]
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias # [batch_size, max_pred, n_vocab]

        return logits_lm, logits_clsf


# In[10]:


# BERT Parameters
maxlen = 30 # maximum of length
batch_size = 6
max_pred = 5  # max tokens of prediction
n_layers = 6 # number of Encoder of Encoder Layer
n_heads = 12 # number of heads in Multi-Head Attention
d_model = 768 # Embedding Size
d_ff = 768 * 4  # 4*d_model, FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_segments = 2


# In[11]:


text = (
    'Hello, how are you? I am Romeo.\n'
    'Hello, Romeo My name is Juliet. Nice to meet you.\n'
    'Nice meet you too. How are you today?\n'
    'Great. My baseball team won the competition.\n'
    'Oh Congratulations, Juliet\n'
    'Thanks you Romeo'
)
sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')  # filter '.', ',', '?', '!'
word_list = list(set(" ".join(sentences).split()))
word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
for i, w in enumerate(word_list):
    word_dict[w] = i + 4
number_dict = {i: w for i, w in enumerate(word_dict)}
vocab_size = len(word_dict)

token_list = list()
for sentence in sentences:
    arr = [word_dict[s] for s in sentence.split()]
    token_list.append(arr)


# In[12]:


model = BERT()
criterion = nn.CrossEntropyLoss()
optimizer = nn.Adam(model.trainable_params(), learning_rate=0.001)


# In[13]:


def forward(input_ids, segment_ids, masked_pos, masked_tokens, isNext):
    logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
    loss_lm = criterion(logits_lm.swapaxes(1, 2), masked_tokens.astype(mindspore.int32))
    loss_lm = loss_lm.mean()
    loss_clsf = criterion(logits_clsf, isNext.astype(mindspore.int32))

    return loss_lm + loss_clsf


# In[14]:


grad_fn = ops.value_and_grad(forward, None, optimizer.parameters)


# In[15]:


@mindspore.jit
def train_step(input_ids, segment_ids, masked_pos, masked_tokens, isNext):
    loss, grads = grad_fn(input_ids, segment_ids, masked_pos, masked_tokens, isNext)
    optimizer(grads)
    return loss


# In[16]:


batch = make_batch()
input_ids, segment_ids, masked_tokens, masked_pos, isNext = map(mindspore.Tensor, zip(*batch))

model.set_train()
for epoch in range(100):
    loss = train_step(input_ids, segment_ids, masked_pos, masked_tokens, isNext) # for sentence classification
    if (epoch + 1) % 10 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss.asnumpy()))


# In[ ]:


# Predict mask tokens ans isNext
input_ids, segment_ids, masked_tokens, masked_pos, isNext = map(mindspore.Tensor, zip(batch[0]))
print(text)
print([number_dict[int(w.asnumpy())] for w in input_ids[0] if number_dict[int(w.asnumpy())] != '[PAD]'])

logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
logits_lm = logits_lm.argmax(2)[0].asnumpy()
print('masked tokens list : ',[pos for pos in masked_tokens[0] if pos != 0])
print('predict masked tokens list : ',[pos for pos in logits_lm if pos != 0])

logits_clsf = logits_clsf.argmax(1).asnumpy()[0]
print('isNext : ', True if isNext else False)
print('predict isNext : ',True if logits_clsf else False)

