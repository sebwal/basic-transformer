
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")
get_ipython().run_line_magic('matplotlib', 'inline')


#%%
import warnings; warnings.simplefilter('ignore')


#%%
# DELETE THIS EVENTUALLY
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

#%% [markdown]
# # Attention

#%%
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1))              / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


#%%
class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model=512, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value =             [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous()              .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


#%%
# class MultiHeadAttention(nn.Module):
#     def __init__(self, n_heads):
#         super(MultiHeadAttention, self).__init__()

#         self.linear = nn.Linear(64, 512)
#         self.wQ = nn.Linear(512, 64)
#         self.wK = nn.Linear(512, 64)
#         self.wV = nn.Linear(512, 64)

#     def forward(self, q, k, v):
#         q = self.wQ(q)
#         k = self.wK(k)
#         v = self.wV(v)

#         # split heads - I think they do this instead of a loop
#         x, attention_weights = self.applyHeads(q, k, v)
#         # transpose ?
#         # reshape ?
#         x = self.linear(x)
#         return x, attention_weights

#     def applyHeads(self, q, k, v, mask=None):
#         x = torch.matmul(q, k.permute(1, 0)) 
#         # scale x
#         # add mask
#         attention_weights = nn.Softmax(dim=-1)(x)
#         x = torch.matmul(attention_weights, v)
#         return x, attention_weights

#%% [markdown]
# # Encoder

#%%
class Encoder(nn.Module):
    def __init__(self, n_layers, n_heads):
        super(Encoder, self).__init__()
        
        self.encoderLayers = nn.ModuleList([EncoderLayer(n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(512)

    def forward(self, inputs, mask):
        x = inputs

        for layer in self.encoderLayers:
            x = layer(x, mask)
        x = self.norm(x)
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, n_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.mhattention = MultiHeadAttention(n_heads)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.feedforward = PositionwiseFeedForward(512, 2048)

    def forward(self, inputs, mask):
        x = inputs 
        z = x
        x = self.mhattention(x, x, x, mask)
        x = self.dropout1(x)
        x = z + x
        x = self.norm1(x)
        z = x
        x = self.feedforward(x)
        x = self.dropout2(x)
        x = z + x
        x = self.norm2(x) 
        return x

#%% [markdown]
# # Decoder

#%%
class Decoder(nn.Module):
    def __init__(self, n_layers, n_heads):
        super(Decoder, self).__init__()
        
        self.decoderLayers = nn.ModuleList([DecoderLayer(n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(512)

    def forward(self, inputs, encoderOut, src_mask, tgt_mask):
        x = inputs

        for layer in self.decoderLayers:
            x = layer(x, encoderOut, src_mask, tgt_mask) 

        return x

class DecoderLayer(nn.Module):
    def __init__(self, n_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.mhattention1 = MultiHeadAttention(n_heads)
        self.mhattention2 = MultiHeadAttention(n_heads)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.norm3 = nn.LayerNorm(512)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.feedforward = PositionwiseFeedForward(512, 2048)

    def forward(self, inputs, encoderOut, src_mask, tgt_mask):
        x = inputs
        z = x
        x = self.mhattention1(x, x, x, tgt_mask)
        x = self.dropout1(x)
        x = z + x        
        x = self.norm1(x)
        z = x
        x = self.mhattention2(x, encoderOut, encoderOut, src_mask) 
        x = self.dropout2(x)
        x = z + x
        x = self.norm2(x)
        z = x
        x = self.feedforward(x)
        x = self.dropout3(x)
        x = z + x
        x = self.norm3(x)
        return x

#%% [markdown]
# # Position-wise feed forward

#%%
class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim_model, dim_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(dim_model, dim_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, dim_model)

    def forward(self, inputs):
        x = inputs
        x = self.linear1(x)
        x = nn.functional.relu(x) 
        x = self.dropout(x) 
        x = self.linear2(x) 
        return x

#%% [markdown]
# # Embeddings

#%%
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

#%% [markdown]
# # Positional Encoding

#%%
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

#%% [markdown]
# # Generator ... ?

#%%
class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

#%% [markdown]
# # Transformer (Outer layer)

#%%
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, n_layers=6, n_heads=8):
        super(Transformer, self).__init__()

        self.encoder = Encoder(n_layers, n_heads)
        self.decoder = Decoder(n_layers, n_heads)
        self.src_embed = nn.Sequential(Embeddings(512, src_vocab), PositionalEncoding(512, 0.1))
        self.tgt_embed = nn.Sequential(Embeddings(512, tgt_vocab), PositionalEncoding(512, 0.1))
        self.generator = Generator(512, tgt_vocab)

#         self.embedding = nn.Embedding(TRANS_CONST['embedding_dic_size'], TRANS_CONST['embedded_vec_size'])
#         # self.posEncoding = #TODO
#         self.linear = nn.Linear(TRANS_CONST['linear_input'], TRANS_CONST['linear_output'])
#         self.softmax = nn.Softmax(dim=1)
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

    def forward(self, src, tgt, src_mask, tgt_mask):
        x = self.encoder(self.src_embed(src), src_mask)
        x = self.decoder(self.tgt_embed(tgt), x, src_mask, tgt_mask)
        return x
    
    def greedy_decode(self, src, src_mask, max_len, start_symbol):
        encoderOut = self.encoder(self.src_embed(src), src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
        for i in range(max_len-1):
#def decode(self, memory, src_mask, tgt, tgt_mask):
            tgt = Variable(ys)
            tgt_mask = Variable(subsequent_mask(ys.size(1)).type_as(src.data))
            out = self.decoder(self.tgt_embed(tgt), encoderOut, src_mask, tgt_mask) 
            prob = self.generator(out[:, -1])
            _, next_word = torch.max(prob, dim = 1)
            next_word = next_word.data[0]
            ys = torch.cat([ys, 
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        return ys    
    
    


#%%
transformer = Transformer(11, 11)

#%% [markdown]
# # Batch + Mask

#%%
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


#%%
class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask =                 self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

#%% [markdown]
# # Optimizer

#%%
class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor *             (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

#%% [markdown]
# # Label Smoothing

#%%
class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 1: # Changed > 0 to > 1 because it throws errors otherwise            
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

#%% [markdown]
# # Copy Paste Task
#%% [markdown]
# ## Loss Definition

#%%
class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        norm = norm.float() # because it throws errors if it's not casted to float
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        ret = loss.data[0] * norm
        return ret

#%% [markdown]
# ## Data Generation

#%%
def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10))).long()
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)

#%% [markdown]
# ## Training

#%%
# Train the simple copy task.
V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
transformer = Transformer(V, V, 2)
model_opt = NoamOpt(transformer.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

losses = []
for epoch in range(100):
    transformer.train()

    data_iter = data_gen(V, 30, 20)
    loss_compute = SimpleLossCompute(transformer.generator, criterion, model_opt)
#    start = time.time()
    total_tokens = 0
    total_loss = 0
#    tokens = 0
    for i, batch in enumerate(data_iter):
        out = transformer.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
#        tokens += batch.ntokens
#        if i % 50 == 1:
#            elapsed = time.time() - start
#            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" % (i, loss / batch.ntokens.float(), tokens / elapsed))
#            start = time.time()
#            tokens = 0
    loss = total_loss / total_tokens.float()
    losses.append(loss)
    transformer.eval()

    print('epoch ' + str(epoch + 1))
    print(loss)
    


#%%
plt.plot(losses)
plt.show()

#%% [markdown]
# ## Evaluation

#%%
transformer.eval()
src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]) )
#src = Variable(torch.LongTensor([[1,1, 1, 3, 3, 1, 1, 1, 4, 1]]) )
src_mask = Variable(torch.ones(1, 1, 10) )
print(transformer.greedy_decode(src, src_mask, max_len=10, start_symbol=1))


#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%



#%%


#%% [markdown]
# # CONSTANTS

#%%
import math

GLOBAL = {
    'embedded_length': 512,
    'continuity_length': 64,
    'n_vocab': 29
}

# Transformer (Top level)
TRANS_CONST = {
    'n_attention_layers': 8,
    'n_attention_heads': 8,    

    'max_output_length': 6,
    
    'embedding_dic_size': GLOBAL['n_vocab'], 
    'embedded_vec_size': GLOBAL['embedded_length'],
    
    # 'pos_encoding_input': GLOBAL['embedded_length'],
    # 'pos_encoding_output': GLOBAL['embedded_length'],
    
    'linear_input': GLOBAL['embedded_length'],
    'linear_output': GLOBAL['n_vocab'] # output vocab size
}

# Encoder, EncoderLayer
ENCODER_CONST = {
    'norm1_size': GLOBAL['embedded_length'], # same as input matrix width
    'norm2_size': GLOBAL['embedded_length'],

    # maybe rename these two, it's just for knowing the input dim and the dim that the FF layer will work with
    'ff1': GLOBAL['embedded_length'], 
    'ff2': GLOBAL['embedded_length'] * 4
}

# Decoder, DecoderLayer
DECODER_CONST = {
    'norm1_size': GLOBAL['embedded_length'], # same as input matrix width
    'norm2_size': GLOBAL['embedded_length'],
    'norm3_size': GLOBAL['embedded_length'],

    'ff1': GLOBAL['embedded_length'],#TODO RENAME
    'ff2': GLOBAL['embedded_length'] * 4#TODO RENAME
}

# MultiHeadAttention, SingleHeadAttention
ATTENTION_CONST = {
    'mh_concat_width': GLOBAL['continuity_length']*TRANS_CONST['n_attention_heads'], # single head attention width * number of heads
    'mh_output_width': GLOBAL['embedded_length'], #TODO - I'm just guessing this. Didn't see in illustrated transformer. Since we have to use this for the add & norm layer though it has to be the same as the input width (I think)

    # W_q weight matrix 
    'sh_linear1_input': GLOBAL['embedded_length'], # same as embedded length to end up with n_words x 64
    'sh_linear1_output': GLOBAL['continuity_length'], # specified in the paper
    # W_k weight matrix 
    'sh_linear2_input': GLOBAL['embedded_length'], # same as embedded length to end up with n_words x 64
    'sh_linear2_output': GLOBAL['continuity_length'], # specified in the paper
    # W_v weight matrix 
    'sh_linear3_input': GLOBAL['embedded_length'], # same as embedded length to end up with n_words x 64
    'sh_linear3_output': GLOBAL['continuity_length'], # specified in the paper
    
    'sh_scale_factor': math.sqrt(GLOBAL['continuity_length']) # specified in the paper, square root of dimension of key vector/matrix (64)
}

# FeedForward
FEEDFORWARD_CONST = {
    'dropout': 0.1
}

#%% [markdown]
# # FEEDFORWARD

#%%
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, dim_model, dim_ff, dropout=FEEDFORWARD_CONST['dropout']):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(dim_model, dim_ff)
        # self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, dim_model)

    def forward(self, inputs):
        x = inputs
        x = self.linear1(x)
        x = nn.functional.relu(x) 
        # x = self.dropout(x) 
        x = self.linear2(x) 
        return x

#%% [markdown]
# # MULTIHEADATTENTION

#%%
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.linear = nn.Linear(64, 512)
        self.wQ = nn.Linear(ATTENTION_CONST['sh_linear1_input'], ATTENTION_CONST['sh_linear1_output'])
        self.wK = nn.Linear(ATTENTION_CONST['sh_linear2_input'], ATTENTION_CONST['sh_linear2_output'])
        self.wV = nn.Linear(ATTENTION_CONST['sh_linear3_input'], ATTENTION_CONST['sh_linear3_output'])

    def forward(self, q, k, v):
        q = self.wQ(q)
        k = self.wK(k)
        v = self.wV(v)

        # split heads - I think they do this instead of a loop
        x, attention_weights = self.applyHeads(q, k, v)
        # transpose ?
        # reshape ?
        x = self.linear(x)
        return x, attention_weights

    def applyHeads(self, q, k, v, mask=None):
        x = torch.matmul(q, k.permute(1, 0)) 
        # scale x
        # add mask
        attention_weights = nn.Softmax(dim=-1)(x)
        x = torch.matmul(attention_weights, v)
        return x, attention_weights

#%% [markdown]
# # DECODER

#%%
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, n_layers, n_attention_heads):
        super(Decoder, self).__init__()
        
        #self.embedding
        #self.pos_encoding
        #self.dropout
        self.decoderLayers = nn.ModuleList([DecoderLayer(n_attention_heads) for _ in range(n_layers)])

    def forward(self, inputs, encoderOut):
        x = inputs
        #x = self.embedding(x)
        #x = x * root of dim, whatever that is exactly
        #x = self.pos_encoding(x)
        #x = self.dropout(x)

        attention_weights = []
        for layer in self.decoderLayers:
            x, att1, att2 = layer(x, encoderOut) 
            attention_weights.append([att1, att2])

        return x, attention_weights

class DecoderLayer(nn.Module):
    def __init__(self, n_attention_heads):
        super(DecoderLayer, self).__init__()

        self.mhattention1 = MultiHeadAttention(n_attention_heads)
        self.mhattention2 = MultiHeadAttention(n_attention_heads)
        self.norm1 = nn.LayerNorm(DECODER_CONST['norm1_size'])
        self.norm2 = nn.LayerNorm(DECODER_CONST['norm2_size'])
        self.norm3 = nn.LayerNorm(DECODER_CONST['norm3_size'])
        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()
        self.dropout3 = nn.Dropout()
        self.feedforward = FeedForward(DECODER_CONST['ff1'], DECODER_CONST['ff2'])

    def forward(self, inputs, encoderOut):
        x = inputs
        z = x
        x, att1 = self.mhattention1(x, x, x)
        x = self.dropout1(x)
        x = z + x        
        x = self.norm1(x)
        z = x
        x, att2 = self.mhattention2(x, encoderOut, encoderOut) 
        x = self.dropout2(x)
        x = z + x
        x = self.norm2(x)
        z = x
        x = self.feedforward(x)
        x = self.dropout3(x)
        x = z + x
        x = self.norm3(x)
        return x, att1, att2

#%% [markdown]
# # ENCODER

#%%
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, n_layer, n_attention_heads):
        super(Encoder, self).__init__()
        
        #self.embedding
        #self.pos_encoding
        #self.dropout
        self.encoderLayers = nn.ModuleList([EncoderLayer(n_attention_heads) for _ in range(n_layer)])

    def forward(self, inputs):
        x = inputs
        #x = self.embedding(x)
        #x = x * root of dim, whatever that is exactly
        #x = self.pos_encoding(x)
        #x = self.dropout(x)

        for layer in self.encoderLayers:
            x = layer(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, n_attention_heads):
        super(EncoderLayer, self).__init__()

        self.mhattention = MultiHeadAttention(n_attention_heads)
        self.norm1 = nn.LayerNorm(ENCODER_CONST['norm1_size'])
        self.norm2 = nn.LayerNorm(ENCODER_CONST['norm2_size'])
        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()
        self.feedforward = FeedForward(ENCODER_CONST['ff1'], ENCODER_CONST['ff2'])

    def forward(self, inputs):
        x = inputs 
        z = x
        x, _ = self.mhattention(x, x, x)
        x = self.dropout1(x)
        x = z + x
        x = self.norm1(x)
        z = x
        x = self.feedforward(x)
        x = self.dropout2(x)
        x = z + x
        x = self.norm2(x) 
        return x

#%% [markdown]
# # TRANSFORMER

#%%
import torch
import torch.nn as nn
import numpy

class Transformer(nn.Module):
    def __init__(self, n_layers=TRANS_CONST['n_attention_layers'], n_attention_heads=TRANS_CONST['n_attention_heads']):
        super(Transformer, self).__init__()

        self.encoder = Encoder(n_layers, n_attention_heads)
        self.decoder = Decoder(n_layers, n_attention_heads)
        self.embedding = nn.Embedding(TRANS_CONST['embedding_dic_size'], TRANS_CONST['embedded_vec_size'])
        # self.posEncoding = #TODO
        self.linear = nn.Linear(TRANS_CONST['linear_input'], TRANS_CONST['linear_output'])
        self.softmax = nn.Softmax(dim=1)

    def __call__(self, inputs=None):
        if inputs != None: 
            raise NotImplementedError

        import random
        inputs = []
        for _ in range(13): inputs.append(numpy.zeros(26)) # 26 is vocab size, should be constant; 13 is just a random amount of words in the sequence
        inputs = torch.Tensor(inputs)
        for i in inputs: i[random.randint(0, len(i) - 1)] = 1

        return self.forward(inputs.long())

    def forward(self, inputs):
        x = self.doEmbedding(inputs)
        encoderOut = self.encoder(x)
        x, weights = self.decoder(x, encoderOut)
        x, weights = self.decoder(x, encoderOut)
        x = self.linear(x)
        x = self.softmax(x)
        return x, weights

    def doEmbedding(self, inputs):
        x = inputs.nonzero()[:, 1] # this gets all indices of nonzero values from the inputs matrix
        x = self.embedding(x)
        # x = self.posEncoding(x)
        return x






#%% [markdown]
# # Check Shapes

#%%
import random
inputs = []
for _ in range(13): inputs.append(numpy.zeros(26)) # 26 is vocab size, should be constant; 13 is just a random amount of words in the sequence
inputs = torch.Tensor(inputs)
for i in inputs: i[random.randint(0, len(i) - 1)] = 1

sample_enc = Encoder(1, 1)
sample_dec = Decoder(1, 1)
sample_tf = Transformer()

#sample_x = inputs
#print(sample_x.shape)
#sample_x = sample_enc(sample_x)
#print(sample_x.shape)
#sample_x = sample_dec(sample_x)
#print(sample_x.shape)
sample_x, weights = sample_tf()
print(sample_x.shape)

#%% [markdown]
# # TRAIN

#%%
import torch 
import torch.nn as nn
import random
import numpy as np

EPOCHS = 200

transformer = Transformer()
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0002)
loss = torch.nn.BCELoss()

real_sample = torch.Tensor([
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
])
for _ in range(1000):
    sample, __ = transformer()        
    # target = torch.ones(sample.shape[0], sample.shape[1])
    error = loss(sample, real_sample)
    if _ % 10 == 0: print(error, sample)
    error.backward()
    optimizer.step()
    if(error > 8): break
print(sample)


#%%
print(torch.argmax(real_sample, dim=1))
sample, _ = transformer()
print(torch.argmax(sample, dim=1))


#%%
print(__)


#%%



