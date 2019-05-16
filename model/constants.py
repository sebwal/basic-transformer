import math

GLOBAL = {
    'embedded_length': 512,
    'continuity_length': 64
}

# Transformer (Top level)
TRANS_CONST = {
    'n_attention_layers': 8,
    'n_attention_heads': 8,    
    
    'embedding_dic_size': 1000,#TODO 
    'embedded_vec_size': GLOBAL['embedded_length'],
    
    # 'pos_encoding_input': GLOBAL['embedded_length'],
    # 'pos_encoding_output': GLOBAL['embedded_length'],
    
    'linear_input': GLOBAL['embedded_length'],
    'linear_output': GLOBAL['embedded_length']
}

# Encoder, EncoderLayer
ENCODER_CONST = {
    'norm1_size': GLOBAL['embedded_length'], # same as input matrix width
    'norm2_size': GLOBAL['embedded_length'],

    # maybe rename these two, it's just for knowing the input dim and the dim that the FF layer will work with
    'ff1': GLOBAL['embedded_length'], 
    'ff2': GLOBAL['continuity_length']
}

# Decoder, DecoderLayer
DECODER_CONST = {
    'norm1_size': GLOBAL['embedded_length'], # same as input matrix width
    'norm2_size': GLOBAL['embedded_length'],
    'norm3_size': GLOBAL['embedded_length'],

    'ff1': GLOBAL['embedded_length'],#TODO RENAME
    'ff2': GLOBAL['continuity_length']#TODO RENAME
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
    
    'sh_scale_factor': 1/math.sqrt(GLOBAL['continuity_length']) # specified in the paper, square root of dimension of key vector/matrix (64)
}

# FeedForward
FEEDFORWARD_CONST = {
    'dropout': 0.1
}

