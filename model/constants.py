# Transformer (Top level)
TRANS_CONST = {
    'n_attention_layers': 1,
    'n_attention_heads': 8,    
    'embedding_dic_size': 1000,#TODO 
    'embedded_vec_size': 512,#TODO
    'pos_encoding_input': 512,#TODO
    'pos_encoding_output': 512,#TODO
    'linear_input': 512,#TODO
    'linear_output': 1#TODO
}

# Encoder, EncoderLayer
ENCODER_CONST = {
    'norm1_size': 512, # same as input matrix width
    'norm2_size': 512,

    # maybe rename these two, it's just for knowing the input dim and the dim that the FF layer will work with
    'ff1': 512, 
    'ff2': 64
}

# Decoder, DecoderLayer
DECODER_CONST = {
    'norm1_size': ENCODER_CONST['norm1_size'],
    'norm2_size': ENCODER_CONST['norm2_size'],
    'ff1': ENCODER_CONST['ff1'],#TODO RENAME
    'ff2': ENCODER_CONST['ff2']#TODO RENAME
}

# MultiHeadAttention, SingleHeadAttention
ATTENTION_CONST = {
    'mh_concat_width': 64*8, # single head attention width * number of heads
    'mh_output_width': 512, #TODO - I'm just guessing this. Didn't see in illustrated transformer. Since we have to use this for the add & norm layer though it has to be the same as the input width (I think)

    # W_q weight matrix 
    'sh_linear1_input': 512, # same as embedded length to end up with n_words x 64
    'sh_linear1_output': 64, # specified in the paper
    # W_k weight matrix 
    'sh_linear2_input': 512, # same as embedded length to end up with n_words x 64
    'sh_linear2_output': 64, # specified in the paper
    # W_v weight matrix 
    'sh_linear3_input': 512, # same as embedded length to end up with n_words x 64
    'sh_linear3_output': 64, # specified in the paper
    
    'sh_scale_factor': 1/8 # specified in the paper, square root of dimension of key vector/matrix (64)
}

# FeedForward
FEEDFORWARD_CONST = {
    'dropout': 0.1
}

