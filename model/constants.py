# Transformer (Top level)
TRANS_CONST = {
    'n_attention_layers': 1,
    'n_attention_heads': 1,    
    'embedding_dic_size': 1000,#TODO 
    'embedded_vec_size': 512,#TODO
    'pos_encoding_input': 512,#TODO
    'pos_encoding_output': 512,#TODO
    'linear_input': 512,#TODO
    'linear_output': 1#TODO
}

# Encoder, EncoderLayer
ENCODER_CONST = {
    'norm1_size': 1,#TODO
    'norm2_size': 1,#TODO
    'ff1': 1,#TODO, also RENAME
    'ff2': 1#TODO, also RENAME
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
    'mh_concat_width': 1,#TODO
    'mh_output_width': 1,#TODO 
    'mh_linear2_input': 512,#TODO
    'mh_linear2_output': 1,#TODO
    'sh_linear1_input': 512,#TODO
    'sh_linear1_output': 1,#TODO
    'sh_linear2_input': 512,#TODO
    'sh_linear2_output': 1,#TODO
    'sh_linear3_input': 512,#TODO
    'sh_linear3_output': 1,#TODO
    'sh_scale_factor': 1/8
}

# FeedForward
FEEDFORWARD_CONST = {
}

