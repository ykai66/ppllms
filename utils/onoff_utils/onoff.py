from utils.onoff_utils import onoff_llama
from utils.onoff_utils import bi_utils, on_off_llama_final

def block_replace(model):
    if 'llama' in model.name.lower():
        model = onoff_llama.block_replace(model) #!!!!
        print("replace to onoff_llama")

    return model

def block_replace_bi(model):
    if 'llama' in model.name.lower():
        model = bi_utils.block_replace(model) #!!!!

    return model

def final_replace(model):
    model = on_off_llama_final.block_replace(model)
    print("replace to final_replace")
    return model

def final_turn_off_ffn(model, block_idx):
    on_off_llama_final.turn_off_ffn(model, block_idx)

def block_replace_final(model):
    model = on_off_llama_final.block_replace(model)
    return model

def turn_off_layer_final(model, block_idx):
    on_off_llama_final.turn_off_layer(model, block_idx)

def turn_on_layer_final(model, block_idx):
    on_off_llama_final.turn_on_layer(model, block_idx)

def turn_off_ffn_final(model, block_idx):
    on_off_llama_final.turn_off_ffn(model, block_idx)

def turn_on_ffn_final(model, block_idx):
    on_off_llama_final.turn_on_ffn(model, block_idx)
