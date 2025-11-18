import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


def get_tokenizer(model):
    if "llama" in model.lower():
        tokenizer = LlamaTokenizer.from_pretrained(model, use_fast=False)
        # fix for transformer 4.28.0.dev0 compatibility
        if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
            try:
                tokenizer.bos_token_id = 1
                tokenizer.eos_token_id = 2
            except AttributeError:
                pass
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    return tokenizer


def get_wikitext2(nsamples, seed, seqlen, model, tokenizer, batch_size):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    new_trainloader = []
    num_batches = nsamples // batch_size + (int)(nsamples % batch_size > 0)
    for i in range(0, num_batches):
        start = i * batch_size
        end = min(start + batch_size, nsamples)
        batched_inp = []
        batched_tar = []
        for j in range(start, end):
            batched_inp.append(trainloader[j][0])
            batched_tar.append(trainloader[j][1])
        batched_inp = torch.cat(batched_inp)
        batched_tar = torch.cat(batched_tar)
        new_trainloader.append((batched_inp, batched_tar))
    del trainloader
    trainloader = new_trainloader
    del new_trainloader

    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, model, tokenizer, batch_size):
    traindata = load_dataset(
        '/workspace/datasets/allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset('/workspace/datasets/allenai/c4',
                           data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    new_trainloader = []
    num_batches = nsamples // batch_size + (int)(nsamples % batch_size > 0)
    for i in range(0, num_batches):
        start = i * batch_size
        end = min(start + batch_size, nsamples)
        batched_inp = []
        batched_tar = []
        for j in range(start, end):
            batched_inp.append(trainloader[j][0])
            batched_tar.append(trainloader[j][1])
        batched_inp = torch.cat(batched_inp)
        batched_tar = torch.cat(batched_tar)
        new_trainloader.append((batched_inp, batched_tar))
    del trainloader
    trainloader = new_trainloader
    del new_trainloader

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None, model='', batch_size=1):
    if tokenizer is None:
        tokenizer = get_tokenizer(model)
    # if 'wikitext2' in name:
    if 'wikitext' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, tokenizer, batch_size)
    if 'c4' in name:
        return get_c4(nsamples, seed, seqlen, model, tokenizer, batch_size)


def get_wikitext2_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size):
    traindata = load_dataset('/workspace/datasets/Salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
    traindata = traindata.shuffle(seed=seed)
    trainenc = tokenizer("\n\n".join(traindata[:nsamples]['text']), return_tensors='pt')

    return trainenc


def get_c4_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size):
    traindata = load_dataset(
        '/workspace/datasets/allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset('/workspace/datasets/allenai/c4',
                           data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
    traindata = traindata.shuffle(seed=seed)

    trainenc = tokenizer(' '.join(traindata[:nsamples]['text']), return_tensors='pt')
    trainenc = trainenc.input_ids

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    trainenc = TokenizerWrapper(trainenc)

    return trainenc


def get_c4_valenc(seed, nsamples, seqlen, model, tokenizer, batch_size):
    valdata = load_dataset('/workspace/datasets/allenai/c4',
                           data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
    valdata = valdata.shuffle(seed=seed)
    valenc = tokenizer(' '.join(valdata[nsamples]['text']), return_tensors='pt')
    valenc = valenc.input_ids

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)
    return valenc


def get_trainloaders(name, nsamples=128, seed=0, seqlen=2048, model='', batch_size=1):
    tokenizer = get_tokenizer(model)
    # if 'wikitext2' in name:
    if 'wikitext' in name:
        return get_wikitext2_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size)
    if 'c4' in name:
        return get_c4_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size)

def get_trainloaders_v2(name, nsamples=128, seed=0, seqlen=2048, model='', batch_size=1, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer(model)
    if 'wikitext' in name:
        return get_wikitext2_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size, device)
    if 'c4' in name:
        return get_c4_trainenc_v2(seed, nsamples, seqlen, model, tokenizer, batch_size, device)

def get_c4_trainenc_v2(seed, nsamples, seqlen, model, tokenizer, batch_size, device):
    traindata = load_dataset(
        '/workspace/datasets/allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset('/workspace/datasets/allenai/c4',
                           data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
    traindata = traindata.shuffle(seed=seed)

    trainenc = tokenizer(' '.join(traindata[:nsamples]['text']), return_tensors='pt')
    trainenc = trainenc.input_ids

    # 将 trainenc 移动到指定设备
    trainenc = trainenc.to(device)

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    trainenc = TokenizerWrapper(trainenc)

    return trainenc

def get_valloaders(name,
                   seed=0,
                   nsamples=128,
                   seqlen=2048,
                   batch_size=1,
                   model=''  # 空字符串，用于参数传递。
                   ):
    tokenizer = get_tokenizer(model)
    if "c4" in name:
        return get_c4_valenc(seed, nsamples, seqlen, model, tokenizer, batch_size)
