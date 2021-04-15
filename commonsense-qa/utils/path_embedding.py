import argparse
import os
import time
import random
import numpy as np 
import logging
import sys
import json
import math
from tqdm import tqdm, trange
import pickle

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler

# EDITED
import transformers
assert transformers.__version__ == '2.8.0'
from transformers import GPT2Config, GPT2Tokenizer, GPT2Model

def _get_path_embedding_greedy(dataset, generator, args, tokenizer=None, output_file=None):
    data_sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=data_sampler, batch_size=args.batch_size)
    generator.eval()
    epoch_iterator = tqdm(dataloader, desc="Path Generation")
    path_embeddings = []

    lm_type = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(lm_type)
    tokenizer.add_tokens(['<PAD>'])
    tokenizer.add_tokens(['<SEP>'])
    tokenizer.add_tokens(['<END>'])

    count1 = 0
    count2 = 0
    count3 = 0

    path_array = []

    for step, context in tqdm(enumerate(epoch_iterator)):

        # questions, contexts, answers, choices = batch
        context = context[0].to(args.device)

        with torch.no_grad():
            batch_size, num_choice, num_context, context_len = context.size()
            context = context.view(-1, context_len)
            context_embedding, generated_paths = generator(context, train=False, return_path=True)
            context_embedding = context_embedding.view(batch_size, num_choice, num_context, -1)

        if not output_file is None:
            for i, path in enumerate(generated_paths):
                if count3 % 5 == 0 and count3 != 0:
                    count2 += 1
                    count3 = 0

                if count2 % 5 == 0 and count2 != 0:
                    count2 = 0
                    count1 += 1

                path = tokenizer.decode(path.tolist(), skip_special_tokens=True)
                path = ' '.join(path.replace('<PAD>', '').split())
                # print(count1, count2, count3, path)
                with open(output_file, 'a') as fout:
                    fout.write(path+'\n')
                
                count3 += 1
                path_array.append(path)

        path_embeddings.extend(context_embedding.tolist())
    
    # path_array = torch.tensor(path_array)
    # path_array = path_array.view(dataset.size(), num_choice, num_context, -1)
    # path_array_file = os.path.join('./path_embeddings/', args.data_dir, 'path_array.pickle')
    # with open(path_array_file, 'wb') as handle:
    #     pickle.dump(path_array, handle, protocol=pickle.HIGHEST_PROTOCOL)

    path_embeddings = torch.tensor(path_embeddings, dtype=torch.float)
    return path_embeddings

def save_path_embedding(datahelper, generator, save_file, args):
    path_embeddings_dict = {}
    path_embeddings_dict['train'] = _get_path_embedding_greedy(datahelper.trainset, generator, args, output_file='./generated_path_original.log')
    path_embeddings_dict['dev'] = _get_path_embedding_greedy(datahelper.devset, generator, args)
    path_embeddings_dict['test'] = _get_path_embedding_greedy(datahelper.testset, generator, args)

    with open(save_file, 'wb') as handle:
        pickle.dump(path_embeddings_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

