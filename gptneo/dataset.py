# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import gc
import shutil
import json

import numpy as np
import torch


class JavafinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, filename, max_length=512,
                 test=False, partition=-1):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.partition = partition
        if not test and 'test' in filename:
            self.if_eval = True
        else:
            self.if_eval = False
        self.test = test
        self.examples = self.read_examples(filename)

    def read_examples(self, filename):
        """Read examples from filename."""
        examples = {'source': [], 'target': []}
        assert len(filename.split(','))==2
        src_filename = filename.split(',')[0]
        trg_filename = filename.split(',')[1]
        with open(src_filename) as f1,open(trg_filename) as f2:
            for line1,line2 in zip(f1,f2):
                examples['source'].append(line1.strip()),
                examples['target'].append(line2.strip()),
        if self.if_eval:
            examples['source'] = examples['source'][:500]
            examples['target'] = examples['target'][:500]
        elif self.test and self.partition != -1:
            start = self.partition * 50000
            #end = (1 + self.partition) * 50000
            if self.partition == 3:
                end = len(examples['target'])
            else:
                end = (1 + self.partition) * 50000
            examples['source'] = examples['source'][start:end]
            examples['target'] = examples['target'][start:end]
        return examples

    def pack_samples(self, idx):
        """
        Repeatedly pick question, answer pairs from self.dataroot until we hit max_tokens.
        This will not include the tokens for the QUESTION and ANSWER prompt, as well as the
        self.question_prefix. These will be added later and the total input will be
        truncated if necessary.
        Always include the sample at idx at the beginning.
        """
        curr_num_tokens = 0
        curr_samples = []
        curr_q, curr_a = self.examples['source'][idx], self.examples['target'][idx]

        if self.test:
            curr_samples.append((curr_q, curr_a))
            return curr_samples

        while curr_num_tokens < self.max_length:

            # Never remove. Fixes stalling bug.
            curr_q = curr_q[:150000]
            curr_a = curr_a[:150000]

            curr_num_tokens += len(self.tokenizer.tokenize(curr_q))
            curr_num_tokens += len(self.tokenizer.tokenize(curr_a))

            curr_samples.append((curr_q, curr_a))

            random_idx = random.choice(range(len(self.examples['target'])))
            curr_q = self.examples['source'][random_idx]
            curr_a = self.examples['target'][random_idx]

        return curr_samples

    def __len__(self):
        return min(len(self.examples['source']),
                   len(self.examples['target']))

    def __getitem__(self, idx):
        input_ids = []
        label_ids = []

        raw_samples = self.pack_samples(idx)
        for q_str, a_str in raw_samples:
            q_str = self.tokenizer.cls_token + self.examples['source'][idx] + \
                self.tokenizer.sep_token
            a_str = self.examples['target'][idx]
            question_token_ids = self.tokenizer.encode(q_str, verbose=False)
            answer_token_ids = self.tokenizer.encode(a_str, verbose=False)
            answer_token_ids.append(self.tokenizer.eos_token_id)
            input_ids.extend(question_token_ids)
            input_ids.extend(answer_token_ids)
            label_ids.extend([-100] * len(question_token_ids))
            label_ids.extend(answer_token_ids)

        if self.test:
            return {'q_str': q_str}
        else:
            # Cut off the excess
            input_ids = input_ids[:self.max_length]
            label_ids = label_ids[:self.max_length]

            padding_length = self.max_length - len(input_ids)
            if padding_length > 0:
                input_ids += [self.tokenizer.pad_token_id] * padding_length
                label_ids += [self.tokenizer.pad_token_id] * padding_length

            retval = {
                "input_ids" : torch.LongTensor(input_ids),
                "labels" :  torch.LongTensor(label_ids)
            }
            gc.collect()
            return retval


if __name__ == '__main__':
    import argparse
    import transformers

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    data_prefix=''
    args.trainfile='{}/context_data.final,{}/body_data.final'.format(data_prefix, data_prefix)
    args.devfile='{}/context_data.final.100,{}/body_data.final.100'.format(data_prefix, data_prefix)

    data_prefix=''
    args.trainfile='{}/train/context_data.final,{}/train/body_data.final'.format(data_prefix, data_prefix)
    args.devfile='{}/test/context_data.final,{}/test/body_data.final'.format(data_prefix, data_prefix)
    args.output_dir = 'save/c2c_data_gptneo'
    tokenizer = transformers.GPT2Tokenizer.from_pretrained(
        'EleutherAI/gpt-neo-125M',
        do_lower_case=False,
        sep_token='</s>', cls_token='<s>',
        pad_token='<pad>', unk_token='<|UNKNOWN|>')
    dataset = JavafinetuneDataset(tokenizer, args.devfile, max_length=1024)
    train_dataset = JavafinetuneDataset(tokenizer, args.trainfile, max_length=1024)

    e = dataset[0]
    print(e)
    print("------- input_ids ------------------------------------------------------------------------------------")
    print(tokenizer.decode(e['input_ids']))
    print("------- labels ------------------------------------------------------------------------------------")
    labels = e['labels']
    labels[labels == -100] = tokenizer.eos_token_id
    labels_str = tokenizer.decode(labels)
    print(labels_str)

    import pdb; pdb.set_trace()
