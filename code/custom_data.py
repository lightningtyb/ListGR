#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   custom_data.py    
@Contact :   lightningtyb@163.com
@License :   (C)Copyright 2019-2020

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/6/1 22:36   tangyubao      1.0         None
'''

# import lib
import json
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from transformers import PreTrainedTokenizer, DataCollatorWithPadding
from typing import List, Iterator

class CustomDataset(Dataset):
    def __init__(self, file_path):
        self.data = self.load_data(file_path)

    def load_data(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            data = [json.loads(line) for line in lines]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        print(index)
        src = item['translation']['src']
        tgt = item['translation']['tgt']
        return src, tgt

import torch
from torch.utils.data import Sampler, RandomSampler

import random
class CustomSampler(Sampler[List[int]]):
    def __init__(self, data_source, grades):
        self.data_source = data_source
        self.grades = grades

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        multiple_of_three_indices = [i for i in indices if i % self.grades == 0]
        random.shuffle(multiple_of_three_indices)
        for idx in multiple_of_three_indices:
            yield idx
            # if idx % 3 == 0 and idx + 2 < len(self.data_source):
            for i in range(1, self.grades):
                yield idx + i

    def __len__(self):
        return len(self.data_source)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    # 进行自定义的数据处理操作
    # ...

    return src_batch, tgt_batch

class IndexingCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        docids = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        labels = self.tokenizer(
            docids, padding="longest", return_tensors="pt"
        ).input_ids

        # replace padding token id's of the labels by -100 according to https://huggingface.co/docs/transformers/model_doc/t5#training
        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels
        # qids = [x[2] for x in features]
        # inputs['qids'] = qids
        return inputs




# 假设有一个名为"data.jsonl"的jsonl文件
file_path = "data.jsonl"

# 创建自定义的数据集
dataset = CustomDataset(file_path)

# 创建自定义的采样器
sampler = CustomSampler(dataset, 3)

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=32, sampler=sampler, collate_fn=collate_fn)

# 遍历数据加载器中的每个数据
for batch in dataloader:
    # 进行训练或其他操作
    src_batch, tgt_batch = batch
    print(src_batch)
    print(tgt_batch)
