from dataclasses import dataclass
from tqdm import tqdm
import datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, DataCollatorWithPadding
import collections
import jsonlines
import torch
import random
from typing import List, Iterator
from torch.utils.data import Sampler
import math

class IndexingTrainDataset(Dataset):
    def __init__(
            self,
            path_to_data,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
            remove_prompt=False,
    ):
        self.train_data = datasets.load_dataset(
            'json',
            data_files=path_to_data,
            ignore_verifications=False,
            cache_dir=cache_dir
        )['train']

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.remove_prompt = remove_prompt
        self.total_len = len(self.train_data)
        self.valid_ids = set()
        for data in tqdm(self.train_data):
            self.valid_ids.add(str(data['text_id']))

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        data = self.train_data[item]
        if self.remove_prompt:
            data['text'] = data['text'][9:] if data['text'].startswith('Passage: ') else data['text']
            data['text'] = data['text'][10:] if data['text'].startswith('Question: ') else data['text']
        input_ids = self.tokenizer(data['text'],
                                   return_tensors="pt",
                                   truncation='only_first',
                                   max_length=self.max_length).input_ids[0]
        return input_ids, str(data['text_id'])


class GenerateDataset(Dataset):
    lang2mT5 = dict(
        ar='Arabic',
        bn='Bengali',
        fi='Finnish',
        ja='Japanese',
        ko='Korean',
        ru='Russian',
        te='Telugu'
    )

    def __init__(
            self,
            path_to_data,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
    ):
        self.data = []
        with open(path_to_data, 'r') as f:
            for data in f:
                if 'xorqa' in path_to_data:
                    docid, passage, title = data.split('\t')
                    for lang in self.lang2mT5.values():
                        self.data.append((docid, f'Generate {lang} question: {title}</s>{passage}'))
                elif 'msmarco' in path_to_data:
                    docid, passage = data.split('\t')
                    self.data.append((docid, f'{passage}'))
                else:
                    raise NotImplementedError(f"dataset {path_to_data} for docTquery generation is not defined.")

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.total_len = len(self.data)


    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        docid, text = self.data[item]
        input_ids = self.tokenizer(text,
                                   return_tensors="pt",
                                   truncation='only_first',
                                   max_length=self.max_length).input_ids[0]
        return input_ids, int(docid)


class EvalDataset(Dataset):

    def __init__(
            self,
            path_to_data,
            path_did2queryID,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
    ):
        self.data = []
        self.src_id = []
        self.tgt_id = []
        self.docid2did = {}
        # with open(path_to_data, 'r') as f:
        #     for data in f:
        #         if 'xorqa' in path_to_data:
        #             docid, passage, title = data.split('\t')
        #             for lang in self.lang2mT5.values():
        #                 self.data.append((docid, f'Generate {lang} question: {title}</s>{passage}'))
        #         elif 'msmarco' in path_to_data:
        #             docid, passage = data.split('\t')
        #             self.data.append((docid, f'{passage}'))
        #         else:
        #             raise NotImplementedError(f"dataset {path_to_data} for docTquery generation is not defined.")
        self.mode = 'test'

        with open(path_to_data, 'r') as fin:
            for line in jsonlines.Reader(fin):
                src_id, tgt_id = line['translation']['src-id'], line['translation']['tgt-id']
                src_prefix, src, tgt = line['translation']['prefix'], line['translation']['src'], line['translation'][
                    'tgt']
                self.data.append((tgt, src))
                self.src_id.append(src_id)
                self.tgt_id.append(tgt_id)
        self.docid2did = load_hash2id_all(path_did2queryID)


        self.max_length = max_length
        self.tokenizer = tokenizer
        self.total_len = len(self.data)
        if self.mode == 'test':
            self.src, self.qids = load_dev(path_to_data)
        self.docid2did = load_hash2id_all(path_did2queryID)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        docid, text = self.data[item]
        input_ids = self.tokenizer(text,
                                   return_tensors="pt",
                                   truncation='only_first',
                                   max_length=self.max_length).input_ids[0]
        # docid = self.tokenizer(docid,
        #                            return_tensors="pt",
        #                            truncation='only_first',
        #                            max_length=self.max_length).input_ids[0]
        return input_ids, docid, self.mode




class TranslationTrainDataset(Dataset):
    def __init__(
            self,
            path_to_data,
            path_did2queryID,
            mode,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
            remove_prompt=False,
    ):
        self.train_data = datasets.load_dataset(
            'json',
            data_files=path_to_data,
            ignore_verifications=False,
            cache_dir=cache_dir
        )['train']

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.remove_prompt = remove_prompt
        self.total_len = len(self.train_data)
        self.valid_ids = set()
        self.query2qid = {}
        self.docid2did = {}
        self.src = []
        self.qids = []

        # for data in tqdm(self.train_data):
        #     self.valid_ids.add(str(data['translation']['tgt']))
        # print(f"valid ids")
        # print(self.valid_ids)

        if mode == 'test':
            self.src, self.qids = load_dev(path_to_data)
        self.docid2did = load_hash2id_all(path_did2queryID)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        data = self.train_data[item]
        src = data['translation']['prefix'] + ": " + data['translation']['src']
        input_ids = self.tokenizer(src,
                                   return_tensors="pt",
                                   truncation='only_first',
                                   max_length=self.max_length).input_ids[0]
        label = data['translation']['tgt']
        src_id = data['translation']['src-id']
        # label_ids = self.tokenizer(label, return_tensors="pt",).label_ids[0]

        return input_ids, label, src_id




class DiscriminationTrainDataset(Dataset):
    def __init__(
            self,
            path_to_data,
            path_did2queryID,
            mode,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
            remove_prompt=False,
    ):
        self.train_data = datasets.load_dataset(
            'json',
            data_files=path_to_data,
            ignore_verifications=False,
            cache_dir=cache_dir
        )['train']

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.remove_prompt = remove_prompt
        self.total_len = len(self.train_data)
        self.valid_ids = set()
        self.query2qid = {}
        self.docid2did = {}
        self.src = []
        self.qids = []

        # for data in tqdm(self.train_data):
        #     self.valid_ids.add(str(data['translation']['tgt']))
        # print(f"valid ids")
        # print(self.valid_ids)

        if mode == 'test':
            self.src, self.qids = load_dev(path_to_data)
            # for data in tqdm(self.train_data):
        self.docid2did = load_hash2id_all(path_did2queryID)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        # print(f'--item {item}')


        data = self.train_data[item]
        src = data['translation']['prefix'] + ": " + data['translation']['src']
        input_ids = self.tokenizer(src,
                                   return_tensors="pt",
                                   truncation='only_first',
                                   max_length=self.max_length).input_ids[0]
        label = data['translation']['tgt']
        src_id = data['translation']['src-id']
        # label_ids = self.tokenizer(label, return_tensors="pt",).label_ids[0]

        return input_ids, label, src_id


class DiscriminationMixDataset(Dataset):
    def __init__(
            self,
            path_to_data,
            path_did2queryID,
            mode,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
            remove_prompt=False,
    ):
        self.train_data = datasets.load_dataset(
            'json',
            data_files=path_to_data,
            ignore_verifications=False,
            cache_dir=cache_dir
        )['train']

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.remove_prompt = remove_prompt
        self.total_len = len(self.train_data)
        self.valid_ids = set()
        self.query2qid = {}
        self.docid2did = {}
        self.src = []
        self.qids = []

        # for data in tqdm(self.train_data):
        #     self.valid_ids.add(str(data['translation']['tgt']))
        # print(f"valid ids")
        # print(self.valid_ids)

        if mode == 'test':
            self.src, self.qids = load_dev(path_to_data)
            # for data in tqdm(self.train_data):
        self.docid2did = load_hash2id_all(path_did2queryID)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        # print(f'--item {item}')


        data = self.train_data[item]
        src = data['translation']['prefix'] + ": " + data['translation']['src']
        input_ids = self.tokenizer(src,
                                   return_tensors="pt",
                                   truncation='only_first',
                                   max_length=self.max_length).input_ids[0]
        label = data['translation']['tgt']
        src_id = data['translation']['src-id']
        type = data['translation']['prefix']
        # label_ids = self.tokenizer(label, return_tensors="pt",).label_ids[0]

        return input_ids, label, src_id, type

class MGCCTrain11Dataset(Dataset):
    def __init__(
            self,
            path_to_data,
            mode,
            dataset_batchsize: int,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
            remove_prompt=False,
    ):
        self.train_data = datasets.load_dataset(
            'json',
            data_files=path_to_data,
            ignore_verifications=False,
            cache_dir=cache_dir
        )['train']
        self.path_data = path_to_data
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.remove_prompt = remove_prompt
        self.total_len = len(self.train_data)
        self.valid_ids = set()
        self.query2qid = {}
        self.docid2did = {}
        self.src = []
        self.qids = []
        self.mode = mode
        self.typelen = 0
        self.dataset_batchsize = 0
        if 'indexing' in path_to_data:
            self.contrastive_type = 'indexing'
            self.typelen = 1
            self.dataset_batchsize = dataset_batchsize
        elif '11' in path_to_data and '1111' not in path_to_data and '1122' not in path_to_data:
            self.contrastive_type = '11'
            self.typelen = 1
        print(f'--mgcc 11 dataset init contrastive type:{self.contrastive_type}, typelen:{self.typelen}')

        if self.mode == 'test':
            self.src, self.qids = load_dev(path_to_data)
        # self.docid2did = load_hash2id_all(path_did2queryID)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        # print(f'--mgccdataset item:{item}, mode:{self.mode}')
        data = self.train_data[item]
        src = data['translation']['prefix'] + ": " + data['translation']['src']
        input_ids = self.tokenizer(src,
                                   return_tensors="pt",
                                   truncation='only_first',
                                   max_length=self.max_length).input_ids[0]
        label = data['translation']['tgt']
        # print(f'--mgccdataset getitem:{item}, con tyep:{self.contrastive_type}, mode:{self.mode}')


        return input_ids, label, self.max_length, self.contrastive_type, self.mode


class MGCCTrainMultiDataset(Dataset):
    def __init__(
            self,
            path_to_data,
            mode,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
            remove_prompt=False,
    ):
        self.train_data = datasets.load_dataset(
            'json',
            data_files=path_to_data,
            ignore_verifications=False,
            cache_dir=cache_dir
        )['train']
        self.path_data = path_to_data
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.remove_prompt = remove_prompt
        self.total_len = len(self.train_data)
        self.valid_ids = set()
        self.query2qid = {}
        self.docid2did = {}
        self.src = []
        self.qids = []
        self.mode = mode
        self.typelen = 0
        self.dataset_batchsize = 0

        if '1222' in path_to_data:
            self.contrastive_type = '1222'
            self.typelen = 7
        elif '1122' in path_to_data:
            self.contrastive_type = '1122'
            self.typelen = 6
        elif '1111' in path_to_data:
            self.contrastive_type = '1111'
            self.typelen = 4
        print(f'--mgcc multi dataset init contrastive type:{self.contrastive_type}, typelen:{self.typelen}')

        # for data in tqdm(self.train_data):
        #     self.valid_ids.add(str(data['translation']['tgt']))
        # print(f"valid ids")
        # print(self.valid_ids)

        if self.mode == 'test':
            self.src, self.qids = load_dev(path_to_data)
        # self.docid2did = load_hash2id_all(path_did2queryID)

    def __len__(self):
        # num_per_group = 0
        # for i in self.contrastive_type:
        #     num_per_group += int(i)
        # print(f'---mgcc dataset datapath:{self.path_data}, type:{self.contrastive_type}, num per group:{num_per_group}, totol:{self.total_len* num_per_group}')
        # return self.total_len * num_per_group
        return self.total_len
    def __getitem__(self, item):
        # print(f'--mgccdataset item:{item}, mode:{self.mode}')
        data = self.train_data[item]
        # print('--mgccdataset data:', data)
        src_ids, tgt = self.getitem_train(data)

        return src_ids, tgt, self.max_length, self.contrastive_type, self.mode

    def getitem_train(self, data):
        src_4grade, tgt_4grade = data['4-grade']['src'], data['4-grade']['tgt']
        src_3grade_1, tgt_3grade_1 = data['3-grade']['1']['src'], data['3-grade']['1']['tgt']
        src_3grade_2, tgt_3grade_2 = data['3-grade']['2']['src'], data['3-grade']['2']['tgt']
        src_2grade_1, tgt_2grade_1 = data['2-grade']['1']['src'], data['2-grade']['1']['tgt']
        src_2grade_2, tgt_2grade_2 = data['2-grade']['2']['src'], data['2-grade']['2']['tgt']
        src_1grade_1, tgt_1grade_1 = data['1-grade']['1']['src'], data['1-grade']['1']['tgt']
        src_1grade_2, tgt_1grade_2 = data['1-grade']['2']['src'], data['1-grade']['2']['tgt']
        src = [src_4grade,
               src_3grade_1, src_3grade_2,
               src_2grade_1, src_2grade_2,
               src_1grade_1, src_1grade_2]
        tgt = [tgt_4grade,
               tgt_3grade_1, tgt_3grade_2,
               tgt_2grade_1, tgt_2grade_2,
               tgt_1grade_1, tgt_1grade_2]

        # print(f'---mgcc dataset getitem all data:{data}')
        # print(f'---mgcc dataset getitem all src:{src}')
        # if self.contrastive_type == '1111':
        #     print(f'------------------mgcc dataset 11111')
        #     print(f'------------------mgcc dataset src len:{len(src)}, tgt len{tgt}')
        #     print(f'------------------mgcc dataset src:{src}')


        if self.contrastive_type == '1122':
            del src[2]
            del tgt[2]
        elif self.contrastive_type == '1111': # delete index at 2, 4, 6
            del src[2]
            del tgt[2]
            del src[3]
            del tgt[3]
            del src[4]
            del tgt[4]
        return src, tgt


class MGCCTrain55MultiDataset(Dataset):
    def __init__(
            self,
            path_to_data,
            mode,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
            remove_prompt=False,
    ):
        self.train_data = datasets.load_dataset(
            'json',
            data_files=path_to_data,
            ignore_verifications=False,
            cache_dir=cache_dir
        )['train']
        self.path_data = path_to_data
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.remove_prompt = remove_prompt
        self.valid_ids = set()
        self.query2qid = {}
        self.docid2did = {}
        self.src = []
        self.qids = []
        self.mode = mode
        self.typelen = 0
        self.dataset_batchsize = 0

        if '1222' in path_to_data:
            self.contrastive_type = '1222'
            self.typelen = 7
        elif '1122' in path_to_data:
            self.contrastive_type = '1122'
            self.typelen = 6
        elif '1111' in path_to_data:
            self.contrastive_type = '1111'
            self.typelen = 4
        elif '55' in path_to_data:
            self.contrastive_type = '55'
            self.typelen = 10
            self.grades = 2
        print(f'--mgcc multi dataset init contrastive type:{self.contrastive_type}, typelen:{self.typelen}')
        self.total_len = len(self.train_data)

        # for data in tqdm(self.train_data):
        #     self.valid_ids.add(str(data['translation']['tgt']))
        # print(f"valid ids")
        # print(self.valid_ids)

        if self.mode == 'test':
            self.src, self.qids = load_dev(path_to_data)
        # self.docid2did = load_hash2id_all(path_did2queryID)

    def __len__(self):
        # num_per_group = 0
        # for i in self.contrastive_type:
        #     num_per_group += int(i)
        # print(f'---mgcc dataset datapath:{self.path_data}, type:{self.contrastive_type}, num per group:{num_per_group}, totol:{self.total_len* num_per_group}')
        # return self.total_len * num_per_group
        return self.total_len
    def __getitem__(self, item):
        # print(f'--mgccdataset item:{item}, mode:{self.mode}')
        data = self.train_data[item]
        # print('--mgccdataset data:', data)
        src_ids, tgt = self.getitem_train(data)

        return src_ids, tgt, self.max_length, self.contrastive_type, self.mode

    def getitem_train(self, data):
    #     print(f' line has grades:{data.keys()}')
        keys = data.keys()
        src, tgt = [], []
        for k in keys:
            # print(f'---k:{k}')
            inner_data = data[k]
            inner_keys = inner_data.keys()
            for ik in inner_keys:
                # print(f'--ik:{ik}')
                src.append(inner_data[ik]['src'])
                tgt.append(inner_data[ik]['tgt'])
        # print(f'------55 src:{src}')
        # print(f'---------55tgt:{tgt}')

        return src, tgt


class MGCCCollator(DataCollatorWithPadding):
    def __call__(self, features):
        self.mode = features[0][-1]
        self.contrastive_type = features[0][-2]
        print(f'---------mgcc collator mode:{self.mode}, contype:{self.contrastive_type}')

        if self.mode == 'train' and self.contrastive_type != '11' and self.contrastive_type != 'indexing':
            inputs = self.multi_graded(features)
        elif self.mode == 'train' and (self.contrastive_type == '11' or self.contrastive_type == 'indexing'):
            inputs = self.singel_graded(features)
        else:
            inputs = self.test_mode(features)

        inputs['mode'] = self.mode

        return inputs

    def multi_graded(self, features):
        max_length = features[0][-3]
        # print(f'---------mgcc collator fea:{features}')

        src_ids = [xx for x in features for xx in x[0]]
        docids = [xx for x in features for xx in x[1]]
        # print(f'---------mgcc collator srcids:{src_ids}')
        # print(f'---------mgcc collator docids:{docids}')

        inputs = self.tokenizer(
            src_ids, padding="longest", truncation='longest_first', return_tensors="pt", max_length=max_length
        )
        # print('---mgcc collator inputs:', inputs)

        labels = self.tokenizer(
            docids, padding="longest", return_tensors="pt"
        ).input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100
        # print('---mgcc collator labels:', labels)

        inputs['labels'] = labels
        contrastive_type = [x[3] for x in features]

        assert len(set(contrastive_type)) == 1
        inputs['type'] = contrastive_type.pop()
        # print('---mgcc collator final inputs:', inputs)
        # print('---------mgcc collator multi graded contype:', inputs['type'])

        return inputs
    def singel_graded(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        docids = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        labels = self.tokenizer(
            docids, padding="longest", return_tensors="pt"
        ).input_ids

        # replace padding token id's of the labels by -100 according to https://huggingface.co/docs/transformers/model_doc/t5#training
        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels
        inputs['type'] = self.contrastive_type
        # print('---------mgcc collator singel graded contype:', inputs['type'])

        return inputs
    def test_mode(self, features):
        # print('---mgcc collator test:')

        input_ids = [{'input_ids': x[0]} for x in features]
        docids = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        labels = self.tokenizer(
            docids, padding="longest", return_tensors="pt"
        ).input_ids

        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels
        inputs['type'] = self.mode
        # print('---------mgcc collator test type', inputs['type'])


        return inputs


class DiscriminationBatchTrainDataset(Dataset):
    def __init__(
            self,
            path_to_data,
            path_did2queryID,
            mode,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
            remove_prompt=False,
    ):
        self.train_data = datasets.load_dataset(
            'json',
            data_files=path_to_data,
            ignore_verifications=False,
            cache_dir=cache_dir
        )['train']

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.remove_prompt = remove_prompt
        self.total_len = len(self.train_data)
        self.valid_ids = set()
        self.query2qid = {}
        self.docid2did = {}
        self.src = []
        self.qids = []

        # for data in tqdm(self.train_data):
        #     self.valid_ids.add(str(data['translation']['tgt']))
        # print(f"valid ids")
        # print(self.valid_ids)

        if mode == 'test':
            self.src, self.qids = load_dev(path_to_data)
            # for data in tqdm(self.train_data):
        self.docid2did = load_hash2id_all(path_did2queryID)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        # print(f'--item {item}')


        # data = self.train_data[item]
        # src = data['translation']['prefix'] + ": " + data['translation']['src']
        # input_ids = self.tokenizer(src,
        #                            return_tensors="pt",
        #                            truncation='only_first',
        #                            max_length=self.max_length).input_ids[0]
        # label = data['translation']['tgt']
        # src_id = data['translation']['src-id']
        # # label_ids = self.tokenizer(label, return_tensors="pt",).label_ids[0]
        #
        # return input_ids, label, src_id

        batch_src, batch_input_ids, batch_label, batch_src_id = [], [], [], []
        for idx in item:
            data = self.train_data[idx]
            # print(f'------data {data} {type(data)}')
            src = data['translation']['prefix'] + ": " + data['translation']['src']
            batch_src.append(src)
            label = data['translation']['tgt']
            src_id = data['translation']['src-id']
            batch_label.append(label)
            batch_src_id.append(src_id)
        input_ids = self.tokenizer(batch_src,
                                   return_tensors="pt",
                                   padding=True,
                                   truncation=True,
                                   max_length=self.max_length).input_ids

        return input_ids, batch_label, batch_src_id


@dataclass
class IndexingCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        docids = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        labels = self.tokenizer(
            docids, padding="longest", return_tensors="pt"
        ).input_ids
        # print(f'---pad token:{self.tokenizer.pad_token_id}')

        # replace padding token id's of the labels by -100 according to https://huggingface.co/docs/transformers/model_doc/t5#training
        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels
        # print(f'labels:{labels}')
        # qids = [x[2] for x in features]
        # inputs['qids'] = qids
        return inputs


class ListMLETrainDataset(Dataset):
    def __init__(
            self,
            path_to_data,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
            remove_prompt=False,
    ):
        self.train_data = datasets.load_dataset(
            'json',
            data_files=path_to_data,
            ignore_verifications=False,
            cache_dir=cache_dir
        )['train']

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.remove_prompt = remove_prompt
        self.total_len = len(self.train_data)
        self.valid_ids = set()
        for data in tqdm(self.train_data):
            self.valid_ids.add(str(data['text_id']))

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        data = self.train_data[item]
        if self.remove_prompt:
            data['text'] = data['text'][9:] if data['text'].startswith('Passage: ') else data['text']
            data['text'] = data['text'][10:] if data['text'].startswith('Question: ') else data['text']
        input_ids = self.tokenizer(data['text'],
                                   return_tensors="pt",
                                   truncation='only_first',
                                   max_length=self.max_length).input_ids[0]
        labels = self.tokenizer(
            str(data['text_id']), padding="longest", return_tensors="pt"
        ).input_ids[0]

        # return input_ids, str(data['text_id'])
        return input_ids, labels

class ListMLECollator(DataCollatorWithPadding):
    def __call__(self, features):
        # print(f'--feathures:{features}')
        input_ids = [{'input_ids': x[0]} for x in features]
        labels = [{'input_ids': x[1]} for x in features]
        inputs = super().__call__(input_ids)

        labels_temp = super().__call__(labels)

        # labels = self.tokenizer(
        #     docids, padding="longest", return_tensors="pt"
        # ).input_ids
        # print(f'---labels :{labels}')
        # label_temp = [{'input_ids': x} for x in labels]
        # label_mask = super().__call__(label_temp)
        # print(f'--label mask:{labels_temp}')
        # replace padding token id's of the labels by -100 according to https://huggingface.co/docs/transformers/model_doc/t5#training
        labels_temp['input_ids'][labels_temp['input_ids'] == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels_temp['input_ids']
        inputs['label_mask'] = labels_temp['attention_mask']
        # print(f'labels:{labels}')
        # qids = [x[2] for x in features]
        # inputs['qids'] = qids
        return inputs



# class ListMLECollator(DataCollatorWithPadding):
#     def __call__(self, features):
#         # print(f'--feathures:{features}')
#         input_ids = [{'input_ids': x[0]} for x in features]
#         docids = [x[1] for x in features]
#         inputs = super().__call__(input_ids)
#
#         labels = self.tokenizer(
#             docids, padding="longest", return_tensors="pt"
#         ).input_ids
#         print(f'---labels :{labels}')
#         label_temp = [{'input_ids': x} for x in labels]
#         label_mask = super().__call__(label_temp)
#         print(f'--label mask:{label_mask}')
#         # replace padding token id's of the labels by -100 according to https://huggingface.co/docs/transformers/model_doc/t5#training
#         labels[labels == self.tokenizer.pad_token_id] = -100
#         inputs['labels'] = labels
#         inputs['label_mask'] = label_mask['attention_mask']
#         # print(f'labels:{labels}')
#         # qids = [x[2] for x in features]
#         # inputs['qids'] = qids
#         return inputs


@dataclass
class QueryEvalCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        labels = [x[1] for x in features]
        inputs = super().__call__(input_ids)


        return inputs, labels


class EvalCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        labels = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        labels = self.tokenizer(
            labels, padding="longest", return_tensors="pt"
        ).input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100


        inputs['labels'] = labels

        return inputs


class DiscriminationCollator(DataCollatorWithPadding):
    def __call__(self, features):
        # print(features)
        print('discr collator')
        batch_size = len(features)
        input_ids = [{'input_ids': x[0]} for x in features]
        docids = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        for key in inputs.keys():
            inputs[key] = inputs[key].repeat(batch_size, 1)


        labels = self.tokenizer(
            docids, padding="longest", return_tensors="pt"
        ).input_ids

        labels[labels == self.tokenizer.pad_token_id] = -100
        # inputs['labels'] = labels
        inputs['labels'] = labels.repeat_interleave(batch_size, dim=0)
        flag = [0] * (batch_size*batch_size)
        for i in range(batch_size):
            flag[i*(batch_size+1)] = 1
        inputs['flag'] = torch.tensor(flag)

        # print(inputs)
        return inputs


class DiscriminationSameDocCollator(DataCollatorWithPadding):
    def __call__(self, features):
        # print(features)
        batch_size = len(features)
        input_ids = [{'input_ids': x[0]} for x in features]
        docids = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        for key in inputs.keys():
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)


        labels = self.tokenizer(
            docids, padding="longest", return_tensors="pt"
        ).input_ids

        labels[labels == self.tokenizer.pad_token_id] = -100
        # inputs['labels'] = labels
        inputs['labels'] = labels.repeat(batch_size, 1)
        flag = [0] * (batch_size*batch_size)
        for i in range(batch_size):
            flag[i*(batch_size+1)] = 1
        inputs['flag'] = torch.tensor(flag)

        # print(inputs)
        return inputs


#global contrastive_num

def set_contrastive_num(num):
    global contrastive_num

    contrastive_num = num


def get_contrastive_num():
    return contrastive_num

class DiscriminationMultiCollator(DataCollatorWithPadding):
    def __call__(self, features):
        # print(features)
        prefix = set([x[3] for x in features])

        if 'Contrastive' in prefix and len(prefix) == 1: # only index
            inputs = self.contrastive_type(features)
        elif 'Contrastive' in prefix and len(prefix) > 1: # mix
            inputs = self.mix_type(features)
        else: # normal
            inputs = self.normal_type(features)

        return inputs

    def contrastive_type(self, features):
        # print('---contrastive type')
        con_num = get_contrastive_num()
        # print(f'con num :{con_num}')
        index_input_ids = [{'input_ids': x[0]} for x in features]
        print('---dis multi coll input ids:', index_input_ids)
        index_inputs = super().__call__(index_input_ids)
        group_size = int((len(features) / con_num))
        index_docids = [x[1] for x in features]
        for key in index_inputs.keys():
            index_inputs[key] = index_inputs[key].repeat(group_size, 1)

        index_labels = self.tokenizer(
            index_docids, padding="longest", return_tensors="pt"
        ).input_ids

        index_labels[index_labels == self.tokenizer.pad_token_id] = -100
        index_inputs['labels'] = index_labels.repeat_interleave(group_size, dim=0)
        index_inputs['flag'] = ['Contrastive'] * group_size * len(features)

        return index_inputs

    def mix_type(self, features):
        # print('---mix type')
        con_num = get_contrastive_num()
        # print(f'con num :{con_num}')

        prefix = [x[3] for x in features]
        normal_size = prefix.index('Contrastive')
        index_size = len(prefix) - normal_size

        all_input_ids = [{'input_ids': x[0]} for x in features]
        print('---dis multi coll input ids:', all_input_ids)

        all_inputs = super().__call__(all_input_ids)
        all_docids = [x[1] for x in features]
        all_labels = self.tokenizer(all_docids, padding="longest", return_tensors="pt").input_ids
        all_labels[all_labels == self.tokenizer.pad_token_id] = -100
        all_inputs['labels'] = all_labels

        # for normal
        normal_inputs = {}
        for key in all_inputs.keys():
            normal_inputs[key] = all_inputs[key][:normal_size]

        # for index
        index_inputs = {}
        for key in all_inputs.keys():
            index_inputs[key] = all_inputs[key][normal_size:]

        group_size = int((index_size / con_num))
        for key in ['input_ids', 'attention_mask']:
            index_inputs[key] = index_inputs[key].repeat(group_size, 1)

        index_inputs['labels'] = index_inputs['labels'].repeat_interleave(group_size, dim=0)

        # combination
        inputs = {}
        for key in normal_inputs.keys():
            inputs[key] = torch.vstack((normal_inputs[key], index_inputs[key]))
        inputs['flag'] = ['normal'] * normal_size
        inputs['flag'].extend(['Contrastive'] * group_size * index_size)
        # print(inputs['flag'])
        return inputs

    def normal_type(self, features):
        # print('---normal type')
        input_ids = [{'input_ids': x[0]} for x in features]
        print('---dis multi coll input ids:', input_ids)

        docids = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        labels = self.tokenizer(
            docids, padding="longest", return_tensors="pt"
        ).input_ids

        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels
        inputs['flag'] = ['normal'] * len(features)

        return inputs


class DiscriminationBatchCollator(DataCollatorWithPadding):
    def __call__(self, features):
        # input_ids = [{'input_ids': x[0]} for x in features]
        # docids = [x[1] for x in features]
        # inputs = super().__call__(input_ids)
        #
        # labels = self.tokenizer(
        #     docids, padding="longest", return_tensors="pt"
        # ).input_ids
        #
        # # replace padding token id's of the labels by -100 according to https://huggingface.co/docs/transformers/model_doc/t5#training
        # labels[labels == self.tokenizer.pad_token_id] = -100
        # inputs['labels'] = labels
        # inputs['flag'] = 'positive'
        # # qids = [x[2] for x in features]
        # # inputs['qids'] = qids
        #
        # return inputs

        inputs = []
        for group in features:
            src_input, labels, srcids = group

            input_ids = [{'input_ids': x} for x in src_input]
            docids = [x for x in labels]

            batch = self.tokenizer.pad(
                input_ids,
                padding=True,
                max_length=self.max_length,
                pad_to_multiple_of=None,
                return_tensors='pt',
            )
            if "label" in batch:
                batch["labels"] = batch["label"]
                del batch["label"]
            if "label_ids" in batch:
                batch["labels"] = batch["label_ids"]
                del batch["label_ids"]

            labels = self.tokenizer(
                docids, padding="longest", return_tensors="pt"
            ).input_ids

        # replace padding token id's of the labels by -100 according to https://huggingface.co/docs/transformers/model_doc/t5#training
            labels[labels == self.tokenizer.pad_token_id] = -100
            flags = ['positive'] * len(labels)
            batch['labels'] = labels
            batch['flag'] = flags
            # inputs['input_ids'] = batch
            # inputs['labels'] = labels
            # inputs['flag'] = flags
        # qids = [x[2] for x in features]
        # inputs['qids'] = qids
            inputs.append(batch)

        return inputs


# combined dataset class
class CombinationDataset(torch.utils.data.DataLoader):
    def __init__(self, datasets):
        self.datasets = datasets
        # print('-----combine init:', datasets)

    def __len__(self):
        # print(f'com dataset len:{(sum([dataset.__len__() for dataset in self.datasets]))}')
        return (sum([dataset.__len__() for dataset in self.datasets]))

    def __getitem__(self, indicies):
        # print(f'--com indicies:{indicies}, dataset:{self.datasets[indicies[0]]}')
        dataset_idx = indicies[0]
        data_idx = indicies[1]
        # print(indicies)

        return self.datasets[dataset_idx].__getitem__(data_idx)


# class that will take in multiple samplers and output batches from a single dataset at a time
class ComboBatchSampler():

    def __init__(self, samplers, batch_size, drop_last):

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.samplers = samplers
        self.iterators = [iter(sampler) for sampler in self.samplers]
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        print(f'drop {self.drop_last}')
        # define how many batches we will grab
        self.min_batches = min([len(sampler) for sampler in self.samplers])
        self.n_batches = self.min_batches * len(self.samplers)

        # define which indicies to use for each batch
        self.dataset_idxs = []
        random.seed(42)
        for j in range((self.n_batches // len(self.samplers) + 1)):
            loader_inds = list(range(len(self.samplers)))
            random.shuffle(loader_inds)
            self.dataset_idxs.extend(loader_inds)
        self.dataset_idxs = self.dataset_idxs[:self.n_batches]

        # return the batch indicies
        batch = []
        for dataset_idx in self.dataset_idxs:
            for idx in self.samplers[dataset_idx]:
                batch.append((dataset_idx, idx))
                if len(batch) == self.batch_size:
                    yield (batch)
                    batch = []
                    break
            if len(batch) > 0 and not self.drop_last:
                yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return (sum([len(sampler) for sampler in self.samplers])) // self.batch_size
        else:
            return (sum([len(sampler) for sampler in self.samplers]) + self.batch_size - 1) // self.batch_size


class CustomComboBatchSampler(Sampler[List[int]]):
    def __init__(self, samplers, batch_size, drop_last):

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.samplers = samplers
        # self.iterators = [iter(sampler) for sampler in self.samplers]
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        print(f'drop {self.drop_last}')

        # default dataset 0 < dataset 1
        # define how many batches we will grab
        self.min_batches = min([len(sampler) for sampler in self.samplers]) // self.batch_size
        self.max_batches = max([len(sampler) for sampler in self.samplers]) // self.batch_size

        gap = self.max_batches // self.min_batches
        random.seed(42)
        self.dataset_idxs = [1] * (self.max_batches + self.min_batches)
        for i in range(1, self.max_batches, gap):
            self.dataset_idxs[i] = 0

        self.n_batches = min([len(sampler) for sampler in self.samplers]) * len(self.samplers)

        self.dataset_idxs = self.dataset_idxs[:self.n_batches]
        # print(self.dataset_idxs)

        # return the batch indicies
        data_idx_0 = [idx for idx in self.samplers[0]]
        data_idx_1 = [idx for idx in self.samplers[1]]
        pointer_0, pointer_1 = 0, 0
        for dataset_idx in self.dataset_idxs:
            if dataset_idx == 0:
                start_idx = pointer_0
                end_idx = pointer_0 + self.batch_size

                batch = [(dataset_idx, idx) for idx in data_idx_0[start_idx:end_idx]]
                pointer_0 = end_idx
                if len(batch) == self.batch_size:
                    yield (batch)
                    batch = []
            else:
                start_idx = pointer_1
                end_idx = pointer_1 + self.batch_size
                batch = [(dataset_idx, idx) for idx in data_idx_1[start_idx:end_idx]]
                pointer_1 = end_idx
                if len(batch) == self.batch_size:
                    yield (batch)
                    batch = []

            if len(batch) > 0 and not self.drop_last:
                yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return (sum([len(sampler) for sampler in self.samplers])) // self.batch_size
        else:
            return (sum([len(sampler) for sampler in self.samplers]) + self.batch_size - 1) // self.batch_size


class MixComboBatchSampler(Sampler[List[int]]):
    def __init__(self, samplers, batch_size, drop_last):

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.samplers = samplers
        # self.iterators = [iter(sampler) for sampler in self.samplers]
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        # print(f'drop {self.drop_last}')

        index_idxs = [i for i in self.samplers[1]]
        normal_idxs = [i for i in self.samplers[0]]

        index_batch_size = int((self.batch_size // 3) ** 0.5)
        normal_batch_size = self.batch_size - (index_batch_size ** 2) * 3

        index_batches, normal_batches = len(index_idxs)//index_batch_size, len(normal_idxs)//normal_batch_size
        up_bounder = min(index_batches, normal_batches)

        i = 0
        for i in range(up_bounder):
            normal_min_batch = normal_idxs[i * normal_batch_size: (i+1) * normal_batch_size]
            index_min_batch = index_idxs[i * index_batch_size: (i+1) * index_batch_size]

            normal_extend_min_batch = [(0, i) for i in normal_min_batch]

            index_extend_min_batch = [(1, 3*i+j) for i in index_min_batch for j in range(3)]
            normal_extend_min_batch.extend(index_extend_min_batch)
            assert self.batch_size == len(normal_min_batch) + len(index_min_batch) ** 2 * 3
            yield normal_extend_min_batch
            normal_extend_min_batch = []
            index_extend_min_batch = []

        rest = max(index_batches, normal_batches) - up_bounder
        if rest > 0:
            if index_batches > normal_batches:
                new_batches = index_batches - up_bounder
                rest_index_idxs = index_idxs[(i+1) * index_batch_size:]
                for i in range(new_batches):
                    index_min_batch = rest_index_idxs[i * index_batch_size: (i+1) * index_batch_size]
                    index_extend_min_batch = [(1, 3*i+j) for i in index_min_batch for j in range(3)]
                    yield index_extend_min_batch
                    index_extend_min_batch = []
            else:
                new_batches = (normal_batches - up_bounder) * normal_batch_size // self.batch_size
                rest_normal_idxs = normal_idxs[(i+1) * normal_batch_size:]
                for i in range(new_batches):
                    normal_min_batch = rest_normal_idxs[i * self.batch_size: (i + 1) * self.batch_size]
                    normal_extend_min_batch = [(0, i) for i in normal_min_batch]
                    yield normal_extend_min_batch
                    normal_extend_min_batch = []

    def __len__(self) -> int:
        return (len(self.samplers[0]) + len(self.samplers[1]) * 3) // self.batch_size

        # if self.drop_last:
        #     return (sum([len(sampler) for sampler in self.samplers])) // self.batch_size
        # else:
        #     return (sum([len(sampler) for sampler in self.samplers]) + self.batch_size - 1) // self.batch_size



class DynamicComboBatchSampler(Sampler[List[int]]):
    def __init__(self, samplers, batch_size, drop_last, contrastive_num):

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.samplers = samplers
        # self.iterators = [iter(sampler) for sampler in self.samplers]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.contrastive_num = contrastive_num

    def __iter__(self) -> Iterator[List[int]]:
        # print(f'drop {self.drop_last}')

        con_num = self.contrastive_num
        index_idxs = [i for i in self.samplers[1]]
        normal_idxs = [i for i in self.samplers[0]]

        index_batch_size = int((self.batch_size // con_num) ** 0.5)
        normal_batch_size = self.batch_size - (index_batch_size ** 2) * con_num

        index_batches, normal_batches = len(index_idxs)//index_batch_size, math.ceil(len(normal_idxs)/normal_batch_size)
        mix_times = min(index_batches, normal_batches)

        i = 0
        for i in range(mix_times):
            normal_min_batch = normal_idxs[i * normal_batch_size: (i+1) * normal_batch_size]
            index_min_batch = index_idxs[i * index_batch_size: (i+1) * index_batch_size]

            normal_extend_min_batch = [(0, i) for i in normal_min_batch]

            index_extend_min_batch = [(1, con_num*i+j) for i in index_min_batch for j in range(con_num)]
            normal_extend_min_batch.extend(index_extend_min_batch)
            assert self.batch_size == len(normal_min_batch) + len(index_min_batch) ** 2 * con_num
            yield normal_extend_min_batch
            normal_extend_min_batch = []
            index_extend_min_batch = []

        rest = max(index_batches, normal_batches) - mix_times
        if rest > 0:
            if index_batches > normal_batches:
                new_batches = index_batches - mix_times
                rest_index_idxs = index_idxs[(i+1) * index_batch_size:]
                for i in range(new_batches):
                    index_min_batch = rest_index_idxs[i * index_batch_size: (i+1) * index_batch_size]
                    index_extend_min_batch = [(1, con_num*i+j) for i in index_min_batch for j in range(con_num)]
                    yield index_extend_min_batch
                    index_extend_min_batch = []
            else:
                new_batches = (normal_batches - mix_times) * normal_batch_size // self.batch_size
                rest_normal_idxs = normal_idxs[(i+1) * normal_batch_size:]
                for i in range(new_batches):
                    normal_min_batch = rest_normal_idxs[i * self.batch_size: (i + 1) * self.batch_size]
                    normal_extend_min_batch = [(0, i) for i in normal_min_batch]
                    yield normal_extend_min_batch
                    normal_extend_min_batch = []

    def __len__(self) -> int:
        return (len(self.samplers[0]) + len(self.samplers[1]) * self.contrastive_num) // self.batch_size

        # if self.drop_last:
        #     return (sum([len(sampler) for sampler in self.samplers])) // self.batch_size
        # else:
        #     return (sum([len(sampler) for sampler in self.samplers]) + self.batch_size - 1) // self.batch_size

class MGCCSampler(Sampler[List[int]]):
    def __init__(self, samplers, batch_size, drop_last):

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.samplers = samplers
        # self.iterators = [iter(sampler) for sampler in self.samplers]
        self.batch_size = batch_size
        self.drop_last = drop_last
        # self.contrastive_num = contrastive_num
        self.type222_len = 7
        self.type122_len = 6
        self.type111_len = 4

    def __iter__(self) -> Iterator[List[int]]:
        # print(f'drop {self.drop_last}')

        dataset222_idx = [i for i in self.samplers[0]]
        # dataset122_idx = [i for i in self.samplers[1]]
        # dataset111_idx = [i for i in self.samplers[2]]

        batchsize_222 = self.batch_size// self.type222_len
        # batchsize_122 = self.batch_size// self.type122_len
        # batchsize_111 = self.batch_size// self.type111_len

        batches222 = len(dataset222_idx) // batchsize_222
        # batches122 = len(dataset122_idx) // batchsize_122
        # batches111 = len(dataset111_idx) // batchsize_111


        for i in range(batches222):
            batch_222 = dataset222_idx[i*batchsize_222: (i+1)*batchsize_222]
            assert self.batch_size >= len(batch_222) * batchsize_222
            yield batch_222
            batch_222 = []

        # for i in range(batches122):
        #     batch_122 = dataset122_idx[i*batchsize_122: (i+1)*batchsize_122]
        #     assert self.batch_size >= len(batch_122) * batchsize_122
        #     yield batch_122
        #     batch_122 = []

        # for i in range(batches111):
        #     batch_111 = dataset111_idx[i*batchsize_111: (i+1)*batchsize_111]
        #     assert self.batch_size >= len(batch_111) * batchsize_111
        #     yield batch_111
        #     batch_111 = []


    def __len__(self) -> int:
        # return len(self.samplers[0]) // self.batch_size // self.type222_len + len(self.samplers[1]) // self.batch_size // self.type122_len + len(self.samplers[2]) // self.batch_size//self.type111_len

        # return len(self.samplers[0]) // self.batch_size // self.type222_len + len(
        #     self.samplers[1]) // self.batch_size // self.type122_len

        return len(self.samplers[0]) // self.batch_size // self.type222_len

class MGCCBatchSampler(Sampler[List[int]]):
    def __init__(self, samplers, batch_size, drop_last, typelen, dataset_batchsize_lst):

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.samplers = samplers
        # self.iterators = [iter(sampler) for sampler in self.samplers]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.typelen = typelen
        self.dataset_batchsize_lst = dataset_batchsize_lst


    def __iter__(self) -> Iterator[List[int]]:
        # print(f'drop {self.drop_last}')
        for i in range(len(self.samplers)):
            idxs = [j for j in self.samplers[i]]
            if self.dataset_batchsize_lst[i] != 0:
                dataset_batch_size = self.dataset_batchsize_lst[i]
                assert self.typelen[i] == 1
            else:
                dataset_batch_size = self.batch_size
            batch_size = dataset_batch_size // self.typelen[i]
            batches = math.ceil(len(idxs) / batch_size)
            # print(f'----mgcc batch sampler:{i}, len(idx):{len(idxs)}, dataset batch size:{dataset_batch_size}, batch size:{batch_size}, batches:{batches}')

            for k in range(batches):
                batch = idxs[k*batch_size: (k+1)*batch_size]
                yield_batch = [(i, h) for h in batch]
                assert dataset_batch_size >= len(yield_batch) * self.typelen[i] or (dataset_batch_size == batch_size and dataset_batch_size >= len(yield_batch))
                # print('--mgcc batch sampler:', yield_batch)
                yield yield_batch
                yield_batch = []


    def __len__(self) -> int:
        total_len = 0
        for i in range(len(self.samplers)):
            dataset_batch_size = self.dataset_batchsize_lst[i] if self.dataset_batchsize_lst[i] != 0 else self.batch_size
            total_len += math.ceil(len(self.samplers[i]) / (dataset_batch_size // self.typelen[i]))
        print(f'----------------------mgcc batch sampler len:{total_len}')
        return total_len


class ListwiseSampler(Sampler[List[int]]):
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


def load_hash2id_all(path):
    h2id = collections.defaultdict(set)
    with open(path, 'r') as f_d:
        for line in tqdm(f_d, desc='loading hash2id...'):
            did, hash_data = line.rstrip().split('\t')
            h2id[hash_data].add(did)
    return h2id


def load_site2domain(path):
    site2domain = {}
    with open(path, 'r', encoding='utf-8',errors='ignore') as f_d:
        for line in f_d:
            site, domain = line.rstrip().split('\t')
            site2domain[site] = domain
    return site2domain


def load_dev(input_file):
    srcs = []
    src_ids = []
    fin = open(input_file, encoding='utf-8',errors='ignore')
    for line in jsonlines.Reader(fin):
        srcs.append(line['translation']['src'])
        src_ids.append(line['translation']['src-id'])

    # print(f'----------{len(srcs)}')
    return srcs, src_ids

