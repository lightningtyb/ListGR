from data import IndexingTrainDataset, GenerateDataset, IndexingCollator, QueryEvalCollator, TranslationTrainDataset,DiscriminationCollator, DiscriminationTrainDataset, DiscriminationBatchCollator,DiscriminationSameDocCollator, DiscriminationMultiCollator, CombinationDataset, ComboBatchSampler, DiscriminationBatchTrainDataset, DiscriminationMixDataset, set_contrastive_num, EvalDataset, EvalCollator, MGCCTrain11Dataset, MGCCCollator, MGCCTrainMultiDataset, MGCCTrain55MultiDataset, ListwiseSampler, ListMLECollator,ListMLETrainDataset

from transformers import (
    T5Tokenizer,
    T5TokenizerFast,
    TrainingArguments,
    TrainerCallback,
    MT5Tokenizer,
    MT5TokenizerFast,
    MT5ForConditionalGeneration,
    HfArgumentParser,
    T5ForConditionalGeneration,
    set_seed,
)
from model import T5ForConditionalGenerationWoLoss, MGCEC
from trainer import DSITrainer, DocTqueryTrainer, TranslationTrainer, DiscriminationTrainer, \
    DisMultiGranularityTrainer, EvalTrainer, MGCECTrainer, PListMLETrainer
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Optional
import json
from tqdm import tqdm
import os
import pickle
import re
set_seed(313)


@dataclass
class RunArguments:
    model_name: str = field(default=None)
    model_path: Optional[str] = field(default=None)
    max_length: Optional[int] = field(default=32)
    max_indexing_length: Optional[int] = field(default=256)
    id_max_length: Optional[int] = field(default=64)
    remove_prompt: Optional[bool] = field(default=False)
    train_file: str = field(default=None)
    valid_file: str = field(default=None)
    task: str = field(default=None,  metadata={"help": "DSI, docTquery, generation"})
    top_k: Optional[int] = field(default=10)
    num_return_sequences: Optional[int] = field(default=10)
    q_max_length: Optional[int] = field(default=32)
    trie_path: Optional[str] = field(default=None)
    path_did2queryID: Optional[str] = field(default=None)
    alpha: Optional[float] = field(default=1)
    beta: Optional[float] = field(default=1)
    temperature: Optional[float] = field(default=0.1)
    include_all_pos: Optional[bool] = field(default=False)
    collator_type: str = field(default=None,  metadata={"help": "samedoc, multi"})
    eval_files_dir: Optional[str] = field(default='./eval_files')
    index_file: str = field(default=None)
    contrastive_num: Optional[int] = field(default=2)
    eval_name: Optional[str] = field(default='eval')
    mgcc_loss_type: str = field(default='hmce')
    mgcc_train_file_root: str = field(default=None)
    train_indexing_batch_size: Optional[int] = field(default=30)
    total_grades: Optional[int] = field(default=3)



def make_compute_metrics(tokenizer, valid_ids):

    def compute_metrics(eval_preds):
        hit_at_1 = 0
        hit_at_10 = 0
        for beams, label in zip(eval_preds.predictions, eval_preds.label_ids):
            rank_list = tokenizer.batch_decode(beams,
                                               skip_special_tokens=True)
            label_id = tokenizer.decode(label, skip_special_tokens=True)
            # filter out duplicates and invalid docids
            filtered_rank_list = []
            for docid in rank_list:
                if docid not in filtered_rank_list and docid in valid_ids:
                    filtered_rank_list.append(docid)

            hits = np.where(np.array(filtered_rank_list)[:10] == label_id)[0]
            if len(hits) != 0:
                hit_at_10 += 1
                if hits[0] == 0:
                    hit_at_1 += 1
        return {"Hits@1": hit_at_1 / len(eval_preds.predictions), "Hits@10": hit_at_10 / len(eval_preds.predictions)}
    return compute_metrics


def make_compute_metrics_mrr(tokenizer, valid_ids):

    def compute_metrics(eval_preds):
        hit_at_1 = 0
        hit_at_10 = 0
        MRR20, MRR5, MRR3 = 0, 0, 0

        def cal_MRR(rank_list, label_id, MAX):
            MRR = 0
            for i, candi in enumerate(rank_list):
                if i < MAX and candi == label_id:
                    MRR = 1 / (i + 1)
            return MRR

        for beams, label in zip(eval_preds.predictions, eval_preds.label_ids):
            rank_list = tokenizer.batch_decode(beams,
                                               skip_special_tokens=True)
            label = np.where(label != -100, label, tokenizer.pad_token_id)
            label_id = tokenizer.decode(label, skip_special_tokens=True)
            # filter out duplicates and invalid docids
            filtered_rank_list = []
            for docid in rank_list:
                if docid not in filtered_rank_list and docid in valid_ids:
                    filtered_rank_list.append(docid)

            MRR20 += cal_MRR(filtered_rank_list, label_id, 20)
            MRR5 += cal_MRR(filtered_rank_list, label_id, 5)
            MRR3 += cal_MRR(filtered_rank_list, label_id, 3)

            hits = np.where(np.array(filtered_rank_list)[:10] == label_id)[0]
            if len(hits) != 0:
                hit_at_10 += 1
                if hits[0] == 0:
                    hit_at_1 += 1
        total_len = len(eval_preds.predictions)
        MRR20, MRR5, MRR3 = MRR20 / total_len, MRR5 / total_len, MRR3 / total_len

        return {"Hits@1": hit_at_1 / total_len, "Hits@10": hit_at_10 / total_len,
                "MRR@20": MRR20, "MRR@5": MRR5, "MRR@3": MRR3}
    return compute_metrics


def make_compute_discrimination_metrics(tokenizer, docid2did, src, qids, eval_files_dir):

    def compute_metrics(eval_preds):
        hit_at_1 = 0
        hit_at_10 = 0
        outputs = []
        out4eval = []

        MRR20, MRR5, MRR3 =0, 0, 0

        def cal_MRR(rank_list, label_id, MAX):
            MRR = 0
            for i, candi in enumerate(rank_list):
                predict_did = docid2did[candi]
                label_did = docid2did[label_id]
                if i < MAX and predict_did == label_did:
                    MRR = 1/(i+1)
            return MRR
        i = 0
        # print(f'----metric keys:{eval_preds.logit_score}')
        for beams, label, score in zip(eval_preds.predictions, eval_preds.label_ids, eval_preds.logit_score):
            rank_list = tokenizer.batch_decode(beams,skip_special_tokens=True)
            label = np.where(label != -100, label, tokenizer.pad_token_id)
            label_id = tokenizer.decode(label, skip_special_tokens=True)
            # print(f'----score:{scores}')
            # print(label_id)

            # filter out duplicates and invalid docids
            filtered_rank_list = []
            for docid in rank_list:
                if docid not in filtered_rank_list and docid in docid2did.keys():
                    filtered_rank_list.append(docid)

            rank = 1
            score_cnt = 0
            for r in filtered_rank_list:
                outputs.append(src[i] + '\t' + r)
                for did in docid2did[r]:
                    # out4eval.append(qids[i] + '\t' + did)
                    out4eval.append(f'{qids[i]}\t0\t{did}\t{rank}\t{score[score_cnt]}\tGen\n')
                    rank += 1
                score_cnt += 1

            MRR20 += cal_MRR(filtered_rank_list, label_id, 20)
            MRR5 += cal_MRR(filtered_rank_list, label_id, 5)
            MRR3 += cal_MRR(filtered_rank_list, label_id, 3)

            if label_id in filtered_rank_list[:10]:
                hit_at_10 += 1
                if label_id in filtered_rank_list[:1]:
                    hit_at_1 += 1

            i += 1
        total_len = len(eval_preds.predictions)
        MRR20, MRR5, MRR3 = MRR20 / total_len, MRR5 / total_len, MRR3 / total_len
        hit_at_1, hit_at_10 = hit_at_1 / total_len, hit_at_10 / total_len

        save_list(eval_files_dir, outputs, out4eval, MRR3, MRR20, hit_at_1, hit_at_10)
        return {"MRR@3": MRR3, "MRR@5": MRR5, "MRR@20": MRR20,
                "Hits@1": hit_at_1, "Hits@10": hit_at_10}

    return compute_metrics

def eval_metrics(all_src, all_predict, all_label):

    hit_at_1 = 0
    hit_at_10 = 0
    MRR20, MRR5, MRR3 =0, 0, 0

    def cal_MRR(rank_list, label_id, MAX):
        MRR = 0
        for i, predict_id in enumerate(rank_list):
            if i < MAX and predict_id == label_id:
                MRR = 1/(i+1)
        return MRR

    for src, label_id, filtered_rank_list in zip(all_src, all_label, all_predict):
        MRR20 += cal_MRR(filtered_rank_list, label_id, 20)
        MRR5 += cal_MRR(filtered_rank_list, label_id, 5)
        MRR3 += cal_MRR(filtered_rank_list, label_id, 3)

        if label_id in filtered_rank_list[:10]:
            hit_at_10 += 1
            if label_id in filtered_rank_list[:1]:
                hit_at_1 += 1

    total_len = len(all_src)
    MRR20, MRR5, MRR3 = MRR20 / total_len, MRR5 / total_len, MRR3 / total_len
    hit_at_1, hit_at_10 = hit_at_1 / total_len, hit_at_10 / total_len

    return {"MRR@3": MRR3, "MRR@5": MRR5, "MRR@20": MRR20,
            "Hits@1": hit_at_1, "Hits@10": hit_at_10}



def main():

    parser = HfArgumentParser((TrainingArguments, RunArguments))
    training_args, run_args = parser.parse_args_into_dataclasses()

    # We use wandb logger: https://wandb.ai/site.
    if training_args.local_rank == 0:  # only on main process
        # Initialize wandb run
        # wandb.login()
        # wandb.init(project="DSI", name=training_args.run_name)
        os.environ["WANDB_API_KEY"] = 'cd4c07642e48512eb03192f77c811027edac1b1c'
        os.environ["WANDB_MODE"] = "offline"


    if 'translation' in run_args.task:
        tokenizer = T5Tokenizer.from_pretrained(run_args.model_name, cache_dir='cache')
        fast_tokenizer = T5TokenizerFast.from_pretrained(run_args.model_name, cache_dir='cache')
        if run_args.model_path:
            model = T5ForConditionalGeneration.from_pretrained(run_args.model_path, cache_dir='cache')
        else:
            model = T5ForConditionalGeneration.from_pretrained(run_args.model_name, cache_dir='cache')
    elif 'mgcc' in run_args.task:
        tokenizer = T5Tokenizer.from_pretrained(run_args.model_name, cache_dir='cache')
        fast_tokenizer = T5TokenizerFast.from_pretrained(run_args.model_name, cache_dir='cache')
        if run_args.model_path:
            model = MGCEC.from_pretrained(run_args.model_path, cache_dir='cache')
        else:
            model = MGCEC.from_pretrained(run_args.model_name, cache_dir='cache')
    else:
        tokenizer = T5Tokenizer.from_pretrained(run_args.model_name, cache_dir='cache')
        fast_tokenizer = T5TokenizerFast.from_pretrained(run_args.model_name, cache_dir='cache')
        if run_args.model_path:
            model = T5ForConditionalGenerationWoLoss.from_pretrained(run_args.model_path, cache_dir='cache')
        else:
            model = T5ForConditionalGenerationWoLoss.from_pretrained(run_args.model_name, cache_dir='cache')


    if run_args.task == 'generation':
        generate_dataset = GenerateDataset(path_to_data=run_args.valid_file,
                                           max_length=run_args.max_length,
                                           cache_dir='cache',
                                           tokenizer=tokenizer)
        # GenerateDataset
        trainer = DocTqueryTrainer(
            do_generation=True,
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=QueryEvalCollator(
                tokenizer,
                padding='longest',
            ),
        )
        predict_results = trainer.predict(generate_dataset,
                                          top_k=run_args.top_k,
                                          num_return_sequences=run_args.num_return_sequences,
                                          max_length=run_args.q_max_length)
        path = f"{run_args.valid_file}.q{run_args.num_return_sequences}.docTquery"
        print(f"save to path {path}")
        with open(path, 'w') as f:
            for batch_tokens, batch_ids in tqdm(zip(predict_results.predictions, predict_results.label_ids),
                                                desc="Writing file"):
                for tokens, docid in zip(batch_tokens, batch_ids):
                    query = fast_tokenizer.decode(tokens, skip_special_tokens=True)
                    jitem = json.dumps({'text_id': docid.item(), 'text': query})
                    f.write(jitem + '\n')

    elif run_args.task == 'eval' or run_args.task == 'eval_site':
        generate_dataset = EvalDataset(path_to_data=run_args.valid_file,
                                       path_did2queryID=run_args.path_did2queryID,
                                           max_length=run_args.max_length,
                                           cache_dir='cache',
                                           tokenizer=tokenizer)

        trie = load_trie(run_args.trie_path)

        trainer = EvalTrainer(
            do_generation=True,
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=EvalCollator(
                tokenizer,
                padding='longest',
            ),
            trie=trie,
            id_max_length=run_args.id_max_length
        )
        predict_results = trainer.predict(generate_dataset,
                                          top_k=run_args.top_k,
                                          num_return_sequences=run_args.num_return_sequences,
                                          max_length=run_args.q_max_length)
        if run_args.task == 'eval':
            print('----eval_save')
            eval_save(run_args, generate_dataset, predict_results, fast_tokenizer)
        else:
            print('----eval_save_site')

            eval_save_site(run_args, generate_dataset, predict_results, fast_tokenizer)


    elif run_args.task == 'translation':
        train_dataset = TranslationTrainDataset(path_to_data=run_args.train_file,
                                                path_did2queryID=run_args.path_did2queryID,
                                                mode='train',
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             tokenizer=tokenizer)

        valid_dataset = TranslationTrainDataset(path_to_data=run_args.valid_file,
                                                path_did2queryID=run_args.path_did2queryID,
                                                mode='test',
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             remove_prompt=run_args.remove_prompt,
                                             tokenizer=tokenizer)
        ################################################################

        trie = load_trie(run_args.trie_path)


        ################################################################

        trainer = TranslationTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=IndexingCollator(
                tokenizer,
                padding='longest',
            ),
            compute_metrics=make_compute_discrimination_metrics(fast_tokenizer, train_dataset.docid2did, valid_dataset.src, valid_dataset.qids,run_args.eval_files_dir),
            trie=trie,
            id_max_length=run_args.id_max_length
        )
        if not os.path.exists(run_args.eval_files_dir):
            os.makedirs(run_args.eval_files_dir)
        trainer.train()

    elif run_args.task == 'mgcc':
        valid_term = ['1222', '1122', '1111']
        print(f'---------------------valid root:{run_args.mgcc_train_file_root}')
        assert os.path.exists(run_args.mgcc_train_file_root)
        train_files = []
        indexing_11_files = []
        for root, dirs, files in os.walk(run_args.mgcc_train_file_root):
            for f in files:
                if 'indexing' in f or ('11' in f and '1111' not in f and '1122' not in f):
                    indexing_11_files.append(os.path.join(root, f))
                for v in valid_term:
                    if v in f:
                        train_files.append(os.path.join(root, f))
                        break
        train_files.reverse()
        print(f'--main the train files include:{train_files}, indexing or 11 file:{indexing_11_files}')


        valid_dataset = EvalDataset(path_to_data=run_args.valid_file,
                                                   path_did2queryID=run_args.path_did2queryID,
                                                   max_length=run_args.max_length,
                                                   cache_dir='cache',
                                                   tokenizer=tokenizer)

        ################################################################


        train_datasets_lsts = [MGCCTrainMultiDataset(path_to_data=t,
                                             mode='train',
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             tokenizer=tokenizer) for t in train_files]

        if indexing_11_files:
            for i in indexing_11_files:
                dataset_batchsize = run_args.train_indexing_batch_size if 'indexing' in i else 0
                # 0 means uses the per_device_train_batch_size, indexing batch size is 30
                train_datasets_lsts.append(MGCCTrain11Dataset(path_to_data=i,
                                                 mode='train',
                                                 max_length=run_args.max_indexing_length,
                                                 cache_dir='cache',
                                                 dataset_batchsize=dataset_batchsize,
                                                 tokenizer=tokenizer))
            print(f'--main the train files has indexing or 11 include:{train_datasets_lsts}')

        trie = load_trie(run_args.trie_path)

        ################################################################
        # set_contrastive_num(run_args.contrastive_num)
        collator_type = MGCCCollator(tokenizer, padding='longest')
        print(collator_type.__class__)

        trainer = MGCECTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_datasets_lsts[0],
            eval_dataset=valid_dataset,
            data_collator=collator_type,
            compute_metrics=make_compute_discrimination_metrics(fast_tokenizer,
                                                                valid_dataset.docid2did,
                                                                valid_dataset.src, valid_dataset.qids,
                                                                run_args.eval_files_dir),
            trie=trie,
            id_max_length=run_args.id_max_length,
            alpha=run_args.alpha,
            temperature=run_args.temperature,
            include_all_pos=run_args.include_all_pos,
            train_datasets_lsts=train_datasets_lsts,
            mgcc_loss_type=run_args.mgcc_loss_type,
        )

        if not os.path.exists(run_args.eval_files_dir):
            os.makedirs(run_args.eval_files_dir)
        trainer.train()

    elif run_args.task == 'mgcc55':
        valid_term = ['1222', '1122', '1111', '55']
        print(f'---------------------valid root:{run_args.mgcc_train_file_root}')
        assert os.path.exists(run_args.mgcc_train_file_root)
        train_files = []
        indexing_11_files = []
        for root, dirs, files in os.walk(run_args.mgcc_train_file_root):
            for f in files:
                if 'indexing' in f or ('11' in f and '1111' not in f and '1122' not in f):
                    indexing_11_files.append(os.path.join(root, f))
                for v in valid_term:
                    if v in f:
                        train_files.append(os.path.join(root, f))
                        break
        train_files.reverse()
        print(f'--main the train files include:{train_files}, indexing or 11 file:{indexing_11_files}')


        valid_dataset = EvalDataset(path_to_data=run_args.valid_file,
                                                   path_did2queryID=run_args.path_did2queryID,
                                                   max_length=run_args.max_length,
                                                   cache_dir='cache',
                                                   tokenizer=tokenizer)

        ################################################################


        train_datasets_lsts = [MGCCTrain55MultiDataset(path_to_data=t,
                                             mode='train',
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             tokenizer=tokenizer) for t in train_files]

        print(f'--main the train files has indexing or 11 include:{train_datasets_lsts}')

        trie = load_trie(run_args.trie_path)

        ################################################################
        # set_contrastive_num(run_args.contrastive_num)
        collator_type = MGCCCollator(tokenizer, padding='longest')
        print(collator_type.__class__)

        trainer = MGCECTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_datasets_lsts[0],
            eval_dataset=valid_dataset,
            data_collator=collator_type,
            compute_metrics=make_compute_discrimination_metrics(fast_tokenizer,
                                                                valid_dataset.docid2did,
                                                                valid_dataset.src, valid_dataset.qids,
                                                                run_args.eval_files_dir),
            trie=trie,
            id_max_length=run_args.id_max_length,
            alpha=run_args.alpha,
            temperature=run_args.temperature,
            include_all_pos=run_args.include_all_pos,
            train_datasets_lsts=train_datasets_lsts,
            mgcc_loss_type=run_args.mgcc_loss_type,
        )

        if not os.path.exists(run_args.eval_files_dir):
            os.makedirs(run_args.eval_files_dir)
        trainer.train()


    elif run_args.task == "listwise":
        train_dataset = ListMLETrainDataset(path_to_data=run_args.train_file,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             tokenizer=tokenizer)

        valid_dataset = ListMLETrainDataset(path_to_data=run_args.valid_file,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             remove_prompt=run_args.remove_prompt,
                                             tokenizer=tokenizer)

        assert len(train_dataset) % run_args.total_grades == 0
        pattern = r'\d'
        ma = re.findall(pattern, run_args.train_file.split('/')[-1])
        assert run_args.total_grades == int(ma[0])

        ################################################################
        # docid generation constrain, we only generate integer docids.
        SPIECE_UNDERLINE = "▁"
        INT_TOKEN_IDS = []
        for token, id in tokenizer.get_vocab().items():
            if token[0] == SPIECE_UNDERLINE:
                if token[1:].isdigit():
                    INT_TOKEN_IDS.append(id)
            if token == SPIECE_UNDERLINE:
                INT_TOKEN_IDS.append(id)
            elif token.isdigit():
                INT_TOKEN_IDS.append(id)
        INT_TOKEN_IDS.append(tokenizer.eos_token_id)

        def restrict_decode_vocab(batch_idx, prefix_beam):
            return INT_TOKEN_IDS
        ################################################################
        # print('Int token ids')
        # print(f'----------{len(INT_TOKEN_IDS)}')
        # print(INT_TOKEN_IDS.sort())
        trainer = PListMLETrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=ListMLECollator(
                tokenizer,
                padding='longest',
            ),
            compute_metrics=make_compute_metrics_mrr(fast_tokenizer, train_dataset.valid_ids),
            restrict_decode_vocab=restrict_decode_vocab,
            id_max_length=run_args.id_max_length,
            total_grades=run_args.total_grades
        )
        if not os.path.exists(run_args.eval_files_dir):
            os.makedirs(run_args.eval_files_dir)
        trainer.train()

    else:
        raise NotImplementedError("--task should be in 'DSI' or 'docTquery' or 'generation'")


def eval_save(run_args, generate_dataset, predict_results, fast_tokenizer):

    docid2id = generate_dataset.docid2did
    all_src = []
    all_predict = []
    all_label = []
    write_list, eval_list = [], []
    for batch_tokens, batch_ids, src_id, tgt_id, line in tqdm(
            zip(predict_results.predictions, predict_results.label_ids, generate_dataset.src_id,
                generate_dataset.tgt_id, generate_dataset.data),
            desc="Writing file"):
        tgt, src = line
        all_src.append(src_id)
        all_label.append(tgt_id)
        temp_tgt = []
        for tokens, docid in zip(batch_tokens, batch_ids):
            predict = fast_tokenizer.decode(tokens, skip_special_tokens=True)
            if predict in docid2id.keys():
                predict_id = docid2id[predict]
                for id in predict_id:
                    write_list.append(f'{src}\t{src_id}\t{tgt}\t{tgt_id}\t{predict}\t{id}\n')
                    eval_list.append(f'{src_id}\t0\t{id}\t1\n')
                    temp_tgt.append(id)
        all_predict.append(temp_tgt)

    metrics = eval_metrics(all_src, all_predict=all_predict, all_label=all_label)
    M3, M20, H1, H10 = metrics['MRR@3'], metrics['MRR@20'], metrics['Hits@1'], metrics['Hits@10']

    model_name = run_args.model_name.split('/')[-1]
    # valid_file = run_args.valid_file.replace('.json','')
    m = f'{M3}.{M20}.{H1}.{H10}'
    path_result = f"{run_args.eval_files_dir}/{run_args.eval_name}.{model_name}.eval.results.{m}.txt"
    path_eval = f"{run_args.eval_files_dir}/{run_args.eval_name}.{model_name}.eval.{m}.txt"
    print(f"save to path {path_result}")

    save_list_simple(write_list, path_result)
    save_list_simple(eval_list, path_eval)


def eval_save_site(run_args, generate_dataset, predict_results, fast_tokenizer):
    path = f"{run_args.valid_file}.q{run_args.num_return_sequences}.{run_args.model_path}.eval.results.txt"
    path_eval = f"{run_args.valid_file}.q{run_args.num_return_sequences}.{run_args.model_path}.eval.txt"
    print(f"save to path {path}")
    # print(predict_results)
    docid2id = generate_dataset.docid2did
    with open(path, 'w') as f, open(path_eval, 'w') as f_eval:
        for batch_tokens, batch_ids, src_id, tgt_id, line in tqdm(
                zip(predict_results.predictions, predict_results.label_ids, generate_dataset.src_id,
                    generate_dataset.tgt_id, generate_dataset.data),
                desc="Writing file"):
            tgt, src = line
            for tokens, docid in zip(batch_tokens, batch_ids):
                predict = fast_tokenizer.decode(tokens, skip_special_tokens=True)
                # docid = np.where(docid != -100, docid, tokenizer.pad_token_id)
                # docid = fast_tokenizer.decode(docid, skip_special_tokens=True)

                f.write(f'{src}\t{src_id}\t{tgt}\t{tgt_id}\t{predict}\t{docid2id[predict]}\n')
                f_eval.write(f'{src_id}\t0\t{docid2id[predict]}\t1\n')

                # jitem = json.dumps({'text_id': docid, 'text': predict})
                # print(f'predict:{predict}, label:{docid}')
                # f.write(jitem + '\n')v



def load_trie(path):
    with open(path, "rb") as f:
        trie = pickle.load(f)
    print('trie tree loaded')
    return trie


def save_list(eval_files_dir, data, data4eval, mrr3, mrr20, hit1, hit10):

    path = 'test-data-%.4f-%.4f-%.4f-%.4f.tsv'%(mrr3, mrr20, hit1, hit10)
    path = os.path.join(eval_files_dir, path)
    path_eval = 'test-data-for-eval-%.4f-%.4f-%.4f-%.4f.tsv'%(mrr3, mrr20, hit1, hit10)
    path_eval = os.path.join(eval_files_dir, path_eval)

    with open(path, 'w', encoding='utf-8',errors='ignore') as f_d:
        for line in data:
            f_d.write(line.rstrip())
            f_d.write('\n')
    print(f"saving {path}")

    with open(path_eval, 'w', encoding='utf-8', errors='ignore') as f_d:
        for line in data4eval:
            f_d.write(line.rstrip())
            f_d.write('\n')
    print(f"saving {path_eval}")

def save_list_simple(data, path):
    print('saving ', path)
    with open(path, 'w') as f:
        for d in data:
            d = str(d).rstrip()
            f.write(d + '\n')


if __name__ == "__main__":
    main()

