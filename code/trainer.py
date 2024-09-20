import math
from typing import Dict, List, Tuple, Optional, Any, Union
from transformers.trainer import Trainer
# from base_trainer import Trainer
from torch import nn
from torch.utils.data import Dataset
from torch.nn import functional as F
import torch
import collections
from transformers.file_utils import is_datasets_available
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    LengthGroupedSampler,
    IterableDatasetShard
)
from collections.abc import Mapping
_is_torch_generator_available = False
if is_datasets_available():
    import datasets
from torch.utils.data import Dataset, IterableDataset, RandomSampler, SequentialSampler,DataLoader,ConcatDataset, BatchSampler, SubsetRandomSampler
# from custom_dataloader import DataLoader
from torch.nn import CrossEntropyLoss
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from transformers.generation_utils import GreedySearchOutput
from transformers.models.gpt2 import GPT2LMHeadModel
from data import ComboBatchSampler, CombinationDataset, CustomComboBatchSampler, DiscriminationCollator, MixComboBatchSampler, DynamicComboBatchSampler, MGCCSampler, MGCCBatchSampler, ListwiseSampler
from con_loss import SupConLoss, HMLC



class DSITrainer(Trainer):
    def __init__(self, restrict_decode_vocab, id_max_length, **kwds):
        super().__init__(**kwds)
        self.restrict_decode_vocab = restrict_decode_vocab
        self.id_max_length = id_max_length

    def compute_loss(self, model, inputs, return_outputs=False):
        loss = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels']).loss
        print(loss)
        if return_outputs:
            return loss, [None, None]  # fake outputs
        return loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.eval()
        # eval_loss = super().prediction_step(model, inputs, True, ignore_keys)[0]
        inputs['labels'] = inputs['labels'].to(self.args.device)
        with torch.no_grad():
            # Greedy search
            # doc_ids = model.generate(
            #     inputs['input_ids'].to(self.args.device),
            #     max_length=20,
            #     prefix_allowed_tokens_fn=self.restrict_decode_vocab,
            #     early_stopping=True,)

            # Beam search
            batch_beams = model.generate(
                inputs['input_ids'].to(self.args.device),
                max_length=20,
                num_beams=20,
                prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                num_return_sequences=20,
                early_stopping=True, )

            if batch_beams.shape[-1] < self.id_max_length:
                batch_beams = self._pad_tensors_to_max_len(batch_beams, self.id_max_length)
            if inputs['labels'].shape[-1] < self.id_max_length:
                inputs['labels'] = self._pad_tensors_to_max_len(inputs['labels'], self.id_max_length)

            batch_beams = batch_beams.reshape(inputs['input_ids'].shape[0], 20, -1)

        return (None, batch_beams, inputs['labels'])

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")
        tensor[tensor == -100] = self.tokenizer.pad_token_id
        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor


class DocTqueryTrainer(Trainer):
    def __init__(self, do_generation: bool, **kwds):
        super().__init__(**kwds)
        self.do_generation = do_generation

    def compute_loss(self, model, inputs, return_outputs=False):
        loss = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels']).loss
        if return_outputs:
            return loss, [None, None]  # fake outputs
        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        if not self.do_generation:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        outputs = self.model.generate(
            input_ids=inputs[0]['input_ids'].to(self.args.device),
            attention_mask=inputs[0]['attention_mask'].to(self.args.device),
            max_length=self.max_length,
            do_sample=True,
            top_k=self.top_k,
            num_return_sequences=self.num_return_sequences)
        labels = torch.tensor(inputs[1], device=self.args.device).repeat_interleave(self.num_return_sequences)

        if outputs.shape[-1] < self.max_length:
            outputs = self._pad_tensors_to_max_len(outputs, self.max_length)
        return (None, outputs.reshape(inputs[0]['input_ids'].shape[0], self.num_return_sequences, -1),
                labels.reshape(inputs[0]['input_ids'].shape[0], self.num_return_sequences, -1))

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    def predict(
            self,
            test_dataset: Dataset,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "test",
            max_length: Optional[int] = None,
            num_return_sequences: Optional[int] = None,
            top_k: Optional[int] = None,
    ):

        self.max_length = max_length
        self.num_return_sequences = num_return_sequences
        self.top_k = top_k
        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)


class EvalTrainer(Trainer):
    def __init__(self, trie, id_max_length, tokenizer, do_generation: bool, **kwds):
        super().__init__(**kwds)
        self.do_generation = do_generation
        self.trie = trie
        self.id_max_length = id_max_length
        self.tokenizer = tokenizer

    def compute_loss(self, model, inputs, return_outputs=False):
        loss = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels']).loss
        if return_outputs:
            return loss, [None, None]  # fake outputs
        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.eval()

        if not self.do_generation:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )
        inputs['labels'] = inputs['labels'].to(self.args.device)

        with torch.no_grad():
            # outputs = self.model.generate(
            #     input_ids=inputs['input_ids'].to(self.args.device),
            #     attention_mask=inputs['attention_mask'].to(self.args.device),
            #     max_length=self.max_length,
            #     do_sample=True,
            #     top_k=self.top_k,
            #     prefix_allowed_tokens_fn=lambda batch_id, sent: self.trie.get(sent.tolist()),
            #     num_return_sequences=self.num_return_sequences)

            outputs = model.generate(
                inputs['input_ids'].to(self.args.device),
                max_length=self.id_max_length,
                num_beams=self.num_return_sequences,
                prefix_allowed_tokens_fn=lambda batch_id, sent: self.trie.get(sent.tolist()),
                num_return_sequences=self.num_return_sequences,
                early_stopping=True,
                output_scores=True,
                return_dict_in_generate=False)
        # outputs_list = build_outputs(outputs, tokenizer=self.tokenizer, num_return_sequences=self.num_return_sequences)
        # print('----------------------')
        # print(outputs_list)




        if outputs.shape[-1] < self.id_max_length:
            outputs = self._pad_tensors_to_max_len(outputs, self.max_length)

        if inputs['labels'].shape[-1] < self.id_max_length:
            inputs['labels'] = self._pad_tensors_to_max_len(inputs['labels'], self.max_length)
        labels = inputs['labels'].repeat_interleave(self.num_return_sequences, dim=0)
        # labels = torch.tensor(inputs[1], device=self.args.device).repeat_interleave(self.num_return_sequences)

        return (None, outputs.reshape(inputs['input_ids'].shape[0], self.num_return_sequences, -1),
                labels.reshape(inputs['input_ids'].shape[0], self.num_return_sequences, -1))

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    def predict(
            self,
            test_dataset: Dataset,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "test",
            max_length: Optional[int] = None,
            num_return_sequences: Optional[int] = None,
            top_k: Optional[int] = None,
    ):

        self.max_length = max_length
        self.num_return_sequences = num_return_sequences
        self.top_k = top_k
        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)


class TranslationTrainer(Trainer):
    def __init__(self, trie, id_max_length, **kwds):
        super().__init__(**kwds)
        self.trie = trie
        self.id_max_length = id_max_length

    def compute_loss(self, model, inputs, return_outputs=False):
        loss = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels']).loss
        # print(loss)
        if return_outputs:
            return loss, [None, None]  # fake outputs
        return loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.eval()
        # eval_loss = super().prediction_step(model, inputs, True, ignore_keys)[0]
        inputs['labels'] = inputs['labels'].to(self.args.device)

        with torch.no_grad():
            # Greedy search
            # doc_ids = model.generate(
            #     inputs['input_ids'].to(self.args.device),
            #     max_length=20,
            #     prefix_allowed_tokens_fn=self.restrict_decode_vocab,
            #     early_stopping=True,)

            # Beam search
            batch_beams = model.generate(
                inputs['input_ids'].to(self.args.device),
                max_length=20,
                num_beams=20,
                prefix_allowed_tokens_fn=lambda batch_id, sent: self.trie.get(sent.tolist()),
                num_return_sequences=20,
                early_stopping=True, )

            if batch_beams.shape[-1] < self.id_max_length:
                batch_beams = self._pad_tensors_to_max_len(batch_beams, self.id_max_length)
            if inputs['labels'].shape[-1] < self.id_max_length:
                inputs['labels'] = self._pad_tensors_to_max_len(inputs['labels'], self.id_max_length)

            batch_beams = batch_beams.reshape(inputs['input_ids'].shape[0], 20, -1)

        return (None, batch_beams, inputs['labels'])

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")
        tensor[tensor == -100] = self.tokenizer.pad_token_id
        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor



class SiteTrainer(Trainer):
    def __init__(self, trie, id_max_length, **kwds):
        super().__init__(**kwds)
        self.trie = trie
        self.id_max_length = id_max_length

    def compute_loss(self, model, inputs, return_outputs=False):
        loss = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels']).loss
        if return_outputs:
            return loss, [None, None]  # fake outputs
        return loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.eval()
        # eval_loss = super().prediction_step(model, inputs, True, ignore_keys)[0]
        inputs['labels'] = inputs['labels'].to(self.args.device)

        with torch.no_grad():
            # Greedy search
            # doc_ids = model.generate(
            #     inputs['input_ids'].to(self.args.device),
            #     max_length=20,
            #     prefix_allowed_tokens_fn=self.restrict_decode_vocab,
            #     early_stopping=True,)

            # Beam search
            batch_beams = model.generate(
                inputs['input_ids'].to(self.args.device),
                max_length=20,
                num_beams=20,
                prefix_allowed_tokens_fn=lambda batch_id, sent: self.trie.get(sent.tolist()),
                num_return_sequences=20,
                early_stopping=True, )

            if batch_beams.shape[-1] < self.id_max_length:
                batch_beams = self._pad_tensors_to_max_len(batch_beams, self.id_max_length)
            if inputs['labels'].shape[-1] < self.id_max_length:
                inputs['labels'] = self._pad_tensors_to_max_len(inputs['labels'], self.id_max_length)

            batch_beams = batch_beams.reshape(inputs['input_ids'].shape[0], 20, -1)

        return (None, batch_beams, inputs['labels'])

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")
        tensor[tensor == -100] = self.tokenizer.pad_token_id
        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor


class DiscriminationTrainer(Trainer):
    def __init__(self, trie, id_max_length, alpha=1, temperature=0.1, include_all_pos=False, **kwds):
        super().__init__(**kwds)
        self.trie = trie
        self.id_max_length = id_max_length
        self.temperature = temperature
        self.include_all_pos = include_all_pos
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False):
        # print(self.temperature, self.alpha, self.include_all_pos)
        # print(inputs)

        device = inputs['labels'].device
        batch_size = inputs['input_ids'].size(0)
        meta_num = int(torch.sqrt(torch.tensor(batch_size).to(torch.double)).item())
        # print(f'batch size {batch_size}')
        # print(f'metanum {meta_num}')

        logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels']).logits
        loss_fct_reduction_none = CrossEntropyLoss(ignore_index=-100, reduction='none')
        loss_matrix = loss_fct_reduction_none(logits.view(-1, logits.size(-1)), inputs['labels'].view(-1))
        # print(loss2)

        length = inputs['labels'].size(1)
        meta_logits = loss_matrix[torch.tensor([l + i * (meta_num + 1) * length for i in range(meta_num) for l in range(length)])]
        meta_labels = inputs['labels'][torch.tensor([i * (meta_num + 1) for i in range(meta_num)])]
        meta_labels_size = meta_labels[:, :][meta_labels[:, :] > -1].shape[0]
        mse_loss = (meta_logits / meta_labels_size).sum()

        # print(f'meta logits size {meta_logits.size()}, logits size:{logits.size()}')

        probs = torch.exp(-loss_matrix).reshape(batch_size,-1)
        tgt_len = [inputs['labels'][i, :][inputs['labels'][i, :] > -1].shape[0] for i in range(batch_size)]
        tgt_len = torch.tensor(tgt_len).to(device)
        scores = probs.sum(dim=-1) / tgt_len

        scores = scores.reshape(meta_num, -1)
        scores = torch.div(scores, self.temperature)
        logits_max, _ = torch.max(scores, dim=1, keepdim=True)
        smooth_scores = scores - logits_max.detach()
        exp_score = torch.exp(smooth_scores)

        # pos_mask = torch.eye(meta_num, meta_num).to(device)
        pos_mask = inputs['flag'].reshape(meta_num, meta_num).to(device)
        neg_mask = (1 - pos_mask).to(device)

        if self.include_all_pos:
            de = (exp_score * pos_mask).sum(dim=1, keepdim=True) + (exp_score * neg_mask).sum(dim=1, keepdim=True)

        else:
            de = exp_score + (exp_score * neg_mask).sum(dim=1, keepdim=True)

        contrastive_logits = (smooth_scores - torch.log(de)) * pos_mask
        contrastive_log_probs = (contrastive_logits * pos_mask).sum(dim=1) / pos_mask.sum(dim=1)
        contrastive_loss = -contrastive_log_probs.mean()

        with open('./all_loss.tsv','a+') as f:
            f.write(f'mse :{mse_loss}, con :{contrastive_loss}\n')
        # print(f'mse :{mse_loss}, con :{contrastive_loss}')
        loss = mse_loss + self.alpha * contrastive_loss

        if return_outputs:
            return loss, [None, None]  # fake outputs
        return loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.eval()
        # eval_loss = super().prediction_step(model, inputs, True, ignore_keys)[0]
        inputs['labels'] = inputs['labels'].to(self.args.device)

        with torch.no_grad():
            # Greedy search
            # doc_ids = model.generate(
            #     inputs['input_ids'].to(self.args.device),
            #     max_length=20,
            #     prefix_allowed_tokens_fn=self.restrict_decode_vocab,
            #     early_stopping=True,)

            # Beam search
            batch_beams = model.generate(
                inputs['input_ids'].to(self.args.device),
                max_length=20,
                num_beams=20,
                prefix_allowed_tokens_fn=lambda batch_id, sent: self.trie.get(sent.tolist()),
                num_return_sequences=20,
                early_stopping=True, )

            if batch_beams.shape[-1] < self.id_max_length:
                batch_beams = self._pad_tensors_to_max_len(batch_beams, self.id_max_length)
            if inputs['labels'].shape[-1] < self.id_max_length:
                inputs['labels'] = self._pad_tensors_to_max_len(inputs['labels'], self.id_max_length)

            batch_beams = batch_beams.reshape(inputs['input_ids'].shape[0], 20, -1)

        return (None, batch_beams, inputs['labels'])

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")
        tensor[tensor == -100] = self.tokenizer.pad_token_id
        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    def _get_train_sampler(self):
        # print('---xxxxsampler')
        if not isinstance(self.train_dataset, collections.abc.Sized):
            return None

        # return BatchSampler(sampler=SequentialSampler(self.train_dataset), batch_size=self.args.train_batch_size,drop_last=False)
        return RandomSampler(self.train_dataset)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `self.train_dataset` does not implement `__len__`, a random sampler (adapted to
        distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            print('no sampler')
            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler()

        print('has sampler')
        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            shuffle=False,
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        # drop_last = self.args.dataloader_drop_last,

        # return DataLoader(
        #     train_dataset,
        #     batch_sampler=train_sampler,
        #     collate_fn=self.data_collator,
        #     num_workers=self.args.dataloader_num_workers,
        #     pin_memory=self.args.dataloader_pin_memory,
        # )


class DisMultiGranularityTrainer(Trainer):
    def __init__(self, trie, id_max_length, index_dataset, contrastive_num, alpha=1, beta=1, temperature=0.1, include_all_pos=False,  **kwds):
        super().__init__(**kwds)
        self.trie = trie
        self.id_max_length = id_max_length
        self.temperature = temperature
        self.include_all_pos = include_all_pos
        self.alpha = alpha # for con
        self.beta = beta
        self.index_dataset = index_dataset
        self.contrastive_num = contrastive_num

    def compute_loss(self, model, inputs, return_outputs=False):
        # print(self.temperature, self.alpha, self.include_all_pos)
        # print(inputs[0]['labels'])

        self.total = inputs['input_ids'].size(0)
        self.flag = [inputs['flag'][i] for i in range(self.total)]
        self.length = inputs['labels'].size(1)
        self.device = inputs['labels'].device

        logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                       labels=inputs['labels']).logits
        loss_fct_reduction_none = CrossEntropyLoss(ignore_index=-100, reduction='none')
        self.loss_matrix = loss_fct_reduction_none(logits.view(-1, logits.size(-1)), inputs['labels'].view(-1)) # [total*len]

        if 'Contrastive' in self.flag and len(set(self.flag)) > 1:
            # mix
            loss = self.compute_mix_loss(inputs)

        elif 'Contrastive' in self.flag and len(set(self.flag)) == 1:
            # only contrastive
            loss = self.compute_contrastive_loss(inputs)
        else:
            loss = self.compute_normal_loss(inputs)

        loss_ = loss.item()
        if isinstance(loss_, float):
            print(f'loss {loss_}, float, {set(self.flag)}')
        else:
            print(f'loss {loss_}, not float, {set(self.flag)}')
        if return_outputs:
            return loss, [None, None]  # fake outputs
        return loss

    def compute_mix_loss(self, inputs):
        normal_size = self.flag.index('Contrastive')
        con_num = self.contrastive_num
        self.index_size = self.total - normal_size
        self.index_group_size = int(int(self.index_size / con_num) ** 0.5)

        # normal loss
        normal_logits = self.loss_matrix[:normal_size * self.length]
        normal_labels = inputs['labels'][:normal_size]

        pos_index_logits = self.loss_matrix[torch.tensor(
            [con_num * j * (self.index_group_size + 1) * self.length + i * self.length + l + normal_size * self.length for j in
             range(self.index_group_size) for i in range(con_num) for l in range(self.length)])]
        pos_index_labels = inputs['labels'][
            torch.tensor([con_num * j * (self.index_group_size + 1) + i + normal_size for j in range(self.index_group_size) for i in range(con_num)])]

        mle_logits = torch.vstack((normal_logits.view(-1, self.length), pos_index_logits.view(-1, self.length))).view(-1)
        mle_labels = torch.vstack((normal_labels, pos_index_labels))

        mle_labels_size = mle_labels[:, :][mle_labels[:, :] > -1].shape[0]
        mle_loss = (mle_logits / mle_labels_size).sum()

        # contrastive loss
        index_loss_matix = self.loss_matrix[normal_size * self.length:]
        index_labels = inputs['labels'][normal_size:]
        # index_pos_num = index_group_size * 3
        #

        # probs = torch.exp(-index_loss_matix).reshape(index_size, -1)
        # tgt_len = [index_labels[i, :][index_labels[i, :] > -1].shape[0] for i in range(index_size)]
        # tgt_len = torch.tensor(tgt_len).to(self.device)
        # scores = probs.sum(dim=-1) / tgt_len
        #
        # scores = scores.reshape(index_pos_num, -1)
        # scores = torch.div(scores, self.temperature)
        # logits_max, _ = torch.max(scores, dim=1, keepdim=True)
        # smooth_scores = scores - logits_max.detach()
        # exp_score = torch.exp(smooth_scores)
        #
        # pos_mask = torch.zeros((index_pos_num, index_group_size)).to(self.device)
        # for i in range(index_pos_num):
        #     pos_mask[i][i//3] = 1
        #
        # neg_mask = torch.ones((index_pos_num, index_group_size)).to(self.device)
        # neg_mask = neg_mask - pos_mask
        #
        # if self.include_all_pos == 'True':
        #     de = (exp_score * pos_mask).sum(dim=1, keepdim=True) + (exp_score * neg_mask).sum(dim=1, keepdim=True)
        #
        # else:
        #     de = exp_score + (exp_score * neg_mask).sum(dim=1, keepdim=True)
        #
        # logits = (smooth_scores - torch.log(de)) * pos_mask
        # log_probs = (logits * pos_mask).sum(dim=1) / pos_mask.sum(dim=1)
        # contrastive_loss = -log_probs.mean()
        contrastive_loss = self.contrastive_loss(index_labels, index_loss_matix)

        loss = self.beta * mle_loss + self.alpha * contrastive_loss
        with open('./all_loss.tsv', 'a+') as f:
            f.write(f'beta:{self.beta}, mle_loss:{mle_loss}, alpha:{self.alpha}, con_loss:{contrastive_loss}\n')

        return loss

    def compute_normal_loss(self, inputs):
        # print('----normal type')
        labels_size = inputs['labels'][:, :][inputs['labels'][:, :] > -1].shape[0]
        mle_loss = (self.loss_matrix / labels_size).sum()
        return mle_loss

    def compute_contrastive_loss(self, inputs):
        con_num = self.contrastive_num
        self.index_size = self.total
        self.index_group_size = int(int(self.index_size / con_num) ** 0.5)

        pos_index_logits = self.loss_matrix[torch.tensor(
            [con_num * j * (self.index_group_size + 1) * self.length + i * self.length + l for j in
             range(self.index_group_size) for i in range(con_num) for l in range(self.length)])]
        pos_index_labels = inputs['labels'][
            torch.tensor(
                [con_num * j * (self.index_group_size + 1) + i for j in range(self.index_group_size) for i in range(con_num)])]

        mle_labels_size = pos_index_labels[:, :][pos_index_labels[:, :] > -1].shape[0]
        mle_loss = (pos_index_logits / mle_labels_size).sum()

        # contrastive loss
        # index_pos_num = index_group_size * 3
        #
        # probs = torch.exp(-self.loss_matrix).reshape(index_size, -1)
        # tgt_len = [inputs['labels'][i, :][inputs['labels'][i, :] > -1].shape[0] for i in range(index_size)]
        # tgt_len = torch.tensor(tgt_len).to(self.device)
        # scores = probs.sum(dim=-1) / tgt_len
        #
        # scores = scores.reshape(index_pos_num, -1)
        # scores = torch.div(scores, self.temperature)
        # logits_max, _ = torch.max(scores, dim=1, keepdim=True)
        # smooth_scores = scores - logits_max.detach()
        # exp_score = torch.exp(smooth_scores)
        #
        # pos_mask = torch.zeros((index_pos_num, index_group_size)).to(self.device)
        # for i in range(index_pos_num):
        #     pos_mask[i][i // 3] = 1
        #
        # neg_mask = torch.ones((index_pos_num, index_group_size)).to(self.device)
        # neg_mask = neg_mask - pos_mask
        #
        # if self.include_all_pos == 'True':
        #     de = (exp_score * pos_mask).sum(dim=1, keepdim=True) + (exp_score * neg_mask).sum(dim=1, keepdim=True)
        #
        # else:
        #     de = exp_score + (exp_score * neg_mask).sum(dim=1, keepdim=True)
        #
        # logits = (smooth_scores - torch.log(de)) * pos_mask
        # log_probs = (logits * pos_mask).sum(dim=1) / pos_mask.sum(dim=1)
        # contrastive_loss = -log_probs.mean()
        contrastive_loss = self.contrastive_loss(inputs['labels'], self.loss_matrix)

        loss = self.beta * mle_loss + self.alpha * contrastive_loss

        return loss

    def contrastive_loss(self, labels, index_loss_matix):
        # contrastive loss
        con_num = self.contrastive_num
        index_pos_num = self.index_group_size * con_num

        probs = torch.exp(-index_loss_matix).reshape(self.index_size, -1)
        tgt_len = [labels[i, :][labels[i, :] > -1].shape[0] for i in range(self.index_size)]
        tgt_len = torch.tensor(tgt_len).to(self.device)
        scores = probs.sum(dim=-1) / tgt_len

        scores = scores.reshape(index_pos_num, -1)
        scores = torch.div(scores, self.temperature)
        logits_max, _ = torch.max(scores, dim=1, keepdim=True)
        smooth_scores = scores - logits_max.detach()
        exp_score = torch.exp(smooth_scores)

        pos_mask = torch.zeros((index_pos_num, self.index_group_size)).to(self.device)
        for i in range(index_pos_num):
            pos_mask[i][i // con_num] = 1

        neg_mask = torch.ones((index_pos_num, self.index_group_size)).to(self.device)
        neg_mask = neg_mask - pos_mask

        if self.include_all_pos == 'True':
            de = (exp_score * pos_mask).sum(dim=1, keepdim=True) + (exp_score * neg_mask).sum(dim=1, keepdim=True)

        else:
            de = exp_score + (exp_score * neg_mask).sum(dim=1, keepdim=True)

        logits = (smooth_scores - torch.log(de)) * pos_mask
        log_probs = (logits * pos_mask).sum(dim=1) / pos_mask.sum(dim=1)
        contrastive_loss = -log_probs.mean()
        return contrastive_loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.eval()
        # eval_loss = super().prediction_step(model, inputs, True, ignore_keys)[0]
        inputs['labels'] = inputs['labels'].to(self.args.device)

        with torch.no_grad():
            # Greedy search
            # doc_ids = model.generate(
            #     inputs['input_ids'].to(self.args.device),
            #     max_length=20,
            #     prefix_allowed_tokens_fn=self.restrict_decode_vocab,
            #     early_stopping=True,)

            # Beam search
            batch_beams = model.generate(
                inputs['input_ids'].to(self.args.device),
                max_length=20,
                num_beams=20,
                prefix_allowed_tokens_fn=lambda batch_id, sent: self.trie.get(sent.tolist()),
                num_return_sequences=20,
                early_stopping=True, )

            if batch_beams.shape[-1] < self.id_max_length:
                batch_beams = self._pad_tensors_to_max_len(batch_beams, self.id_max_length)
            if inputs['labels'].shape[-1] < self.id_max_length:
                inputs['labels'] = self._pad_tensors_to_max_len(inputs['labels'], self.id_max_length)

            batch_beams = batch_beams.reshape(inputs['input_ids'].shape[0], 20, -1)

        return (None, batch_beams, inputs['labels'])

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")
        tensor[tensor == -100] = self.tokenizer.pad_token_id
        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    def _get_train_sampler(self):
        print('---xxxxsampler')
        if not isinstance(self.train_dataset, collections.abc.Sized):
            return None

        # return RandomSampler(self.train_dataset)
        # return MixComboBatchSampler(
        #     [RandomSampler(self.train_dataset), SubsetRandomSampler([i for i in range(len(self.index_dataset)//3)])],
        #     batch_size=self.args.train_batch_size, drop_last=True)

        return DynamicComboBatchSampler(
            [RandomSampler(self.train_dataset), SubsetRandomSampler([i for i in range(len(self.index_dataset)// self.contrastive_num)])],
            batch_size=self.args.train_batch_size, drop_last=True, contrastive_num=self.contrastive_num)


    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `self.train_dataset` does not implement `__len__`, a random sampler (adapted to
        distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            print('no sampler')
            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler()

        # print('has sampler')
        # return DataLoader(
        #     train_dataset,
        #     batch_size=self.args.train_batch_size,
        #     sampler=train_sampler,
        #     shuffle=False,
        #     collate_fn=self.data_collator,
        #     drop_last=True,
        #     num_workers=self.args.dataloader_num_workers,
        #     pin_memory=self.args.dataloader_pin_memory,
        # )
        # drop_last = self.args.dataloader_drop_last,

        return DataLoader(
            CombinationDataset([train_dataset, self.index_dataset]),
            batch_sampler=train_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        # return DataLoader(
        #     ConcatDataset([train_dataset, self.eval_dataset]),
        #     sampler=train_sampler,
        #     collate_fn=self.data_collator,
        #     num_workers=self.args.dataloader_num_workers,
        #     pin_memory=self.args.dataloader_pin_memory,
        # )


class MGCECTrainer(Trainer):
    def __init__(self, trie, id_max_length, train_datasets_lsts, alpha=1, beta=1, temperature=0.1, include_all_pos=False, layer_penalty=None, mgcc_loss_type='hmce', **kwds):
        super().__init__(**kwds)
        self.trie = trie
        self.id_max_length = id_max_length
        self.temperature = temperature
        self.include_all_pos = include_all_pos
        self.alpha = alpha  # for con
        self.beta = beta
        # self.train_dataset_122 = train_dataset_122
        # self.train_dataset_111 = train_dataset_111
        self.train_datasets_lsts = train_datasets_lsts
        # self.hmlc_loss = HMLC(temperature=0.07, base_temperature=0.07, layer_penalty=None, loss_type='hmce')

        if not layer_penalty:
            self.layer_penalty = self.pow_2
        else:
            self.layer_penalty = layer_penalty

        self.loss_type = mgcc_loss_type
        # self.contrastive_type = '222'


    def pow_2(self, value):
        return torch.pow(2, value)

    def get_grades_table(self):
        if self.contrastive_type == '1222':
            self.grades_table = torch.tensor([[4,1],[3,2],[2,2],[1,2]])
        elif self.contrastive_type == '1122':
            self.grades_table = torch.tensor([[4,1],[3,1],[2,2],[1,2]])
        elif self.contrastive_type == '1111':
            self.grades_table = torch.tensor([[4,1],[3,1],[2,1],[1,1]])
        elif self.contrastive_type == '11':
            self.grades_table = torch.tensor([[1,1]])
        elif self.contrastive_type == '55':
            self.grades_table = torch.tensor([[2, 5],[1,5]])
        else:
            print(f'-------wrong {self.contrastive_type}')

        return self.grades_table

    def compute_loss(self, model, inputs, return_outputs=False):
        # print(self.temperature, self.alpha, self.include_all_pos)
        # print(inputs[0]['labels'])
        mode = inputs['mode']
        self.contrastive_type = inputs['type']
        print(f'--cpt loss contype:{self.contrastive_type}, mode:{mode}' )
        if self.contrastive_type == 'indexing' or mode == 'test':
            # print(f'--cpt loss indexing mode:{mode} con type:{self.contrastive_type}')

            logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                  labels=inputs['labels'], eval=False, indexing=True).logits
            loss = self.compute_mle_loss(logits, inputs['labels'])
            # print(f'--cpt loss indexing loss:{loss}, mode:{mode}')

        else:
            # print(f'--cpt loss mgcc loss, con type:{self.contrastive_type}')

            loss = self.mgcc_loss(model, inputs)

        loss_ = loss.item()
        # if isinstance(loss_, float):
        #     print(f'loss {loss_}, float')
        # else:
        #     print(f'loss {loss_}, not float')
        if return_outputs:
            return loss, [None, None]  # fake outputs
        return loss

    def mgcc_loss(self, model, inputs):
        model_outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                              labels=inputs['labels'], eval=False)
        logits, avg_docid_rep, query_rep = model_outputs.logits, model_outputs.avg_docid_rep, model_outputs.query_rep  # lo[14,9 v], avg_docid_rep [b,768], q[14,768]
        # print(f'--model mode:{mode}, con type:{self.contrastive_type}, lo size:{logits.size()},avg doc size:{avg_docid_rep.size()}, q rep size:{query_rep.size()}')
        mle_loss = self.compute_mle_loss(logits, inputs['labels'])

        # print(mode)
        # print(inputs)
        # input_ids_lsts = inputs['input_ids']
        self.total = inputs['input_ids'].size(0)
        # self.flag = [inputs['flag'][i] for i in range(self.total)]
        self.length = inputs['labels'].size(1)
        self.device = inputs['labels'].device

        # self.grades_table = inputs['table']
        self.grades_table = self.get_grades_table().to(self.device)
        self.grades = self.grades_table.shape[0]
        # print(f'----total:{self.total}, sum :{torch.sum(self.grades_table, dim=0)}')
        total_pos_per_group = torch.sum(self.grades_table, dim=0)[1].item()  # sum of the coloum
        group = int(torch.div(self.total, total_pos_per_group))
        # print(f'---grades table:{self.grades_table}, self.grades;{self.grades}, total pos per group:{total_pos_per_group}, group:{group}')

        scores = torch.matmul(query_rep, avg_docid_rep.T)  # [14,14]
        # print('--total scores all ele > 0', torch.all(torch.ge(scores, 0)))
        base = 0
        cumulative_loss = torch.tensor(0.0).to(self.device)
        max_loss_lower_layer = torch.tensor(float('-inf'))

        for l in range(self.grades):
            new_scores = torch.vstack([torch.hstack([scores[
                                                     base + g * total_pos_per_group:base + g * total_pos_per_group +
                                                                                    self.grades_table[l][1].item(),
                                                     gg * total_pos_per_group + base: (gg + 1) * total_pos_per_group]
                                                     for gg in range(group)]) for g in range(group)])
            smooth_scores, exp_score = self.compute_score(new_scores)  # smooth < 0 , exp >0
            pos = torch.zeros((new_scores.shape)).to(self.device)
            # print(
                # f'--l:{l}, scores new scores > 0: {torch.all(torch.ge(new_scores, 0))}, smooth score > 0: {torch.all(torch.ge(smooth_scores, 0))}, exp score >0 : {torch.all(torch.ge(exp_score, 0))}')

            for g in range(group):
                row_min = g * self.grades_table[l][1]
                row_max = g * self.grades_table[l][1] + self.grades_table[l][1]
                pos[row_min:row_max,
                g * (total_pos_per_group - base): g * (total_pos_per_group - base) + total_pos_per_group - base] = 1
                # print(f'-----l:{l}, row_min:{row_min}, row max:{row_max}, col min:{g * (total_pos_per_group - base)}, col max:{g * (total_pos_per_group - base) + total_pos_per_group - base}')
            # print(f'--l:{l},  base:{base}, pos:{pos}')

            base = base + self.grades_table[l][1].item()

            neg = 1 - pos
            layer_loss = self.con_loss(exp_score, smooth_scores, pos, neg)
            # print(f'--l:{l}, layer loss:{layer_loss}')

            if self.loss_type == 'hmc':
                cumulative_loss += self.layer_penalty(torch.tensor(
                    1 / self.grades_table[l][0]).type(torch.float)) * layer_loss
            elif self.loss_type == 'hce':
                layer_loss = torch.max(max_loss_lower_layer.to(layer_loss.device), layer_loss)
                cumulative_loss += layer_loss
            elif self.loss_type == 'hmce':
                layer_loss = torch.max(max_loss_lower_layer.to(layer_loss.device), layer_loss)
                cumulative_loss += self.layer_penalty(torch.tensor(
                    1 / self.grades_table[l][0]).type(torch.float)) * layer_loss
            else:
                raise NotImplementedError('Unknown loss')
            # print(f'---for loss type:{self.loss_type}, cmulative loss:{cumulative_loss}, layer loss:{layer_loss}')

            with open('./all_loss.tsv', 'a+') as f:
                f.write(
                    f'mle loss :{mle_loss}, l: {l}, layer loss :{layer_loss}, max loss:{max_loss_lower_layer}, cum loss:{cumulative_loss}, con type:{self.contrastive_type}\n')

        mgccloss = cumulative_loss / self.grades_table.shape[0]
        loss = self.beta * mle_loss + self.alpha * mgccloss
        with open('./all_loss.tsv', 'a+') as f:
            f.write(f'total loss :{loss}, mle:{mle_loss}, mgcc loss:{mgccloss}\n')

        return loss

    def compute_score(self, scores):
        scores = torch.div(scores, self.temperature)
        logits_max, _ = torch.max(scores, dim=1, keepdim=True)
        smooth_scores = scores - logits_max.detach()
        exp_score = torch.exp(smooth_scores)  # [14,768]
        return smooth_scores, exp_score

    def con_loss(self, exp_score, smooth_scores, pos_mask, neg_mask):
        if self.include_all_pos == 'True':
            de = (exp_score * pos_mask).sum(dim=1, keepdim=True) + (exp_score * neg_mask).sum(dim=1, keepdim=True)
        else:
            de = exp_score + (exp_score * neg_mask).sum(dim=1, keepdim=True) # >0

        logits = (smooth_scores - torch.log(de)) * pos_mask # <0
        log_probs = (logits * pos_mask).sum(dim=1) / pos_mask.sum(dim=1) #<0
        contrastive_loss = -log_probs.mean()
        # print(f'--con loss  de > 0:{torch.all(torch.ge(de, 0))}, logits > 0:{torch.all(torch.ge(logits, 0))}, log probs > 0 :{torch.all(torch.ge(log_probs, 0))}, con loss:{contrastive_loss}')

        return contrastive_loss


    def compute_mle_loss(self, lm_logits, labels):
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        return loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.eval()
        # eval_loss = super().prediction_step(model, inputs, True, ignore_keys)[0]
        inputs['labels'] = inputs['labels'].to(self.args.device)

        with torch.no_grad():
            # Greedy search
            # doc_ids = model.generate(
            #     inputs['input_ids'].to(self.args.device),
            #     max_length=20,
            #     prefix_allowed_tokens_fn=self.restrict_decode_vocab,
            #     early_stopping=True,)

            # Beam search
            gen = model.generate(
                inputs['input_ids'].to(self.args.device),
                max_length=20,
                num_beams=20,
                prefix_allowed_tokens_fn=lambda batch_id, sent: self.trie.get(sent.tolist()),
                num_return_sequences=20,
                early_stopping=True,
                output_scores=True,
                return_dict_in_generate = True,
                )
            # print(f'-----gen---- {gen}')

            batch_beams = gen['sequences']
            score = gen['sequences_scores']
            # print(f'--------- {batch_beams.shape}')
            # print(f'----score--shape--- {score.shape}')
            # print(f'---score------ {score}')



            # output_scores = True,
            # return_dict_in_generate = True
            # print(batch_beams)

            if batch_beams.shape[-1] < self.id_max_length:
                batch_beams = self._pad_tensors_to_max_len(batch_beams, self.id_max_length)
            if inputs['labels'].shape[-1] < self.id_max_length:
                inputs['labels'] = self._pad_tensors_to_max_len(inputs['labels'], self.id_max_length)

            batch_beams = batch_beams.reshape(inputs['input_ids'].shape[0], 20, -1)
            score = score.reshape(inputs['input_ids'].shape[0], 20)
            # print(f'--after-------batch beamsL{batch_beams}')

        return (None, batch_beams, inputs['labels'], score)

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")
        tensor[tensor == -100] = self.tokenizer.pad_token_id
        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    def _get_train_sampler(self):
        # print('---xxxxsampler')
        if not isinstance(self.train_dataset, collections.abc.Sized):
            return None

        # return MGCCBatchSampler(
        #     [RandomSampler(self.train_dataset), RandomSampler(self.train_dataset_122),
        #      RandomSampler(self.train_dataset_111)],
        #     batch_size=self.args.train_batch_size, drop_last=True, contrastive_num=3, typelen=[7, 6, 4])
        typelens = [t.typelen for t in self.train_datasets_lsts]
        dataset_batchsize_lst = [t.dataset_batchsize for t in self.train_datasets_lsts]
        print(f'---get train sampler, typelens:{typelens}, batchsize lst:{dataset_batchsize_lst}')
        return MGCCBatchSampler(
            [RandomSampler(t) for t in self.train_datasets_lsts],
            batch_size=self.args.train_batch_size, drop_last=True, typelen=typelens, dataset_batchsize_lst=dataset_batchsize_lst)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `self.train_dataset` does not implement `__len__`, a random sampler (adapted to
        distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            print('no sampler')
            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler()

        # print('has sampler')
        # return DataLoader(
        #     train_dataset,
        #     batch_size=self.args.train_batch_size,
        #     sampler=train_sampler,
        #     shuffle=False,
        #     collate_fn=self.data_collator,
        #     drop_last=True,
        #     num_workers=self.args.dataloader_num_workers,
        #     pin_memory=self.args.dataloader_pin_memory,
        # )
        # drop_last = self.args.dataloader_drop_last,

        return DataLoader(
            CombinationDataset(self.train_datasets_lsts),
            batch_sampler=train_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


class PListMLETrainer(Trainer):
    def __init__(self, restrict_decode_vocab, id_max_length, total_grades,  **kwds):
        super().__init__(**kwds)
        self.id_max_length = id_max_length
        self.total_grades = total_grades
        self.pad_token_id = 0 # t5
        self.restrict_decode_vocab = restrict_decode_vocab



    def compute_loss(self, model, inputs, return_outputs=False):
        self.device = inputs['labels'].device
        # for k, v in inputs.items(): # input_ids [b,seq,len] end with 1, pad with 0,att word 1,other 0, label end with 1,pad with -100
        #     print(f'--{k}:{v.size()}--{v}')
        # label_size = inputs['labels'].size() # [bz,len]
        # print(f'--lable size:{label_size}')  #
        logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                       labels=inputs['labels']).logits
        mle_loss = self.compute_mle_loss(logits, inputs['labels'])
        listloss = self.compute_plistMLE_loss(inputs, logits)

        loss = mle_loss + listloss

        loss_ = loss.item()
        if isinstance(loss_, float):
            print(f'loss {loss_}, float')
        else:
            print(f'loss {loss_}, not float')
        if return_outputs:
            return loss, [None, None]  # fake outputs
        return loss

    def compute_plistMLE_loss(self, inputs, output):
        batch_size = output.size(0)
        group = int(batch_size / self.total_grades)
        # print(f'---output size:{output.size()}')
        # output = logits  # [bz , seq_len, word_dim]
        output = output.view(group, -1, output.size(1), output.size(2))  # [group, cand_num, seq_len, word_dim]
        # print(f'---output view size:{output.size()}')

        output = output[:, :, :-1]  # truncate last token
        label_size = inputs['labels'].size()
        candidate_id = inputs['labels'].view(group, -1, label_size[-1])[:, :, :-1]
        cand_mask = candidate_id != self.pad_token_id
        ms = inputs['label_mask']
        # print(f'--cand_mask:{cand_mask}')
        # print(f'--lable mask:{ms}')
        candidate_id = candidate_id.unsqueeze(-1)

        normalize = True
        score_mode = 'base'
        adding = 0
        length_penalty = 1
        hyper = 1e-10
        if normalize:
            if score_mode == "log":
                _output = F.log_softmax(output, dim=3)
            else:
                _output = F.softmax(output, dim=3)
            print(f'---output size:{output.size()}, _out:{_output.size()},candi.size:{candidate_id.size()}')
            scores = torch.gather(_output, 3, candidate_id).squeeze(-1)  # [group, cand_num, seq_len]
        else:
            scores = torch.gather(output, 3, candidate_id).squeeze(-1)  # [group, cand_num, seq_len]
        cand_mask = cand_mask.float()
        scores = torch.mul(scores, cand_mask).sum(-1) / (
                    (cand_mask.sum(-1) + adding) ** length_penalty)  # [group, cand_num]

        cumsums = scores.exp().flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
        listmle_loss = torch.log(cumsums + hyper) - scores

        listmle_loss *= self.position_weight()

        listmle_loss = listmle_loss.sum(dim=1).mean()
        return listmle_loss

    def position_weight(self):
        weight = [math.pow(2, g) - 1 for g in range(self.total_grades,0,-1)]
        weight = torch.tensor(weight).to(self.device)
        return weight


    def compute_mle_loss(self, lm_logits, labels):
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        return loss



    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.eval()
        # eval_loss = super().prediction_step(model, inputs, True, ignore_keys)[0]
        inputs['labels'] = inputs['labels'].to(self.args.device)

        with torch.no_grad():
            # Greedy search
            # doc_ids = model.generate(
            #     inputs['input_ids'].to(self.args.device),
            #     max_length=20,
            #     prefix_allowed_tokens_fn=self.restrict_decode_vocab,
            #     early_stopping=True,)

            # Beam search
            batch_beams = model.generate(
                inputs['input_ids'].to(self.args.device),
                max_length=20,
                num_beams=20,
                prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                num_return_sequences=20,
                early_stopping=True, )

            if batch_beams.shape[-1] < self.id_max_length:
                batch_beams = self._pad_tensors_to_max_len(batch_beams, self.id_max_length)
            if inputs['labels'].shape[-1] < self.id_max_length:
                inputs['labels'] = self._pad_tensors_to_max_len(inputs['labels'], self.id_max_length)

            batch_beams = batch_beams.reshape(inputs['input_ids'].shape[0], 20, -1)

        return (None, batch_beams, inputs['labels'])

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")
        tensor[tensor == -100] = self.tokenizer.pad_token_id
        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    def _get_train_sampler(self):
        # print('---xxxxsampler')
        if not isinstance(self.train_dataset, collections.abc.Sized):
            return None

        return ListwiseSampler(self.train_dataset, self.total_grades)




class cat_dataloaders():
    """Class to concatenate multiple dataloaders"""

    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        len(self.dataloaders)

    def __iter__(self):
        self.loader_iter = []
        for data_loader in self.dataloaders:
            self.loader_iter.append(iter(data_loader))
        return self

    def __len__(self):
        cnt = 0
        for dataloader in self.dataloaders:
            cnt += len(dataloader)
        return cnt

    def __next__(self):
        out = []
        for data_iter in self.loader_iter:
            out.append(next(data_iter)) # may raise StopIteration
        return tuple(out)


def build_outputs(outputs, tokenizer, num_return_sequences):
    # from genre.utils import chunk_it

    return chunk_it(
        [
            {
                "text": text,
                "score": score,
            }
            for text, score in zip(
            tokenizer.batch_decode(
                outputs.sequences, skip_special_tokens=True
            ),
            outputs.sequences_scores,
        )
        ],
        num_return_sequences,
    )

def chunk_it(seq, num):
    assert num > 0
    # chunk_len = len(seq) // num
    # chunks = [seq[i * chunk_len : i * chunk_len + chunk_len] for i in range(num)]

    chunk_len = num
    group = len(seq) // num
    chunks = [seq[i * chunk_len: i * chunk_len + chunk_len] for i in range(group)]

    # diff = len(seq) - chunk_len * num
    # for i in range(diff):
    #     chunks[i].append(seq[chunk_len * num + i])

    return chunks
