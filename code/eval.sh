
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 trans.py \
        --task eval \
        --model_name /data/users/tangyubao/translation/data/t5 \
        --eval_name eval \
        --max_length 64 \
        --valid_file /data/users/tangyubao/translation/filter-msdoc/trans-alldev/dev20.json \
        --output_dir output/discrimination-indexing \
        --per_device_eval_batch_size 15 \
        --dataloader_num_workers 1 \
        --save_steps 2 \
        --trie_path /data/users/tangyubao/translation/data/trie/t5-base-100k-leadpsg.pkl \
        --path_did2queryID /data/users/tangyubao/translation/filter-msdoc/trans-alldev/did2queryID.tsv \
        --fp16
:<<!

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 trans.py \
        --task multi \
        --model_name /data/users/tangyubao/translation/data/t5 \
        --run_name trans-test-lab \
        --max_length 256 \
        --train_file /data/users/tangyubao/translation/filter-msdoc/trans-alldev/train.qd.20.json \
        --valid_file /data/users/tangyubao/translation/filter-msdoc/trans-alldev/dev20.json \
        --index_file ../data/filter-msdoc-1class/train.index.9.json \
        --output_dir output/discrimination-indexing \
        --learning_rate 0.0005 \
        --warmup_steps 10000 \
        --per_device_train_batch_size 15 \
        --per_device_eval_batch_size 15 \
        --evaluation_strategy steps \
        --eval_steps 2 \
        --max_steps 4 \
        --save_strategy steps \
        --dataloader_num_workers 1 \
        --save_steps 2 \
        --save_total_limit 10 \
        --load_best_model_at_end \
        --gradient_accumulation_steps 2 \
        --report_to wandb \
        --logging_steps 2 \
        --dataloader_drop_last False \
        --metric_for_best_model Hits@10 \
        --greater_is_better True \
        --trie_path /data/users/tangyubao/translation/data/trie/t5-base-100k-leadpsg.pkl \
        --path_did2queryID /data/users/tangyubao/translation/filter-msdoc/trans-alldev/did2queryID.tsv \
        --alpha 1 \
        --beta 1 \
        --contrastive_num 2

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 trans.py \
        --task "translation" \
        --model_name /data/users/tangyubao/translation/data/t5 \
        --run_name trans-indexing \
        --max_length 256 \
        --train_file /data/users/tangyubao/translation/filter-msdoc/trans-alldev/train.dd.pair.fulldoc.json \
        --valid_file /data/users/tangyubao/translation/filter-msdoc/trans-partdev/dev.qd.pair.json \
        --output_dir "output/translation-indexing" \
        --learning_rate 0.0005 \
        --warmup_steps 10000 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --evaluation_strategy steps \
        --eval_steps 1000 \
        --max_steps 100000 \
        --save_strategy steps \
        --dataloader_num_workers 1 \
        --save_steps 1000 \
        --save_total_limit 10 \
        --load_best_model_at_end \
        --gradient_accumulation_steps 300 \
        --report_to wandb \
        --logging_steps 100 \
        --dataloader_drop_last False \
        --metric_for_best_model Hits@10 \
        --greater_is_better True \
        --trie_path /data/users/tangyubao/translation/data/trie/t5-base-100k-leadpsg.pkl \
        --path_did2queryID /data/users/tangyubao/translation/filter-msdoc/trans-alldev/did2queryID.tsv


CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 trans.py \
        --task "discrimination" \
        --model_name /data/users/tangyubao/translation/data/t5 \
        --run_name trans-test-lab \
        --max_length 256 \
        --train_file /data/users/tangyubao/translation/filter-msdoc/trans-alldev/train.dd.pair.fulldoc.json \
        --valid_file /data/users/tangyubao/translation/filter-msdoc/trans-partdev/dev.qd.pair.json \
        --output_dir "output/discrimination-indexing" \
        --learning_rate 0.0005 \
        --warmup_steps 10000 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --evaluation_strategy steps \
        --eval_steps 1000 \
        --max_steps 100000 \
        --save_strategy steps \
        --dataloader_num_workers 1 \
        --save_steps 1000 \
        --save_total_limit 10 \
        --load_best_model_at_end \
        --gradient_accumulation_steps 300 \
        --report_to wandb \
        --logging_steps 100 \
        --dataloader_drop_last False \
        --metric_for_best_model Hits@10 \
        --greater_is_better True \
        --trie_path /data/users/tangyubao/translation/data/trie/t5-base-100k-leadpsg.pkl \
        --path_did2queryID /data/users/tangyubao/translation/filter-msdoc/trans-alldev/did2queryID.tsv

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 trans.py \
        --task "discrimination" \
        --model_name /data/users/tangyubao/translation/data/t5 \
        --run_name trans-test-lab \
        --max_length 256 \
        --train_file /data/users/tangyubao/translation/filter-msdoc/trans-alldev/train.qd.10.json \
        --valid_file /data/users/tangyubao/translation/filter-msdoc/trans-alldev/dev.qd.10.json \
        --output_dir "models/t5-base-DSI-100k-offcila" \
        --learning_rate 0.0005 \
        --warmup_steps 2 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --evaluation_strategy steps \
        --eval_steps 2 \
        --max_steps 4 \
        --save_strategy steps \
        --dataloader_num_workers 1 \
        --save_steps 4 \
        --save_total_limit 10 \
        --load_best_model_at_end \
        --gradient_accumulation_steps 2 \
        --report_to wandb \
        --logging_steps 2 \
        --dataloader_drop_last False \
        --metric_for_best_model Hits@10 \
        --greater_is_better True \
        --trie_path /data/users/tangyubao/translation/data/trie/t5-base-100k-leadpsg.pkl \
        --path_did2queryID /data/users/tangyubao/translation/filter-msdoc/trans-alldev/did2queryID.tsv

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 trans.py \
        --task generation \
        --model_name models/t5base-doc2query/checkpoint-6 \
        --model_path models/t5base-doc2query/checkpoint-6 \
        --per_device_eval_batch_size 4 \
        --run_name docTquery-generation-test-lab \
        --max_length 256 \
        --valid_file data/msmarco_data/100k/cor10.tsv \
        --output_dir temp \
        --dataloader_num_workers 10 \
        --report_to wandb \
        --logging_steps 2 \
        --num_return_sequences 10
!
:<<!
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 trans.py \
        --task "docTquery" \
        --model_name /data/users/tangyubao/translation/data/t5 \
        --run_name trans-lab-test \
        --max_length 128 \
        --train_file data/msmarco_data/100k/train100.json \
        --valid_file data/msmarco_data/100k/train100.json \
        --output_dir "models/t5base-doc2query" \
        --learning_rate 0.0001 \
        --warmup_steps 0 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --evaluation_strategy steps \
        --eval_steps 5 \
        --max_steps 10 \
        --save_strategy steps \
        --dataloader_num_workers 10 \
        --save_steps 5 \
        --save_total_limit 2 \
        --load_best_model_at_end \
        --gradient_accumulation_steps 4 \
        --report_to wandb \
        --logging_steps 2 \
        --dataloader_drop_last False



CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 run.py \
        --task "DSI" \
        --model_name /data/users/tangyubao/translation/data/t5 \
        --run_name lab \
        --max_length 256 \
        --train_file data/msmarco_data/100k/train100.json \
        --valid_file data/msmarco_data/100k/dev100.json \
        --output_dir "models/t5-base-DSI-100k-offcila" \
        --learning_rate 0.0005 \
        --warmup_steps 2 \
        --per_device_train_batch_size 5 \
        --per_device_eval_batch_size 5 \
        --evaluation_strategy steps \
        --eval_steps 5 \
        --max_steps 10 \
        --save_strategy steps \
        --dataloader_num_workers 1 \
        --save_steps 5 \
        --save_total_limit 2 \
        --load_best_model_at_end \
        --gradient_accumulation_steps 2 \
        --report_to wandb \
        --logging_steps 2 \
        --dataloader_drop_last False \
        --metric_for_best_model Hits@10 \
        --greater_is_better True
!
#   --train_file data/msmarco_data/100k/msmarco_DSI_train_data.json \
 #       --valid_file data/msmarco_data/100k/msmarco_DSI_dev_data.json \