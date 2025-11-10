### Train
#### Stage 1: SFT
```shell
bash scripts/train.sh vlm3d/train.py config/veomni/qwen2_5vl/qwen2_5_vl_32b_fspd1_fov_packing_llavaov800k.yaml \ 
    --model.model_path 'Qwen2.5-VL-3B-Instruct' \ 
    --data.train_path 'config/dataconfig/parquet/toy_sft.yaml' \ 
    --data.train_size 13700_000_000 \ 
    --train.output_dir 'work_dir/vst_3b_toy_sft' \ 
    --train.wandb_name 'vst_3b_toy_sft'
```

CoT Cold Start:
```shell
bash scripts/train.sh vlm3d/train.py config/veomni/qwen2_5vl/qwen2_5_vl_32b_fspd1_fov_packing_llavaov800k.yaml \ 
    --model.model_path 'trained' \ 
    --data.train_path 'config/dataconfig/parquet/toy_sft.yaml' \ 
    --data.train_size 13700_000_000 \ 
    --train.output_dir 'work_dir/vst_3b_toy_sft' \ 
    --train.wandb_name 'vst_3b_toy_sft'
```


## Adapt to VLA Model

Please refer to [this doc](./vla.md)
