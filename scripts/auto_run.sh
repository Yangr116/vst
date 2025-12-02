
bash scripts/debug.sh vst/train.py config/veomni/qwen2_5_vl_fspd1_fov_packing_example.yaml \
    --model.model_path 'Qwen2.5-VL-3B-Instruct' \
    --data.train_path 'config/data/llavanext.yaml' \
    --data.train_size 20_531_761 \
    --train.output_dir 'work_dir/qwen2_5vl_sft_llavanext_example' \
    --train.wandb_name 'qwen2_5vl_sft_llavanext_example'
