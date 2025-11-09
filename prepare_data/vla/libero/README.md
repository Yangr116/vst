# Adapting to VLA Model

## Prepare LIBERO data

### Step1: Download the LIBERO dataset
You should download the LIBERO dataset following [instructions](https://github.com/Lifelong-Robot-Learning/LIBERO?tab=readme-ov-file#Dataset).

### Step2: Preprocess
We follow [OpenVLA](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/regenerate_libero_dataset.py) to filter data:
```
python prepare_data/vla/libero/regenerate_libero_dataset.py \
    --libero_task_suite libero_spatial \
    --libero_raw_data_dir ./LIBERO/libero/datasets/libero_spatial \
    --libero_target_dir ./LIBERO/libero/datasets/libero_spatial_no_noops
# you should replace the path
```
Then, we got processed dataset:
```shell
├── libero_10_no_noops
├── libero_goal_no_noops
├── libero_object_no_noops
└── libero_spatial_no_noops
```

### Step3: Convert to parquet
```shell
python prepare_data/vla/libero/preprocess_libero.py \ 
    --save_dir "./dataset/parquet/vla/libero" \ 
    --libero_dir "./LIBERO/libero/datasets"
```

## Train VLA model

To train the action model on the spatial subset:
```shell
bash scripts/train.sh vlm3d/train_vla.py config/veomni/qwen2_5vla/vla_qwen2_5_vl_fspd1.yaml \ 
    --model.model_path 'VST-3B-SFT' \ 
    --data.train_path 'config/dataconfig/vla/libero_spatial.yaml' \ 
    --data.train_size 5_500_000 \ 
    --train.output_dir 'work_dirs/20250824_vla_qwen2_5vl_3b_spatial_sft_libero_spatial' \ 
    --train.wandb_name '20250824_vla_qwen2_5vl_3b_spatial_sft_libero_spatial' \ 
    --data.num_workers 2 \ 
    --data.buffer_size 6000 \ 
    --train.lr_warmup_ratio 0.03 \ 
    --train.num_train_epochs 50 \ 
    --train.lr 0.00008 \ 
    --train.vit_lr 0.000008 \ 
    --data.max_seq_len 2048 \ 
    --train.micro_batch_size 16
```

## Evaluation on LIBERO

Prepare the LIBERO evaluation env:
```shell
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
cd ..
pip install -r benchmark/libero/libero_requirements.txt
echo N | python -c "from libero.libero import benchmark"
sudo apt-get install libegl-dev -y
```

Then:

```shell
bash benchmark/libero/auto_run_vst_vla_libero_norm.sh \ 
    $your_model_path \ 
    "libero_spatial" \ # This can be "libero_spatial" "libero_object" "libero_goal" "libero_10"
    $your_work_dir
```
