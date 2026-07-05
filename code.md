conda activate /media/data3/zhangjy/miniconda3/envs/ImAge
cd /media/data3/zhangjy/EDTformer

# GSV训练
 nohup bash -c 'CUDA_VISIBLE_DEVICES=3 python train.py --eval_datasets_folder=/media/data1/zhangjingyi/datasets --eval_dataset_names pitts30k Msls_740 sped amstertime --train_dataset_path=/home/ittc402-xu4/shared_fast_datasets/vpr_dataset/ReflectCities/gsv/ --foundation_model_path=./dinov2_vitb14_pretrain.pth --epochs_num=20 --train_batch_size=72 --infer_batch_size=16 --resize 322 322' > log_0616_GSV.txt 2>&1 &

# GSVreflect训练
 nohup bash -c 'CUDA_VISIBLE_DEVICES=2 python train.py --eval_datasets_folder=/media/data1/zhangjingyi/datasets --eval_dataset_names pitts30k Msls_740 sped amstertime --foundation_model_path=./dinov2_vitb14_pretrain.pth --epochs_num=20 --train_batch_size=72 --infer_batch_size=16 --resize 322 322' > log_0618_GSVreflect.txt 2>&1 &

# gsv+gift
 nohup bash -c 'CUDA_VISIBLE_DEVICES=3 python train.py \
  --train_dataset_path=/media/data3/zhangjy/gift_GSV/ \
  --eval_datasets_folder=/media/data1/zhangjingyi/datasets \
  --eval_dataset_names pitts30k Msls_740 sped amstertime \
  --foundation_model_path=./dinov2_vitb14_pretrain.pth \
  --epochs_num=20 --train_batch_size=72 --infer_batch_size=16 --resize 322 322' \
  > log_gift_gsv_fixed.txt 2>&1 &

# GSV训练322分辨率
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 python train.py \
  --train_dataset_path=/home/ittc402-xu4/shared_fast_datasets/vpr_dataset/ReflectCities/gsv/ \
  --eval_datasets_folder=/media/data1/zhangjingyi/datasets \
  --eval_dataset_names pitts30k Msls_740 sped amstertime \
  --foundation_model_path=./dinov2_vitb14_pretrain.pth \
  --epochs_num=20 --train_batch_size=72 --infer_batch_size=16 --resize 322 322' \
  > log_0618_GSV_baseline_322.txt 2>&1 &

# GSVMIX 322 finetuning
  nohup bash -c 'CUDA_VISIBLE_DEVICES=0 python train.py \
  --eval_datasets_folder=/media/data1/zhangjingyi/datasets \
  --eval_dataset_names pitts30k Msls_740 \
  --train_dataset_path=/home/ittc402-xu4/shared_fast_datasets/vpr_dataset/ReflectCities/mixed_GSV/ \
  --foundation_model_path=/media/data/zhangjingyi/ImAge/module/EDTformer.pth \
  --epochs_num=20 --train_batch_size=120 --infer_batch_size=16 --resize 322 322 \
  --lr=5e-6 --patience=3 \
  --stage1_epochs=5 --synthetic_ratio=0.125 \
  --save_dir=exp4_EDTformer' \
  > log_0619_finetuning.txt 2>&1 &