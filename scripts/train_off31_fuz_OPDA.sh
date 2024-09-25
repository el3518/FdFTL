gpuid=${1:-0}
random_seed=${2:-2021}

# export CUDA_VISIBLE_DEVICES=$gpuid 
# --epochs 50

echo "OPDA SOURCE TRAIN ON OFFICE"
python train_source_fuz.py  --dataset Office --t_idx 0  --target_label_type OPDA --epochs 50 --lr 0.01 
python train_source_fuz.py  --dataset Office --t_idx 1  --target_label_type OPDA --epochs 50 --lr 0.01 
python train_source_fuz.py  --dataset Office --t_idx 2  --target_label_type OPDA --epochs 50 --lr 0.01 
