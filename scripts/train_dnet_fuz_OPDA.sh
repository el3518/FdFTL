gpuid=${1:-0}
random_seed=${2:-2021}

# export CUDA_VISIBLE_DEVICES=$gpuid


echo "OPDA SOURCE TRAIN ON DomainNet"
python train_source_fuz.py  --dataset DomainNet --t_idx 0  --target_label_type OPDA --epochs 50 --lr 0.01 --backbone_arch resnet101
python train_source_fuz.py  --dataset DomainNet --t_idx 1  --target_label_type OPDA --epochs 50 --lr 0.01 --backbone_arch resnet101
python train_source_fuz.py  --dataset DomainNet --t_idx 2  --target_label_type OPDA --epochs 50 --lr 0.01 --backbone_arch resnet101
