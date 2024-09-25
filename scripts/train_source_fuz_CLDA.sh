gpuid=${1:-0}
random_seed=${2:-2021}

#export CUDA_VISIBLE_DEVICES=$gpuid

echo "PDA SOURCE TRAIN ON OFFICE"
python train_source_fuz.py  --dataset Office --t_idx 0  --target_label_type CLDA --epochs 50 --lr 0.01 
python train_source_fuz.py  --dataset Office --t_idx 1  --target_label_type CLDA --epochs 50 --lr 0.01 
python train_source_fuz.py  --dataset Office --t_idx 2  --target_label_type CLDA --epochs 50 --lr 0.01 

#echo "PDA SOURCE TRAIN ON OFFICEHOME"
#python train_source_fuz.py  --dataset OfficeHome --t_idx 0  --target_label_type CLDA --epochs 50 --lr 0.01 
#python train_source_fuz.py  --dataset OfficeHome --t_idx 1  --target_label_type CLDA --epochs 50 --lr 0.01 
#python train_source_fuz.py  --dataset OfficeHome --t_idx 2  --target_label_type CLDA --epochs 50 --lr 0.01 
#python train_source_fuz.py  --dataset OfficeHome --t_idx 3  --target_label_type CLDA --epochs 50 --lr 0.01 

#echo "CLDA SOURCE TRAIN ON DomainNet"
#python train_source_fuz.py  --dataset DomainNet --t_idx 0  --target_label_type CLDA --epochs 50 --lr 0.01 --backbone_arch resnet101
#python train_source_fuz.py  --dataset DomainNet --t_idx 1  --target_label_type CLDA --epochs 50 --lr 0.01 --backbone_arch resnet101
#python train_source_fuz.py  --dataset DomainNet --t_idx 2  --target_label_type CLDA --epochs 50 --lr 0.01 --backbone_arch resnet101
#python train_source_fuz.py  --dataset DomainNet --t_idx 3  --target_label_type CLDA --epochs 50 --lr 0.01 --backbone_arch resnet101
#python train_source_fuz.py  --dataset DomainNet --t_idx 4  --target_label_type CLDA --epochs 50 --lr 0.01 --backbone_arch resnet101
#python train_source_fuz.py  --dataset DomainNet --t_idx 5  --target_label_type CLDA --epochs 50 --lr 0.01 --backbone_arch resnet101
#python train_source_fuz.py  --dataset DomainNet --t_idx 1  --target_label_type CLDA --epochs 50 --lr 0.01 --backbone_arch resnet101


