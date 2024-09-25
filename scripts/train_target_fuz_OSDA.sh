gpuid=${1:-0}
random_seed=${2:-2021}

# export CUDA_VISIBLE_DEVICES=$gpuid

lam_psd=0.30
echo "OSDA Adaptation ON Office"
python train_target_fuz.py --dataset Office --t_idx 1 --lr 0.001  --lam_psd $lam_psd --target_label_type OSDA
python train_target_fuz.py --dataset Office --t_idx 2 --lr 0.001  --lam_psd $lam_psd --target_label_type OSDA
python train_target_fuz.py --dataset Office --t_idx 0 --lr 0.001  --lam_psd $lam_psd --target_label_type OSDA


#lam_psd=1.50
#echo "OSDA Adaptation ON Office-Home"
#python train_target_fuz.py --dataset OfficeHome --t_idx 1 --lr 0.001  --lam_psd $lam_psd --target_label_type OSDA
#python train_target_fuz.py --dataset OfficeHome --t_idx 2 --lr 0.001  --lam_psd $lam_psd --target_label_type OSDA
#python train_target_fuz.py --dataset OfficeHome --t_idx 3 --lr 0.001  --lam_psd $lam_psd --target_label_type OSDA
#python train_target_fuz.py --dataset OfficeHome --t_idx 0 --lr 0.001  --lam_psd $lam_psd --target_label_type OSDA
