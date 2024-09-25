gpuid=${1:-0}
random_seed=${2:-2021}

# export CUDA_VISIBLE_DEVICES=$gpuid

echo "OPDA TSNE ON Office-Home"
python tsne_target_fuz.py --dataset OfficeHome --t_idx 1 --target_label_type OPDA
python tsne_target_fuz.py --dataset OfficeHome --t_idx 2 --target_label_type OPDA
python tsne_target_fuz.py --dataset OfficeHome --t_idx 3 --target_label_type OPDA
python tsne_target_fuz.py --dataset OfficeHome --t_idx 0 --target_label_type OPDA

python tsne_source_fuz.py --dataset OfficeHome --t_idx 1 --target_label_type OPDA
python tsne_source_fuz.py --dataset OfficeHome --t_idx 2 --target_label_type OPDA
python tsne_source_fuz.py --dataset OfficeHome --t_idx 3 --target_label_type OPDA
python tsne_source_fuz.py --dataset OfficeHome --t_idx 0 --target_label_type OPDA



