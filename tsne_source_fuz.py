import os
import faiss
import torch 
import shutil 
import numpy as np

from tqdm import tqdm 
from model.SFUniDA_fuz import SFUniDA
from dataset.dataset import SFUniDADataset
from torch.utils.data.dataloader import DataLoader

from config.model_config_fuz import build_args
from utils.net_utils import set_logger, set_random_seed
from utils.net_utils import compute_h_score, Entropy

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

best_score = 0.0
best_coeff = 1.0
  
@torch.no_grad()
def get_feat(args, model, dataloader, src_flg=False):
    
    model.eval()
    gt_label_stack = []
    embed_feat_bank = []
    
    if src_flg:
        class_list = args.source_class_list
        open_flg = False
    else:
        class_list = args.target_class_list
        open_flg = args.target_private_class_num > 0
    
    for _, imgs_test, imgs_label, _ in tqdm(dataloader, ncols=60):
        
        imgs_test = imgs_test.cuda() 
        _, embed_feat, _, _ = model(imgs_test, apply_softmax=False)
        
        gt_label_stack.append(imgs_label)
        embed_feat_bank.append(embed_feat)
    
    gt_label_all = torch.cat(gt_label_stack, dim=0) #[N]
    embed_feat_bank = torch.cat(embed_feat_bank, dim=0) #[N, D]

    return gt_label_all, embed_feat_bank 

   
def main(args):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
           
    model = SFUniDA(args)
    
    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(args.checkpoint)
        raise ValueError("YOU MUST SET THE APPROPORATE SOURCE CHECKPOINT FOR TARGET MODEL ADPTATION!!!")
#############################################################################    
    model = model.cuda()

    ##############################################
    
    target_data_list = open(os.path.join(args.target_data_dir, "image_unida_list.txt"), "r").readlines()
    target_dataset = SFUniDADataset(args, args.target_data_dir, target_data_list, d_type="target", preload_flg=True)
    
    target_test_dataloader = DataLoader(target_dataset, batch_size=args.batch_size*2, shuffle=False,
                                        num_workers=args.num_workers, drop_last=False)
       
    gt_label_all, embed_feat_bank  = get_feat(args, model, target_test_dataloader, src_flg=False)
    
    labels = gt_label_all.detach().cpu().numpy()
    
    data = embed_feat_bank.detach().cpu().numpy()
    
    tsne = TSNE(n_components = 2)
    data_tsne = tsne.fit_transform(data)    
    data_min, data_max = data_tsne.min(0), data_tsne.max(0)
    data_norm = (data_tsne - data_min) / (data_max - data_min)

    fig = plt.figure(figsize = (5, 5))  
    plt.scatter(data_norm[:,0], data_norm[:,1], c=labels, s=5, cmap = plt.cm.get_cmap("jet", args.class_num), marker = '.')#, alpha = 0.5, fontdict = {'weight': 'bold', 'size': 14}                
    plt.show()
    fig.savefig('figs/tsne-{}-{}-son.jpg'.format(args.dataset, args.t_idx), bbox_inches='tight', dpi=300, pad_inches=0.02)
    
        
if __name__ == "__main__":
    args = build_args()
    set_random_seed(args.seed)
    args.rule_num = 4
    args.t_idx = 3
    domain_list = ['Art', 'Clipart', 'Product', 'RealWorld']
    #args.source_data_dir = os.path.join("./data/OfficeHome", domain_list[args.s_idx])
    args.target_data_dir = os.path.join("./data/OfficeHome", domain_list[args.t_idx])
        
    args.target_domain = domain_list[args.t_idx] 

######################################################################################################    
    # SET THE CHECKPOINT     
    args.checkpoint = os.path.join("checkpoints_glc_fuz", args.dataset, str(args.rule_num), "for_target_{}".format(args.t_idx),\
                    "source_{}_{}".format(args.source_train_type, args.target_label_type),
                    "latest_source_checkpoint.pth")
    main(args)
