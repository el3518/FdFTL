import os 
import argparse
import numpy as np 
import torch

def def_comb_rule(corr, args):
    mem_list = []
    cls_list = []
    comb_list = []
    for i in range(args.class_num):#i=0  args.
        idxi = np.where(corr[i,:] >= args.sim) # and corr[i,:] < 1 
        #print(i)
        if i == 0:
            comb_list.append(idxi[0])
            mem_list.extend(idxi[0])
        else:# len(idxi[0]) > 1:            
            count = 0            
            for j in idxi[0]:
                if j not in mem_list:
                    mem_list.extend([j])
                    cls_list.extend([j])
            for c1 in range(len(comb_list)):
                #print(c1)
                if len(set(comb_list[c1]) & set(idxi[0])) > 0 :
                    comb_list[c1] = np.array(list((set(comb_list[c1]) | set(cls_list))))
                    break
                else:
                    count += 1
            if count == len(comb_list):
                comb_list.append(idxi[0])
            cls_list=[]    
    return comb_list


def build_args():
    
    parser = argparse.ArgumentParser("This script is used to Source-free Universal Domain Adaptation")
    
    parser.add_argument("--dataset", type=str, default="OfficeHome")
    parser.add_argument("--backbone_arch", type=str, default="resnet50")
    parser.add_argument("--embed_feat_dim", type=int, default=256)
    #parser.add_argument("--s_idx", type=int, default=0)
    parser.add_argument("--t_idx", type=int, default=0)

    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--epochs", default=50, type=int)
    
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--seed", default=2021, type=int)
    # we set lam_psd to 0.3 for Office and VisDA, 1.5 for OfficeHome and DomainNet
    parser.add_argument("--lam_psd", default=0.3, type=float) 
    parser.add_argument("--lam_knn", default=1.0, type=float)
    parser.add_argument("--local_K", default=4, type=int)
    parser.add_argument("--w_0", default=0.55, type=float)
    parser.add_argument("--rho", default=0.75, type=float)
    
    parser.add_argument("--source_train_type", default="smooth", type=str, help="vanilla, smooth")
    parser.add_argument("--target_label_type", default="OPDA", type=str)
    parser.add_argument("--target_private_class_num", default=None, type=int)
    parser.add_argument("--note", default="GLC_Fuz")
    
    ##################################################################
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"]) 
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    
    
    args = parser.parse_args()
    
    '''
    assume classes across domains are the same.
    [0 1 ............................................................................ N - 1]
    |---- common classes --||---- source private classes --||---- target private classes --|

    |-------------------------------------------------|
    |                DATASET PARTITION                |
    |-------------------------------------------------|
    |DATASET    |  class split(com/sou_pri/tar_pri)   |
    |-------------------------------------------------|
    |DATASET    |    PDA    |    OSDA    | OPDA/UniDA |
    |-------------------------------------------------|
    |Office-31  |  10/21/0  |  10/0/11   |  10/10/11  |
    |-------------------------------------------------|
    |OfficeHome |  25/40/0  |  25/0/40   |  10/5/50   |
    |-------------------------------------------------|
    |VisDA-C    |   6/6/0   |   6/0/6    |   6/3/3    |
    |-------------------------------------------------|  
    |DomainNet  |           |            | 150/50/145 |
    |-------------------------------------------------|
    '''
    ##################################################################
    args.da = 'poda'
    
    if args.dataset == "Office":
        args.sk = 2
        domain_list = ['amazon', 'dslr', 'webcam']
        #args.source_data_dir = os.path.join("./data/Office", domain_list[args.s_idx] )
        args.target_data_dir = os.path.join("./data/Office", domain_list[args.t_idx])
        args.target_domain = domain_list[args.t_idx] 
        
        args.source_domain_list = [domain_list[idx] for idx in range(3) if idx != args.t_idx]
        args.source_domain_dir_list = [os.path.join("./data/Office", item) for item in args.source_domain_list]
        
        folder = "./data/Office/"
        args.s_dset_path = [folder + args.source_domain_list[0] + '.txt', folder + args.source_domain_list[1] + '.txt']
                    
        #args.target_domain_list = [domain_list[idx] for idx in range(3) if idx != args.s_idx]
        #args.target_domain_dir_list = [os.path.join("./data/Office", item) for item in args.target_domain_list]
         
        args.shared_class_num = 10
        
        if args.target_label_type == "PDA":
            args.source_private_class_num = 21
            args.target_private_class_num = 0
            
            args.sim = 0.3
            cen_name = "pda_cen" + str(args.sim)+ ".npy"
        
        elif args.target_label_type == "OSDA":
            args.source_private_class_num = 0
            if args.target_private_class_num is None:
                args.target_private_class_num = 10#21 ################
                
            args.sim = 0.25
            cen_name = "oda_cen" + str(args.sim)+ ".npy"
            
        elif args.target_label_type == "OPDA":
            args.source_private_class_num = 10
            if args.target_private_class_num is None:
                args.target_private_class_num = 11
                
            args.sim = 0.3
            cen_name = "category_cen" + str(args.sim)+ ".npy"
        
        elif args.target_label_type == "CLDA":
            args.shared_class_num = 31 
            args.source_private_class_num = 0
            args.target_private_class_num = 0
            
            args.sim = 0.3
            cen_name = "pda_cen" + str(args.sim)+ ".npy"
        
        else:
            raise NotImplementedError("Unknown target label type specified")
        '''
        args.class_share = [0,1,5,10,11,12,15,16,17,22]
        args.src_private = [2,3,4,6,7,8,9,13,14,18]
        args.tar_private = [19,20,21,23,24,25,26,27,28,29,30]

        args.src_classes = list(set(args.class_share)|set(args.src_private))
        args.tar_classes = list(set(args.class_share)|set(args.tar_private)) 
        '''
        shared_classes = [i for i in range(args.shared_class_num)]
        source_private_classes = [i + args.shared_class_num for i in range(args.source_private_class_num)]
        #target_private_classes = [i + args.shared_class_num + args.source_private_class_num for i in range(args.target_private_class_num)]
            
        
        args.src_classes = shared_classes + source_private_classes
        #args.tar_classes = shared_classes + target_private_classes
        #####################################################################################
        args.class_num = len(args.src_classes)
        #args.sim = 0.3
        #cen_name = "category_cen" + str(args.sim)+ ".npy"
        cen_path = os.path.join("./data/Office", 'fuz_lab_src', cen_name) 
        category_cen = np.load(cen_path)
         #D 0.3 #CR-0.55-3
        corr = np.corrcoef(category_cen)
        corr = np.triu(corr, k=0)   
        args.rule_group = def_comb_rule(corr, args) 
        print(args.rule_group)
        
        args.rule_num = len(args.rule_group)
        args.sel_rule_num = len(args.rule_group)
        
####################################################################### 
    elif args.dataset == "OfficeHome":
        args.sk = 3
        domain_list = ['Art', 'Clipart', 'Product', 'RealWorld']
        #args.source_data_dir = os.path.join("./data/OfficeHome", domain_list[args.s_idx])
        args.target_data_dir = os.path.join("./data/OfficeHome", domain_list[args.t_idx])
        
        args.target_domain = domain_list[args.t_idx] 
        
        args.source_domain_list = [domain_list[idx] for idx in range(args.sk+1) if idx != args.t_idx]
        args.source_domain_dir_list = [os.path.join("./data/OfficeHome", item) for item in args.source_domain_list]
        
        folder = "./data/OfficeHome/"
        args.s_dset_path = [folder + args.source_domain_list[0] + '.txt', 
                            folder + args.source_domain_list[1] + '.txt',
                            folder + args.source_domain_list[2] + '.txt']
     
        #args.target_domain_list = [domain_list[idx] for idx in range(3) if idx != args.s_idx]
        #args.target_domain_dir_list = [os.path.join("./data/Office", item) for item in args.target_domain_list]
         
        #args.target_domain_list = [domain_list[idx] for idx in range(4) if idx != args.s_idx]
        #args.target_domain_dir_list = [os.path.join("./data/OfficeHome", item) for item in args.target_domain_list]
        
        #args.shared_class_num = 10
        
        if args.target_label_type == "PDA":
            args.shared_class_num = 25
            args.source_private_class_num = 40
            args.target_private_class_num = 0
            
            args.sim = 0.58
            cen_name = "pda_cen" + str(args.sim)+ ".npy"
            
        elif args.target_label_type == "OSDA":
            args.shared_class_num = 25
            args.source_private_class_num = 0
            if args.target_private_class_num is None:
                args.target_private_class_num = 40
            
            args.sim = 0.55
            cen_name = "oda_cen" + str(args.sim)+ ".npy"
#######################################################################################       
        elif args.target_label_type == "OPDA":
            args.shared_class_num = 10
            args.source_private_class_num = 5
            if args.target_private_class_num is None:
                args.target_private_class_num = 50
            
            args.sim = 0.55 ##########################################################
            cen_name = "category_cen" + str(args.sim)+ ".npy"
        
        elif args.target_label_type == "CLDA":
            args.shared_class_num = 65 
            args.source_private_class_num = 0
            args.target_private_class_num = 0
            
            args.sim = 0.58
            cen_name = "pda_cen" + str(args.sim)+ ".npy"
        else:
            raise NotImplementedError("Unknown target label type specified")
        
        shared_classes = [i for i in range(args.shared_class_num)]
        source_private_classes = [i + args.shared_class_num for i in range(args.source_private_class_num)]
        #target_private_classes = [i + args.shared_class_num + args.source_private_class_num for i in range(args.target_private_class_num)]
            
        
        args.src_classes = shared_classes + source_private_classes
        #args.tar_classes = shared_classes + target_private_classes
        #####################################################################################
        args.class_num = len(args.src_classes)
        #args.sim = 0.55
        #cen_name = "category_cen" + str(args.sim)+ ".npy"
        cen_path = os.path.join("./data/OfficeHome", 'fuz_lab_src', cen_name) 
        category_cen = np.load(cen_path)
         #D 0.3 #CR-0.55-3
        corr = np.corrcoef(category_cen)
        corr = np.triu(corr, k=0)  
        ########################################################################
        args.sim = 0.55 ### rule number ablation study
        args.rule_group = def_comb_rule(corr, args) 
        print(args.rule_group)
        
        args.rule_num = len(args.rule_group)
        args.sel_rule_num = len(args.rule_group)

    elif args.dataset == "VisDA":
        args.source_data_dir = "./data/VisDA/train/"
        args.target_data_dir = "./data/VisDA/validation/"
        args.target_domain_list = ["validataion"]
        args.target_domain_dir_list = [args.target_data_dir]
        
        args.shared_class_num = 6
        if args.target_label_type == "PDA":
            args.source_private_class_num = 6
            args.target_private_class_num = 0
        
        elif args.target_label_type == "OSDA":
            args.source_private_class_num = 0
            args.target_private_class_num = 6
        
        elif args.target_label_type == "OPDA":
            args.source_private_class_num = 3
            args.target_private_class_num = 3
            
        elif args.target_label_type == "CLDA":
            args.shared_class_num = 12 
            args.source_private_class_num = 0
            args.target_private_class_num = 0
            
        else:
            raise NotImplementedError("Unknown target label type specified", args.target_label_type)
    elif args.dataset == "DomainNet":# and args.target_label_type == "CLDA":
        args.sk = 5
        domain_list = ['clipart','infograph','painting', 'quickdraw', 'real', 'sketch']
        
        args.target_data_dir = os.path.join("./data/DomainNet", domain_list[args.t_idx])
        args.target_domain = domain_list[args.t_idx] 
        
        args.source_domain_list = [domain_list[idx] for idx in range(args.sk+1) if idx != args.t_idx]
        args.source_domain_dir_list = [os.path.join("./data/DomainNet", item) for item in args.source_domain_list]
         
        args.embed_feat_dim = 512 # considering that DomainNet involves more than 256 categories.
        
        args.shared_class_num = 345
        if args.target_label_type == "CLDA":
            args.source_private_class_num = 0
            args.target_private_class_num =  0
            
            args.sim = 0.6
            cen_name = "pda_cen" + str(args.sim)+ ".npy"
        else:
            raise NotImplementedError("Unknown target label type specified")
           
        shared_classes = [i for i in range(args.shared_class_num)]
        source_private_classes = [i + args.shared_class_num for i in range(args.source_private_class_num)]
        target_private_classes = [i + args.shared_class_num + args.source_private_class_num for i in range(args.target_private_class_num)]
            
        
        args.src_classes = shared_classes + source_private_classes
        args.tar_classes = shared_classes + target_private_classes
        
        args.class_num = len(args.src_classes)
        cen_path = os.path.join("./data/DomainNet", 'fuz_lab_src', cen_name) 
        category_cen = np.load(cen_path)
         #D 0.3 #CR-0.55-3
        corr = np.corrcoef(category_cen)
        corr = np.triu(corr, k=0)   
        args.rule_group = def_comb_rule(corr, args) 
        print(args.rule_group)
        
        args.rule_num = len(args.rule_group)
        args.sel_rule_num = len(args.rule_group)    
    elif args.dataset == "DomainNet-":
        args.sk = 2
        domain_list = ["painting", "real", "sketch"]
        
        args.target_data_dir = os.path.join("./data/DomainNet", domain_list[args.t_idx])
        args.target_domain = domain_list[args.t_idx] 
        
        args.source_domain_list = [domain_list[idx] for idx in range(args.sk+1) if idx != args.t_idx]
        args.source_domain_dir_list = [os.path.join("./data/DomainNet", item) for item in args.source_domain_list]
         
        args.embed_feat_dim = 512 # considering that DomainNet involves more than 256 categories.
        
        args.shared_class_num = 150
        if args.target_label_type == "OPDA":
            args.source_private_class_num = 50
            args.target_private_class_num = 145
            
            #args.sim = 0.5
            #cen_name = "category_cen" + str(args.sim)+ ".npy"
        else:
            raise NotImplementedError("Unknown target label type specified")
           
        shared_classes = [i for i in range(args.shared_class_num)]
        source_private_classes = [i + args.shared_class_num for i in range(args.source_private_class_num)]
        target_private_classes = [i + args.shared_class_num + args.source_private_class_num for i in range(args.target_private_class_num)]
            
        
        args.src_classes = shared_classes + source_private_classes
        args.tar_classes = shared_classes + target_private_classes
        
        args.class_num = len(args.src_classes)
        args.sim = 0.5
        cen_name = "category_cen" + str(args.sim)+ ".npy"
        cen_path = os.path.join("./data/DomainNet", 'fuz_lab_src', cen_name) 
        category_cen = np.load(cen_path)
         #D 0.3 #CR-0.55-3
        corr = np.corrcoef(category_cen)
        corr = np.triu(corr, k=0)   
        args.rule_group = def_comb_rule(corr, args) 
        print(args.rule_group)
        
        args.rule_num = len(args.rule_group)
        args.sel_rule_num = len(args.rule_group)
            
    args.source_class_num = args.shared_class_num + args.source_private_class_num
    args.target_class_num = args.shared_class_num + args.target_private_class_num
    args.class_num = args.source_class_num
    
    args.source_class_list = [i for i in range(args.source_class_num)]
    args.target_class_list = [i for i in range(args.shared_class_num)]
    if args.target_private_class_num > 0:
        args.target_class_list.append(args.source_class_num)
    
        

    return args
