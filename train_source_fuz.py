import os
import shutil
import torch
import numpy as np 
from tqdm import tqdm 
from model.SFUniDA_fuz import SFUniDA
from dataset.dataset import SFUniDADataset
from torch.utils.data.dataloader import DataLoader 

from config.model_config_fuz import build_args
from utils.net_utils import set_logger, set_random_seed
from utils.net_utils import compute_h_score, CrossEntropyLabelSmooth

from scipy.spatial.distance import cdist
import random, pdb, math, copy

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def norm_fea(fea):    
    fea = torch.cat((fea, torch.ones(fea.size(0), 1)), 1)
    fea = (fea.t() / torch.norm(fea, p=2, dim=1)).t()   
    return fea

def clu_mem(fea, cen, args):  
    if args.distance == 'cosine':
        fea = norm_fea(fea).numpy()
    else:
        fea = (fea.t() / torch.norm(fea, p=2, dim=1)).t() 
    dist_c = cdist(fea, cen, args.distance)
    dist_a = (1/(1e-8 + dist_c)).sum(axis=1)
    dist_a = np.expand_dims(dist_a, axis=1)
    dda=dist_a.repeat(cen.shape[0], axis=1)
    #mem_ship = nn.Softmax(dim=1)(torch.from_numpy(1/(1e-8 + (dist_c*dda)))).numpy() 
    mem_ship = torch.from_numpy(1/(1e-8 + (dist_c*dda))).numpy()     
        
    return mem_ship

def fuz_mem(fea, cen, args):
    
    mem_ship = clu_mem(fea.detach().cpu(), cen, args)
    tar_rule = np.argsort(-mem_ship, axis=1)
    mem_ship = torch.from_numpy(mem_ship).float().cuda()            
         
    return mem_ship, tar_rule

def cal_output(output, member, args):    
    outputs = torch.zeros([output[0].shape[0], args.class_num]).cuda()
    for i in range(len(output)):                       
        outputs += member[:,i].reshape(output[0].shape[0],1)*output[i]               			
    return outputs

def train(cen, args, model, dataloader, criterion, optimizer, epoch_idx=0.0):#idx = 0
    model.train() #
    loss_stack = []
    #rule_cen_s[idx], args, model, source_dataloader[idx], criterion, optimizer, epoch_idx
    #cen = rule_cen_s[idx] dataloader = source_dataloader[idx]
    iter_idx = epoch_idx * len(dataloader)
    iter_max = args.epochs * len(dataloader)
  
    for imgs_train, _, imgs_label, imgs_idx in tqdm(dataloader, ncols=60):
        
        iter_idx += 1
        ###########################################################
        
        imgs_train = imgs_train.cuda()
        imgs_label = imgs_label.cuda()
        
        ########################################################
        #lr_scheduler(optimizer, iter_idx, iter_max)
        #optimizer.zero_grad()
        
        ##########################
        bb_fea, fea, pred_cls, pred_rule = model(imgs_train, apply_softmax=False)
        
        mem_ship, tar_rule = fuz_mem(bb_fea, cen, args)  
        pred_cls_final = cal_output(pred_cls, mem_ship, args)
        
        #cls_out = torch.softmax(cls_out, dim=1)
        
        imgs_onehot_label = torch.zeros_like(pred_cls_final).scatter(1, imgs_label.unsqueeze(1), 1)
        loss = criterion(torch.softmax(pred_cls_final, dim=1), imgs_onehot_label)
        
        for idx in range(len(pred_cls)):
            loss += criterion(torch.softmax(pred_cls[idx], dim=1), imgs_onehot_label)
        
        #loss = criterion(pred_cls, imgs_onehot_label)
        #clu_label_org = imgs_label  ####keep original labels
        rule_label = gen_clu_lab(imgs_label.cpu(), args.rule_group).cuda()
        rule_onehot_label = torch.zeros_like(pred_rule).scatter(1, rule_label.unsqueeze(1), 1)
        loss += criterion(torch.softmax(pred_rule, dim=1), rule_onehot_label)
        
        
        lr_scheduler(optimizer, iter_idx, iter_max)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_stack.append(loss.cpu().item())
        
    train_loss = np.mean(loss_stack)
    
    return train_loss, model.state_dict()

@torch.no_grad()
def test(args, model, dataloader, src_flg=True):
    #model.eval()
    gt_label_stack = []
    pred_cls_stack = []
    cen = clu_anc_para(model.rule_layer, args)
    #cen = clu_rule_cen(dataloader, model, args)
    
    if src_flg:
        class_list = args.source_class_list
        open_flg = False
    else:
        class_list = args.target_class_list
        open_flg = args.target_private_class_num > 0
    
    for _, imgs_test, imgs_label, _ in tqdm(dataloader, ncols=60):
        
        imgs_test = imgs_test.cuda()
        bb_fea, fea, pred_cls, _ = model(imgs_test, apply_softmax=False)
        
        #####################################
        mem_ship, tar_rule = fuz_mem(bb_fea, cen, args)  
        pred_cls = torch.softmax(cal_output(pred_cls, mem_ship, args), dim=1)        
        #cls_out = torch.softmax(cls_out, dim=1)
        
        gt_label_stack.append(imgs_label)
        pred_cls_stack.append(pred_cls.cpu())
        
    gt_label_all = torch.cat(gt_label_stack, dim=0) #[N]
    pred_cls_all = torch.cat(pred_cls_stack, dim=0) #[N, C]
    
    h_score, known_acc,\
    unknown_acc, per_cls_acc = compute_h_score(args, class_list, gt_label_all, pred_cls_all, open_flg, open_thresh=0.50)
    
    return h_score, known_acc, unknown_acc, per_cls_acc

def clu_anc_para(net, args):#netC=netC1
    para=[]
    for k,v in net.named_parameters():
        v.requires_grad = False
        last=k.rfind('v')
        if k[last] =='v':
            #print(v.shape)
            para.append(v.cpu().detach())   
                
    #anc = np.zeros([para[0].shape[0], para[0].shape[1]])
    anc = np.zeros([para[0].shape[0], para[0].shape[1]+1])
    for i in range(len(para)):#i=0
        parai= para[i]
        parai = norm_fea(parai).numpy()
        anc = anc + parai
    anc = anc / len(para) 
    return anc

def gen_clu_lab(label, rule_num):
    #clu_label = label 
    for i in range(len(rule_num)):#i=0, rule_num=rule_num1, label=label1
        idx = []
        for c in rule_num[i]:#c=0
            #print(c)
            idxi = np.where(label == c)
            idx.extend(idxi[0])
        label[idx] = i
    
    return label

@torch.no_grad()
def clu_rule_cen(dataloader, model, args): #dataloader = source_dataloader[idx] idx = 0
    gt_label_bank = []
    embed_feat_bank = []
#with torch.no_grad():      
    for _, imgs_test, imgs_label, _ in tqdm(dataloader, ncols=60):
        
        imgs_test = imgs_test.cuda()
        _, embed_feat, _ , _ = model(imgs_test, apply_softmax=False)
        embed_feat_bank.append(embed_feat)
        gt_label_bank.append(imgs_label.cuda())

    gt_label_bank = torch.cat(gt_label_bank, dim=0) #[N]
    embed_feat_bank = torch.cat(embed_feat_bank, dim=0) #[N, D]
    #embed_feat_bank = embed_feat_bank / torch.norm(embed_feat_bank, p=2, dim=1, keepdim=True)
    embed_feat_bank = embed_feat_bank.cpu()
    if args.distance == 'cosine':
        embed_feat_bank = torch.cat((embed_feat_bank, torch.ones(embed_feat_bank.size(0), 1)), 1)
        embed_feat_bank = embed_feat_bank / torch.norm(embed_feat_bank, p=2, dim=1, keepdim=True)
    else:
        embed_feat_bank = embed_feat_bank / torch.norm(embed_feat_bank, p=2, dim=1, keepdim=True)
    
    embed_feat_bank = embed_feat_bank.cpu().numpy()
    
    label_clu = gt_label_bank.int().cpu().numpy()
    label_clu = gen_clu_lab(label_clu, args.rule_group)
    
    K = args.rule_num
    aff = np.eye(K)[label_clu] #label vector sample size classes  .int()
    cls_count = np.eye(K)[label_clu].sum(axis=0)
    
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]
    
    initc = update_cen_src(aff, embed_feat_bank, labelset, K, args)
    
    return initc  #, torch.from_numpy(gt_label_bank).cuda()

def update_cen_src(aff, all_fea, labelset, K, args):
    
    initc0 = aff.transpose().dot(all_fea)
    initc0 = initc0 / (1e-8 + aff.sum(axis=0)[:,None]) 
    
    dist_c = cdist(all_fea, initc0[labelset], args.distance)#
    dist_a = (1/(1e-8 + dist_c)).sum(axis=1)
    dist_a = np.expand_dims(dist_a, axis=1)
    dda=dist_a.repeat(len(labelset), axis=1)
    mem_ship = torch.from_numpy(1/(1e-8 + (dist_c*dda))).numpy() 

    for round in range(1):
        aff = np.power(mem_ship,2)#mem_ship #########################
        initc1 = aff.transpose().dot(all_fea)
        initc1 = initc1 / (1e-8 + aff.sum(axis=0)[:,None])   
    
    return initc1

@torch.no_grad()
def clu_rule_cen_b(dataloader, model, args):
    gt_label_bank = []
    backb_feat_bank = []
#with torch.no_grad():    
    for _, imgs_test, imgs_label, _ in tqdm(dataloader, ncols=60):
        
        imgs_test = imgs_test.cuda()
        backb_feat, _, _ , _ = model(imgs_test, apply_softmax=False)
        backb_feat_bank.append(backb_feat)
        gt_label_bank.append(imgs_label.cuda())

    gt_label_bank = torch.cat(gt_label_bank, dim=0) #[N]
    backb_feat_bank = torch.cat(backb_feat_bank, dim=0) #[N, D]
    #backb_feat_bank = backb_feat_bank / torch.norm(backb_feat_bank, p=2, dim=1, keepdim=True)
    backb_feat_bank = backb_feat_bank.cpu()
    if args.distance == 'cosine':
        backb_feat_bank = torch.cat((backb_feat_bank, torch.ones(backb_feat_bank.size(0), 1)), 1)
        backb_feat_bank = backb_feat_bank / torch.norm(backb_feat_bank, p=2, dim=1, keepdim=True)
    else:
        backb_feat_bank = backb_feat_bank / torch.norm(backb_feat_bank, p=2, dim=1, keepdim=True)
    
    backb_feat_bank = backb_feat_bank.cpu().numpy()
    
    label_clu = gt_label_bank.int().cpu().numpy()
    label_clu = gen_clu_lab(label_clu, args.rule_group)
    
    K = args.rule_num
    aff = np.eye(K)[label_clu] #label vector sample size classes  .int()
    cls_count = np.eye(K)[label_clu].sum(axis=0)
    
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]
    
    initc = update_cen_src(aff, backb_feat_bank, labelset, K, args)
    
    return initc#, torch.from_numpy(gt_label_bank).cuda()


def avg_model(net, args):
    net_avg = copy.deepcopy(net[0])
    for key in net_avg.keys():
        for i in range(1,args.sk):
            net_avg[key] += net[i][key] 
        net_avg[key] = torch.div(net_avg[key], args.sk)
    return net_avg
  
def main(args):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    this_dir = os.path.join(os.path.dirname(__file__), ".")
    
    model = SFUniDA(args)
###################################################################################    
    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        save_dir = os.path.dirname(args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        save_dir = os.path.join(this_dir, "checkpoints_glc_fuz", args.dataset, str(args.rule_num), "for_r0.65_target_{}".format(args.t_idx),
                                "source_{}_{}".format(args.source_train_type, args.target_label_type))
#######################################################################################        
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
            
    model.cuda()
    args.save_dir = save_dir     
    logger = set_logger(args, log_name="log_source_training.txt")
    
    params_group = []
    for k, v in model.backbone_layer.named_parameters():
        params_group += [{"params":v, 'lr':args.lr*0.1}]
    for k, v in model.feat_embed_layer.named_parameters():
        params_group += [{"params":v, 'lr':args.lr}]
    for k, v in model.class_layer.named_parameters():
        params_group += [{"params":v, 'lr':args.lr}]
    for k, v in model.rule_layer.named_parameters():
        params_group += [{"params":v, 'lr':args.lr}]
        
    optimizer = torch.optim.SGD(params_group)
    optimizer = op_copy(optimizer)
       
    source_data_list = [open(os.path.join(args.source_domain_dir_list[idx], "image_unida_list.txt"), "r").readlines() for idx in range(len(args.source_domain_dir_list))]
    source_dataset = [SFUniDADataset(args, args.source_domain_dir_list[idx], source_data_list[idx], d_type="source", preload_flg=True) for idx in range(len(args.source_domain_dir_list))]
    source_dataloader = [DataLoader(source_dataset[idx], batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, drop_last=True) for idx in range(len(args.source_domain_dir_list))]
                                   
    ###########################################################
    #source_dataset_te = [SFUniDADataset(args, args.source_domain_dir_list[idx], source_data_list[idx], d_type="source", preload_flg=False) for idx in range(len(args.source_domain_dir_list))]
    #source_dataloader_te = [DataLoader(source_dataset_te[idx], batch_size=args.batch_size, shuffle=False,
    #                               num_workers=args.num_workers, drop_last=False) for idx in range(len(args.source_domain_dir_list))]
                                   
    ###################################################################
    
    
    target_data_list = open(os.path.join(args.target_data_dir, "image_unida_list.txt"), "r").readlines()
    target_dataset = SFUniDADataset(args, args.target_data_dir, target_data_list, d_type="target", preload_flg=False)
    target_dataloader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers, drop_last=False)
    
    
    if args.source_train_type == "smooth":
        criterion = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=0.1, reduction=True)
    elif args.source_train_type == "vanilla":
        criterion = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=0.0, reduction=True)
    else:
        raise ValueError("Unknown source_train_type:", args.source_train_type) 
    
    notation_str =  "\n=================================================\n"
    notation_str += "    START PRE-TRAINING ON MULTI-SOURCE FOR THE TARGET:{} == {}         \n".format(args.t_idx, args.target_label_type)
    notation_str += "================================================="
    
    logger.info(notation_str)
    
    #rule_label_s = [torch.from_numpy(np.load(args.rule_lab_path[idx]).astype('int')).cuda() for idx in range(len(args.source_domain_list))]
    
    #args.iter_idx = epoch_idx * len(dataloader)
    #args.iter_max = args.epochs * len(dataloader)
    #clu_rule_cen(dataloader, model, args)
    
    for epoch_idx in tqdm(range(args.epochs), ncols=60):#epoch_idx  = 1
        local_model = []
        if epoch_idx % 1 == 0:
            model.eval() #source_dataloader_te  args, s_dset_path, rule_lab_file_path, model
            rule_cen_s = [clu_rule_cen_b(source_dataloader[idx], model, args) for idx in range(args.sk)] 
            model.train()
        for idx in range(len(args.source_domain_list)):
            train_loss, source_model = train(rule_cen_s[idx], args, model, source_dataloader[idx], criterion, optimizer, epoch_idx)
            logger.info("Epoch:{}/{} train_loss:{:.3f}".format(epoch_idx, args.epochs, train_loss))
            
            local_model.append(copy.deepcopy(source_model))
            del source_model
        
            if epoch_idx % 1 == 0:
                model.eval()# EVALUATE ON SOURCE
                source_h_score, source_known_acc, source_unknown_acc, src_per_cls_acc = test(args, model, source_dataloader[idx], src_flg=True)
                model.train()
                logger.info("EVALUATE ON SOURCE: H-Score:{:.3f}, KnownAcc:{:.3f}, UnknownAcc:{:.3f}".\
                            format(source_h_score, source_known_acc, source_unknown_acc))
                if args.dataset == "VisDA":
                    logger.info("VISDA PER_CLS_ACC:")
                    logger.info(src_per_cls_acc)
                            
        model.load_state_dict(avg_model(local_model, args)) 
        #local_model = []
          
        checkpoint_file = "latest_source_checkpoint.pth"
        torch.save({
            "epoch":epoch_idx,
            "model_state_dict":model.state_dict()}, os.path.join(save_dir, checkpoint_file))
        
    notation_str =  "\n=================================================\n"
    notation_str += "        EVALUATE ON THE TARGET:{}                  \n".format(args.target_domain)
    notation_str += "================================================="
    logger.info(notation_str)
        
    hscore, knownacc, unknownacc, _ = test(args, model, target_dataloader, src_flg=False)
    logger.info("H-Score:{:.3f}, KnownAcc:{:.3f}, UnknownACC:{:.3f}".format(hscore, knownacc, unknownacc))
    
if __name__ == "__main__":
    args = build_args()
    set_random_seed(args.seed)
    main(args)
