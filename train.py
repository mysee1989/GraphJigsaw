import os
import torch
import argparse
import numpy as np
import random
import time
import torch.nn as nn
from torch.nn import DataParallel
import torch.optim as optim
from data.dataset import Dataset
from torch.utils import data

from model_block_progressive import GJPN
from models.my_resnet import resnet50,resnet101
from models.densenet import densenet169
from models.metrics import ArcMarginProduct,CosineMarginProduct
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from evaluation_icartoon_face import evaluation

import pdb


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--train_root',type=str,default='/data/laolingjie/database/iQiYi/personai_icartoonface_rectrain/icartoonface_rectrain/')
    parse.add_argument('--train_list',type=str,default='/data/laolingjie/database/iQiYi/personai_icartoonface_rectrain/train_lbl.list')
    # parse.add_argument('--train_root',type=str,default='/data/laolingjie/database/Danbooru/danbooru2018/')
    # parse.add_argument('--train_list',type=str,default='/data/laolingjie/database/Danbooru/train.txt')
    parse.add_argument('--test_root',type=str,default='/data/laolingjie/database/iQiYi/personai_icartoonface_rectest/icartoonface_rectest')
    parse.add_argument('--test_list',type=str,default='/data/laolingjie/database/iQiYi/personai_icartoonface_rectest/icartoonface_rectest_22500.list')
    parse.add_argument('--checkpoint',type=str,default='checkpoints/',help='the path that save the model')
    parse.add_argument('--rank_dir',type=str,default='rank-n/',help='the path that save the rank reslut')
    parse.add_argument('--resume',type=bool,default=False)
    parse.add_argument('--model_path',type=str,default='checkpoints/xxx.pth',help='pretrain model dir')
    parse.add_argument('--train_log',type=str,default='log/xxx.log',help='the path that save the train log')
    parse.add_argument('--save_name',type=str,default='xxx')
    parse.add_argument('--start_epoch',type=int,default=0)
    parse.add_argument('--nb_epoch',type=int,default=61)
    parse.add_argument('--batch_size',type=int,default=256)
    parse.add_argument('--num_workers',type=int,default=16)
    parse.add_argument('--block_num',type=int,default=3,help='the block size of recontrust feature node, e.g., 3*3')
    parse.add_argument('--classes_num',type=int,default=5013,help="danbooru 5127  iQiYi 5013") 
    parse.add_argument('--after_epoch',type=int,default=7)
    parse.add_argument('--use_arcface',type=bool,default=False)
    parse.add_argument('--metric_name',type=str,default='arcface_m3_after15')
    parse.add_argument('--margin',type=float,default=0.35)
    parse.add_argument('--alpha',type=float,default=1,help='the loss weight of recontrust loss')

    return parse.parse_args()

def save_model(model, save_path, name, iter_cnt):                                                                                   
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name

def save_rank(rank,epoch,args):
    with open(os.path.join(args.rank_dir,args.save_name+'_'+str(epoch)+'_'+str(rank[0])+'_'+str(rank[4])+'_'+str(rank[5])),'w') as f:
        for i,r in enumerate(rank):
            f.write("rank{}: {}".format(str(i+1),str(r))+'\n')


def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1
    return float(lr / 2 * cos_out)

def val(model,val_loader,CELoss,args,epoch):
    val_loss = 0
    ce_loss = 0
    reconstruct_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs,targets,path) in enumerate(val_loader):
        idx = batch_idx
        inputs,targets = inputs.cuda(),targets.long().cuda()
        output,loss1 = model(inputs)
        loss2 = CELoss(output,targets)
        
        loss = args.alpha*loss1 + loss2
        
        ce_loss += loss2.item()
        reconstruct_loss += loss1.item()
        val_loss +=loss.item()

        #prediction
        _,predicted = torch.max(output.data,1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    val_acc = 100. *float(correct) /total
    val_loss = val_loss /(idx+1)
    print(
        'Iteration %d | val_acc = %.3f | val_loss = %.3f | Loss_ce: %.3f | Loss_reconstruct: %.3f |\n' % (
            epoch,val_acc,val_loss, ce_loss /(idx+1), args.alpha*reconstruct_loss / (idx+1)))
    print('--'*40)
    with open(args.train_log,'a') as file:
        file.write(
            '\nIteration %d | val_acc = %.3f | val_loss = %.3f | Loss_ce: %.3f | Loss_reconstruct: %.3f |\n' % (
                epoch,val_acc,val_loss, ce_loss /(idx+1), args.alpha*reconstruct_loss / (idx+1)))
        file.write('--'*40+'\n')

def val_iqiyi(model,val_loader,args,epoch):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for batch_i,(imgs,label,pathes) in enumerate(val_loader):
            imgs = imgs.cuda()
            label = label.cuda()
            feature = model(imgs,0,0,0,phase='test')
            feature = F.normalize(feature)
            label = label.cpu().detach().numpy()
            feature = feature.cpu().detach().numpy()

            features.extend(feature)
            labels.extend(label)
    features = np.array(features)
    rank = evaluation(features,labels)
    
    print(
        'Iteration %d | Rank@1 = %.3f | Rank@5 = %.3f | Rank@10: %.3f |\n' % (
            epoch,rank[0],rank[4],rank[9]))
    print('--'*40)
    with open(args.train_log,'a') as file:
        file.write(
            '\nIteration %d | Rank@1 = %.3f | Rank@5 = %.3f | Rank@10: %.3f |\n' % (
            epoch,rank[0],rank[4],rank[9]))
        file.write('--'*40+'\n')
    
    return rank

def L2_Loss(recon_feature,feature):
    return torch.sqrt(torch.sum((feature-recon_feature)**2))/feature.size(0)

def train():
    args = get_args()

    print('===>loading data')
    train_dataset = Dataset(args.train_root,args.train_list,phase='train',input_shape=(3,224,224))
    train_loader = data.DataLoader(train_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers)
    
    print('Train set size:', train_dataset.__len__())


    val_dataset = Dataset(args.test_root,args.test_list,phase='test',input_shape=(3,224,224))
    val_loader = data.DataLoader(val_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers)
    
    print('Val set size:', val_dataset.__len__())

    print('===>loading Model')
    backbone = resnet50()
    backbone.load_state_dict(torch.load('../checkpoints/resnet50-19c8e357.pth'))
    model = GJPN(backbone,512,args.classes_num,args.block_num)

    if args.resume:
        state_dict = torch.load(args.model_path)
        model.load_state_dict(state_dict)
    model.cuda()
    model = torch.nn.DataParallel(model)
    print(model)

    CELoss = nn.CrossEntropyLoss()
    MSELoss = nn.MSELoss()
    if args.use_arcface:
        metric =  ArcMarginProduct(512, args.classes_num, s=64, m=args.margin, easy_margin=False)
        # metric = CosineMarginProduct(512,args.classes_num,s=30,m=args.margin)
        metric.cuda()
    optimizer = optim.SGD([
        {'params': model.module.feature.parameters(),'lr':0.07},
        {'params': model.module.classifier.parameters(),'lr':0.07},
        {'params': model.module.gc1.parameters(),'lr':0.007},
        {'params': model.module.gc2.parameters(),'lr':0.007},
        {'params': model.module.gc1_1.parameters(),'lr':0.007},
        {'params': model.module.gc2_1.parameters(),'lr':0.007},
        {'params': model.module.gc1_2.parameters(),'lr':0.007},
        {'params': model.module.gc2_2.parameters(),'lr':0.007},
        {'params': model.module.gc1_3.parameters(),'lr':0.007},
        {'params': model.module.gc2_3.parameters(),'lr':0.007},
    ],
        momentum=0.9,weight_decay=5e-4)
    lr = [0.07,0.07,0.007,0.007,0.007,0.007,0.007,0.007,0.007,0.007]
    

    time_str = time.asctime(time.localtime(time.time()))
    print('\n{} Starting train!!!'.format(time_str))
    print(str(args))
    with open(args.train_log,'a') as file:
        file.write(str(args))
        file.write('\n{} Starting train!!!\n'.format(time_str))
    max_rank1 = 0.0
    for epoch in range(args.start_epoch+1,args.nb_epoch+1):
        print('\nEpoch: %d' % epoch)
        model.train()
        # metric.train()
        train_loss = 0
        ce_loss = 0
        ce_loss_jigsaw = 0
        reconstruct_loss = 0
        correct = 0
        total = 0
        idx = 0
        print('lr:',optimizer.param_groups[0]['lr'])
        for batch_idx, (inputs,targets) in enumerate(train_loader):
            idx = batch_idx
            inputs,targets = inputs.cuda(),targets.long().cuda()
            
            #update learning rate
            for nlr in range(len(optimizer.param_groups)):
                    optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, args.nb_epoch, lr[nlr])

            optimizer.zero_grad()

            if epoch<=args.after_epoch:
                output = model(inputs,epoch,idx,args.after_epoch)
                loss2 = CELoss(output,targets)
                loss = loss2
            else:
                output,recon_feature,feature = model(inputs,epoch,idx,args.after_epoch)
                loss1 = L2_Loss(recon_feature,feature)
                loss2 = CELoss(output,targets)
                loss = args.alpha*loss1 + loss2

            loss.backward()
            optimizer.step()

            ce_loss += loss2.item()
            if epoch>args.after_epoch:
                reconstruct_loss += loss1.item()
            train_loss +=loss.item()

            #prediction
            _,predicted = torch.max(output.data,1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            if batch_idx % 200 ==0:
                print(
                    'Iteration %d Step: %d | Loss_ce: %.3f | Loss_reconstruct: %.3f | Loss_total: %.3f | Acc: %.3f%% (%d/%d)' %(
                        epoch,batch_idx,ce_loss / (batch_idx + 1),args.alpha*reconstruct_loss / (batch_idx + 1),train_loss /(batch_idx+1),
                        100. *float(correct)/total,correct,total))
                with open(args.train_log,'a') as file:
                    file.write(
                    'Iteration %d Step: %d | Loss_ce: %.3f | Loss_reconstruct: %.3f | Loss_total: %.3f | Acc: %.3f%% (%d/%d)\n' %(
                        epoch,batch_idx,ce_loss / (batch_idx + 1),args.alpha*reconstruct_loss / (batch_idx + 1),train_loss /(batch_idx+1),
                        100. *float(correct)/total,correct,total))
        # scheduler.step()
        train_acc = 100. *float(correct) /total
        train_loss = train_loss /(idx+1)
        time_str = time.asctime(time.localtime(time.time()))
        print(
            'Iteration %d | train_acc = %.3f | train_loss = %.3f | Loss_ce: %.3f | Loss_reconstruct: %.3f |\n' % (
                epoch,train_acc,train_loss, ce_loss /(idx+1),args.alpha*reconstruct_loss / (idx+1)))
        print('{} Iteration {} finish!!!\n'.format(time_str,epoch))
        print('--'*40)
        with open(args.train_log,'a') as file:
            file.write(
                '\nIteration %d | train_acc = %.3f | train_loss = %.3f | Loss_ce: %.3f | Loss_reconstruct: %.3f |\n' % (
                    epoch,train_acc,train_loss, ce_loss /(idx+1), args.alpha*reconstruct_loss / (idx+1)))
            file.write('\n{} Iteration {} finish!!!\n'.format(time_str,epoch))
            file.write('--'*40+'\n')
        # val(model,val_loader,CELoss,args,epoch)
        
        rank = val_iqiyi(model,val_loader,args,epoch)
        if max_rank1 < rank[0]:
            max_rank1 = rank[0]
            save_model(model.module,args.checkpoint,args.save_name,epoch)
            save_rank(rank,epoch,args)


if __name__=='__main__':
    train()
