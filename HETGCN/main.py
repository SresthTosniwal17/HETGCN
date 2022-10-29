from __future__ import division
from __future__ import print_function
from json import load
from webbrowser import get
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils import *
from models import SFGCN
import numpy
from sklearn.metrics import f1_score
import os
import torch.nn as nn
import argparse
from config import Config
import random 
from scipy import sparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
#AMGCN
###################

if __name__ == "__main__":
    alphat = []
    alphaf = []
    alphac = []
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parse = argparse.ArgumentParser()
    parse.add_argument("-d", "--dataset", help="dataset", type=str, required=True)
    parse.add_argument("-l", "--labelrate", help="labeled data for train per class", type = int, required = True)
    # parse.add_argument("-w", "--weight", help="weight", type = int, required = True)
    # parse.add_argument("-b", "--beta", help="beta", type = int, required = True)
    # parse.add_argument("-t", "--theta", help="theta", type = int, required = False)
    # parse.add_argument("-l1", "--nhid1", help="beta", type = int, required = True)
    # parse.add_argument("-l2", "--nhid2", help="theta", type = int, required = False)
    # parse.add_argument("-hp", "--heterophily", help="heterophily", type = int, required = True)
    
    args = parse.parse_args()
    config_file = "./config/" + str(args.labelrate) + str(args.dataset) + ".ini"
    config = Config(config_file)

    cuda = not config.no_cuda and torch.cuda.is_available()

    use_seed = not config.no_seed
    if use_seed:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if cuda:
            torch.cuda.manual_seed(config.seed)
    # weight = 2
    # heterophily = 85
    # sadj = load_topology_graph(args.labelrate,config,args.heterophily,int(config.n),int(config.class_num))
    # fadj = load_feature_graph(args.labelrate,config)
    sadj, fadj = load_graph(args.labelrate, config)
    features, labels, idx_train, idx_test = load_data(config)
    
    
    model = SFGCN(nfeat = config.fdim,
              nhid1 = config.nhid1,
            #   nhid1 = args.nhid1,
              nhid2 = config.nhid2,
            #   nhid2 = args.nhid2,
              nclass = config.class_num,
              n = config.n,
              dropout = config.dropout)
    if cuda:
        model.cuda()
        features = features.cuda()
        sadj = sadj.cuda()
        fadj = fadj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_test = idx_test.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    def att(m):
        mat = m.detach().cpu().numpy()
        norm = np.linalg.norm(mat, ord='fro')
        # norm = np.expand_dims(norm,axis=1)
        norm = norm/mat.shape[0]
        return norm 


    # print(features)
    def train(model, epochs):
        model.train()
        optimizer.zero_grad()
        # output1,output2, emb1, com1, com2, emb2, emb= model(features, sadj, fadj)
        output, emb1, com1, com2, emb2, emb,lst= model(features, sadj, fadj)
        
        loss_class =  F.nll_loss(output[idx_train], labels[idx_train])
        # loss_dep = (loss_dependence(emb1, com1, config.n) + loss_dependence(emb2, com2, config.n))/2
        loss_dep = cos_sim(emb1, com1) + cos_sim(emb2, com2)
 
        loss_com = common_loss(com1,com2)
        loss = config.weight*loss_class + config.beta * loss_dep + config.theta * loss_com
        # loss = args.weight*loss_class + args.beta * loss_dep + args.theta * loss_com

        lst.append(loss)
        acc = accuracy(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()
        acc_test, macro_f1, emb_test = main_test(model)
        print('e:{}'.format(epochs),
              'atr: {:.4f}'.format(acc.item()),
              'ate: {:.4f}'.format(acc_test.item()),
              'f1te:{:.4f}'.format(macro_f1.item()),)
        return loss.item(), acc_test.item(), macro_f1.item(), emb_test

    def main_test(model):
        model.eval()
        output1 , emb1, com1, com2, emb2, emb,lst = model(features, sadj, fadj)
        # output, emb1, com1, com2, emb2, emb = model(features, sadj, fadj
        lst_norm = []
        for i in lst:
            norm = att(i)
            lst_norm.append(norm)
        acc_test = accuracy(output1[idx_test], labels[idx_test])
        alphat.append(lst_norm[0])
        alphaf.append(lst_norm[1])
        alphac.append(lst_norm[2])
        label_max = []
        for idx in idx_test:
            label_max.append(torch.argmax(output1[idx]).item())
        labelcpu = labels[idx_test].data.cpu()
        macro_f1 = f1_score(labelcpu, label_max, average='macro')
        return acc_test, macro_f1, emb
    acc_dmax = 0
    acc_smax = 0
    acc_max = 0
    f1_max = 0
    epoch_smax = 0
    epoch_dmax = 0
    for epoch in range(150):
        loss, acc_test, macro_f1, emb = train(model, epoch)
        if acc_test >= acc_max:
            acc_max = acc_test
            f1_max = macro_f1
            epoch_max = epoch
    print('epoch:{}'.format(epoch_max),'acc_max: {:.4f}'.format(acc_max),'f1_max: {:.4f}'.format(f1_max))