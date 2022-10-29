from dbm import ndbm
from telnetlib import X3PAD
from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from torch.nn.parameter import Parameter
import torch
import math


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    
    def forward(self, x, adj):
        h0 = x
        x = F.relu(self.gc1(x, adj, h0))
        x = F.dropout(x, self.dropout, training = self.training)
        x = self.gc2(x, adj, h0)
        return x

class consists(nn.Module):
    def __init__(self,nhid2,nclass):
        super(consists,self).__init__()

        self.linear1 = nn.Linear(nhid2,nhid2)
        self.linear2 = nn.Linear(nhid2,nhid2)
        self.linear3 =  nn.Linear(nhid2,nhid2)
        self.linear4 =  nn.Linear(nhid2,nhid2)
        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim =1)
    
    def forward(self,Zt,Zf,Zc):
        xt = self.softmax(self.tanh(self.linear1(Zt)))
        at = xt
        xt = torch.mul(xt,Zt)
        xf = self.softmax(self.tanh(self.linear2(Zf)))
        af = xf
        xf = torch.mul(xf,Zf)
        xc1 = self.softmax(self.tanh(self.linear3(Zc)))
        xc2 = self.softmax(self.tanh(self.linear3(Zc)))
        ac1 = xc1
        ac2 = xc2
        xc1 = torch.mul(xc1,Zc)
        xc2 = torch.mul(xc2,Zc)
        Ztc = xt+xc1
        Zfc = xf+xc2
        Z = torch.cat((Ztc,Zfc),-1)
        L = self.softmax(self.linear4(Z))
        return L,at,af,ac1,ac2


class SFGCN(nn.Module):
    def __init__(self, nfeat, nclass, nhid1, nhid2, n, dropout):
        super(SFGCN, self).__init__()

        self.SGCN1 = GCN(nfeat, nhid1, nhid2, dropout)
        self.SGCN2 = GCN(nfeat, nhid1, nhid2, dropout)
        # self.CGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.CGCN = GCN(nhid2, nhid2, nhid2, dropout)

        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(nhid2, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.consistancy = consists(nhid2,nclass)

        self.MLPavg = nn.Sequential
        (
          nn.Linear(2*nhid2,nhid2)
        )

        self.MLP_init = nn.Linear(nfeat,nhid1)
        self.MLP_dim = nn.Linear(nhid1,nhid2)

        self.MLP = nn.Sequential
        (
            nn.Linear(nhid2, nclass),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, sadj, fadj):
        x = self.MLP_init(x)
        emb1 = self.SGCN1(x, sadj) # Special_GCN out1 -- sadj structure graph
        emb2 = self.SGCN2(x, fadj) # Special_GCN out2 -- fadj feature graph
        x = self.MLP_dim(x)
        alpha = 0.85
        emb1c = alpha*emb1 + (1-alpha)*x
        emb2c = alpha*emb2 + (1-alpha)*x
    
        com1 = self.CGCN(emb1c, sadj)  
        com2 = self.CGCN(emb2c, fadj)
  
        Xcat = torch.cat((com1,com2),-1)
        Xcom = self.MLPavg(Xcat) 


        output1,at,af,ac1,ac2= self.consistancy(emb1,emb2,Xcom)
        lst = [at,af,ac1,ac2]
        # return output, emb1, com1, com2, emb2, emb
        return output1, emb1, com1, com2, emb2, emb2,lst