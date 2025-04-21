import os
import torch
import numpy as np
import torch.nn as nn
import sys
import torch.utils.data as data
import torch.optim as optim

class grnnsdr(object):

    def __init__(self, hidden1_units, hidden2_units, epoch_num, lr, 
                 x_tr, y_tr, x_va, y_va, x_te, y_te, beta, weight_decay):

        self.hidden1_units = hidden1_units
        self.hidden2_units = hidden2_units
        self.epoch_num = epoch_num
        self.lr = lr
        self.x_tr = x_tr
        self.y_tr = y_tr
        self.x_va = x_va
        self.y_va = y_va
        self.x_te = x_te
        self.y_te = y_te
        self.beta = beta
        self.weight_decay = weight_decay

    class MyDataSet(data.Dataset):
        def __init__(self, x=None, y=None):
            self.X = x
            self.Y = y

        def __getitem__(self, index):
            return self.X[index], self.Y[index]

        def getall(self):
            return self.X, self.Y

        def __len__(self):
            return len(self.X)

    def train(self):

        tr_best = torch.FloatTensor([1000])
        va_best = torch.FloatTensor([1000])
        va_best_te = torch.FloatTensor([1000])
        parameters_best = []
        x_va_data = self.x_va
        x_te_data = self.x_te

        net = nn.Sequential(
            nn.Linear(self.x_tr.shape[1], self.hidden1_units, bias=False),  
            nn.Linear(self.hidden1_units, self.hidden2_units),
            nn.Tanh(),
            nn.Linear(self.hidden2_units, 1),
        )
        net = net.to('cuda')
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        cri_loss = nn.MSELoss()
        cri_loss = cri_loss.to("cuda")
        x_va_data = x_va_data.to('cuda')
        x_te_data = x_te_data.to('cuda')
        self.y_va = self.y_va.to('cuda')
        self.y_te = self.y_te.to('cuda')
        self.y_tr = self.y_tr.to('cuda')

        for epoch in range(self.epoch_num):
            train_set = self.MyDataSet(self.x_tr, self.y_tr)
            train_loader = data.DataLoader(train_set, batch_size=self.x_tr.shape[0], shuffle=True)

            tr_mse = torch.FloatTensor([0])
            for i, batch in enumerate(train_loader, 0):
                x_batch, y_batch = batch
                x_batch = x_batch.to('cuda')
                y_batch = y_batch.to('cuda')
                pre = net(x_batch)
                loss = cri_loss(pre, y_batch)
                tr_mse += loss.data.cpu()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            tr_mse = tr_mse / (i+1)
            
            with torch.no_grad():
                pre_va = net(x_va_data)
                pre_te = net(x_te_data)

                pre_loss_va = cri_loss(pre_va, self.y_va).cpu()
                pre_loss_te = cri_loss(pre_te, self.y_te).cpu()

                if pre_loss_va.data < va_best:
                    va_best = pre_loss_va.data
                    va_best_te = pre_loss_te.data
                    tr_best = tr_mse.data
                    parameters_best = list(net.parameters())
                va_best = min(pre_loss_va.data, va_best)
                sys.stdout.flush()

        net.to('cpu')
        p1 = parameters_best[0].detach().numpy()
        p1 = p1.T
        self.beta = self.beta.T

        self.beta, R = np.linalg.qr(self.beta, mode='full')
        p1, R = np.linalg.qr(p1, mode='full')

        tmp = np.matmul(p1.T ,self.beta)
        tmp = np.matmul(tmp ,self.beta.T)
        tmp = np.matmul(tmp, p1)

        q = np.linalg.det(tmp)
        q = np.sqrt(q)

        return tr_best[0], va_best, va_best_te, parameters_best,q

def nn(trained, hidden1_units, hidden2_units, epoch_num, lr, 
                x_tr, y_tr, x_va, y_va, x_te, y_te, beta, weight_decay, k):
    min_mse_tr = np.array([10000000.])
    min_mse_va = np.array([10000000.])
    min_mse_te = np.array([10000000.])
    w = np.ones((hidden1_units, x_tr.shape[1]))
    q = 100
    for i in range(k):
 
        f = grnnclass(hidden1_units=hidden1_units, hidden2_units=hidden2_units, 
                   epoch_num=epoch_num, lr=lr, 
                   x_tr=x_tr, y_tr=y_tr, x_va=x_va, y_va=y_va, x_te=x_te, y_te=y_te, beta=beta,
                   weight_decay=weight_decay)

        
        tr_best, va_best, te_best, net_parameter, net_q = f.train()
       
        if min_mse_va >= va_best.data.numpy():
            min_mse_tr = tr_best.data.numpy()
            min_mse_va = va_best.data.numpy()
            min_mse_te = te_best.data.numpy()
            w = net_parameter[0]
            q = net_q
    trained[str(hidden1_units)] = [min_mse_tr, min_mse_va, min_mse_te, w, q]
    return min_mse_tr,min_mse_va, min_mse_te, w, q


def grnn(hidden2_units, epoch_num, lr, 
          x_tr, y_tr, x_va, y_va, x_te, y_te, beta, weight_decay, k, lam, alp):
    trained = {}

    start = 1
    p = x_tr.shape[1]
    tr_len = x_tr.shape[0]
   
    lam = lam
    alp = alp
    pen =  tr_len**(-alp)

    m = start
    n = p

    k1 = int(m + 0.382 * (n - m))
    k2 = int(m + 0.618 * (n - m))

    tmp,p_mse, ___, _, __ = nn(trained, p, hidden2_units, epoch_num, lr, 
                                           x_tr, y_tr, x_va, y_va, x_te, y_te, beta, weight_decay, k)
    tmp,k1_mse, ___, _, __ = nn(trained, int(k1), hidden2_units, epoch_num, lr, 
                                           x_tr, y_tr, x_va, y_va, x_te, y_te, beta, weight_decay, k)
    tmp,k2_mse, ___, _, __ = nn(trained, int(k2), hidden2_units, epoch_num, lr, 
                                           x_tr, y_tr, x_va, y_va, x_te, y_te, beta, weight_decay, k)
    n_mse = p_mse

    while (n-m>=4):
        if  (k1_mse - k2_mse <= lam*(k2-k1)*pen and k2_mse - n_mse <= lam*(n-k2)*pen):

            n = k2
            n_mse = k2_mse
            k2 = k1
            k2_mse = k1_mse
            k1 = int(m + 0.382 * (n - m))
            tmp, k1_mse, ___, _, __= nn(trained, int(k1), hidden2_units, epoch_num, lr, 
                                                      x_tr, y_tr, x_va, y_va, x_te, y_te, beta, weight_decay, k)
        else:
            m = k1
            k1 = k2
            k1_mse = k2_mse
            k2 = int(m + 0.618 * (n - m))
            tmp, k2_mse, _, __, ___= nn(trained, int(k2), hidden2_units, epoch_num, lr, 
                                                      x_tr, y_tr, x_va, y_va, x_te, y_te, beta, weight_decay, k)

    l=n
    if (str(int(l)) in trained) == True:
        l_mse_tr = trained[str(int(l))][0]
        l_mse = trained[str(int(l))][1]
        l_mse_te = trained[str(int(l))][2]
        l_w = trained[str(int(l))][3]
        l_q = trained[str(int(l))][4]
        
    else:
        l_mse_tr, l_mse, l_mse_te, l_w, l_q= nn(trained, int(l), hidden2_units, epoch_num, lr, 
                                                             x_tr, y_tr, x_va, y_va, x_te, y_te, beta, weight_decay, k) 

    d=l-1
    if (str(int(d)) in trained) == True:
        d_mse_tr = trained[str(int(d))][0]
        d_mse = trained[str(int(d))][1]
        d_mse_te = trained[str(int(d))][2]
        d_w = trained[str(int(d))][3]
        d_q = trained[str(int(d))][4]
    else:
        d_mse_tr, d_mse, d_mse_te, d_w, d_q = algorithm1(trained, int(d), hidden2_units, epoch_num, lr, 
                                                            x_tr, y_tr, x_va, y_va, x_te, y_te, beta, weight_decay, k)

    while  ( int(l) >= m and d_mse-l_mse <= lam*pen*1 ) :
        l=l-1
        d=l-1
        if (str(int(l)) in trained) == True:
            l_mse_tr = trained[str(int(l))][0]
            l_mse = trained[str(int(l))][1]
            l_mse_te = trained[str(int(l))][2]
            l_w = trained[str(int(l))][3]
            l_q = trained[str(int(l))][4]
        else:
            l_mse_tr, l_mse, l_mse_te, l_w = nn(trained, int(l), hidden2_units, epoch_num, lr, 
                                                                x_tr, y_tr, x_va, y_va, x_te, y_te, beta, weight_decay, k)

        if d==0 :
            break

        if (str(int(d)) in trained) == True:
            d_mse_tr = trained[str(int(d))][0]
            d_mse = trained[str(int(d))][1]
            d_mse_te = trained[str(int(d))][2]
            d_w = trained[str(int(d))][3]
            d_q = trained[str(int(d))][4]
        else:
            d_mse_tr, d_mse, d_mse_te, d_w, d_q = nn(trained, int(d),hidden2_units, epoch_num, lr, 
                                                                x_tr, y_tr, x_va, y_va, x_te, y_te, beta, weight_decay, k)

    best_d = l
    best_d_mse_tr = l_mse_tr
    best_d_mse = l_mse
    best_d_mse_te = l_mse_te
    best_d_w = l_w
    best_d_q = l_q








