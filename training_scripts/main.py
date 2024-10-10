# ORIGINAL CODE FROM DOC2PPT REPOSITORY - Training Code

from datetime import datetime

import pickle, json, os
from tqdm import tqdm
from glob import glob

import numpy as np

import torch as T

from dataset import *
from model import *

dat = {}
for conf in json.load(open('./data/v1.0/train_val_test.json', 'r')):
    pkl = pickle.load(open('./data/v1.0/%s.pkl'%(conf), 'rb'))
    dat[conf] = {}
    for item in pkl:
        dat[conf][item['idd']] = item
dl_tr = DLoader(dat, typ='train')
dl_vl = DLoader(dat, typ='val')

device = T.device("cuda" if T.cuda.is_available() else "cpu")
model = Model().to(device)
loss_func = Loss().to(device)
optzr = T.optim.Adam(model.parameters(), lr=0.0003)

PATH = 'log_%s'%(datetime.now().strftime('%Y%m%d_%H%M%S'))
os.system('mkdir -p %s' % (PATH))

log = {}
json.dump(log, open('%s/log.json'%(PATH), 'w'))

with tqdm(range(100), ascii=True) as TQ:
    for e in TQ:
        log['epoch %d'%(e)] = {'train': {'ls_tok': [], 'ls_obj': []}, 'val': {'ls_tok': [], 'ls_obj': []}}
        
        model.train()
        ls_tr = 0
        for item in dl_tr:
            out = model(item)
            ls_tok, ls_obj = loss_func(out, item)
            
            ls = ls_tok+ls_obj
            optzr.zero_grad()
            ls.backward()
            optzr.step()
            
            ls_tr += ls.data.cpu().numpy()
            ls_tok = ls_tok.data.cpu().numpy() if not ls_tok==0 else 0
            ls_obj = ls_obj.data.cpu().numpy() if not ls_obj==0 else 0
            
            TQ.set_postfix(ls_tok='%.3f'%(ls_tok), ls_obj='%.3f'%(ls_obj))
            log['epoch %d'%(e)]['train']['ls_tok'].append(float('%.6f'%(ls_tok)))
            log['epoch %d'%(e)]['train']['ls_obj'].append(float('%.6f'%(ls_obj)))
        ls_tr /= len(dl_tr)
        
        model.eval()
        ls_vl = 0
        for item in dl_vl:
            out = model(item)
            ls_tok, ls_obj = loss_func(out, item)
            
            ls = ls_tok+ls_obj
            
            ls_vl += ls.data.cpu().numpy()
            ls_tok = ls_tok.data.cpu().numpy() if not ls_tok==0 else 0
            ls_obj = ls_obj.data.cpu().numpy() if not ls_obj==0 else 0
            
            TQ.set_postfix(ls_tok='%.3f'%(ls_tok), ls_obj='%.3f'%(ls_obj))
            log['epoch %d'%(e)]['val']['ls_tok'].append(float('%.6f'%(ls_tok)))
            log['epoch %d'%(e)]['val']['ls_obj'].append(float('%.6f'%(ls_obj)))
        ls_vl /= len(dl_vl)
        
        print('Ep %d: ls_tr=%.4f, ls_vl=%.4f' % (e+1, ls_tr, ls_vl))
        T.save(model.state_dict(), '%s/model_%d.pt'%(PATH, e+1))
        json.dump(log, open('%s/log.json'%(PATH), 'w'))
        