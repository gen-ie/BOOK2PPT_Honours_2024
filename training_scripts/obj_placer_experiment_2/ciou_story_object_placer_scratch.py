import torch as T
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
from datetime import datetime

import pickle, json, os
from tqdm import tqdm
from glob import glob

from dataset import *
from model import *
from mlp_layout import *

if __name__=='__main__':
    def pad_tensor(vec, pad, dim):
        pad_size = list(vec.shape)
        pad_size[dim] = pad - vec.size(dim)
        return T.cat([vec, T.zeros(*pad_size, device=device)], dim=dim)

    dat = {}
    for conf in json.load(open('../../data/v1.0/book_json.json', 'r')):
        pkl = pickle.load(open(f'../../books/{conf}.pkl', 'rb'))
        dat[conf] = {}
        for item in pkl:
            dat[conf][item['idd']] = item
    dl_tr = DLoader(dat, typ='train', domain='stories')
    dl_vl = DLoader(dat, typ='val', domain='stories')

    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    loaded_model = T.load('../../models/storybook_model_dropout_best.pt', map_location=device, weights_only=True)
  
    # Enable Data Parallelism
    model = Model().to(device)
    model.load_state_dict(loaded_model)
    model = nn.DataParallel(model)  # Wrap the model with DataParallel

    # compute the bounding boxes for
    train_outputs = [model(item) for item in dl_tr]
    val_outputs = [model(item) for item in dl_vl]

    # Create a parallel version of the loc_model
    loc_model = MLPlayout().to(device)
    loc_model = nn.DataParallel(loc_model)
    loss_func = CIoULoss().to(device)

    optzr = T.optim.Adam(loc_model.parameters(), lr=0.0003)

    # added to regulate performance of validation rate
    scheduler = ReduceLROnPlateau(optzr, mode='min', factor=0.1, patience=5)

    # initialize variables for early stopping
    best_val_loss = np.inf
    patience = 10  # number of epochs to wait before stopping

    PATH = 'ciou_story_locmodel_scratch'
    os.system('mkdir -p %s' % (PATH))

    log = {}
    json.dump(log, open('%s/log.json'%(PATH), 'w'))

    with tqdm(range(75), ascii=True) as TQ:
        for e in TQ:
            log['epoch %d'%(e)] = {'train': [], 'val': []}
            
            loc_model.train()
            ls_tr = 0
            for item, out in zip(dl_tr, train_outputs):
                slide_size = [960, 720] 
                for section, gd_sec in zip(out['pd_obj'], item['out_bbox']):
                    for slide, gd in zip(section, gd_sec):
                        if(len(slide) == 1):
                            continue
                        slide, gd = slide[:-1], gd[:-1]
                        
                        # eliminate other Nones from groundtruth
                        none_list = [i for i in range(len(gd)) if gd[i] is None]
                        slide = [slide[i] for i in range(len(slide)) if i not in none_list]
                        gd = [gd[i] for i in range(len(gd)) if i not in none_list]

                        pd = T.cat(slide, dim=0)
                        gd_bbox = T.tensor(gd, device=device)
                        pd_padded = [pad_tensor(i, 1024, 0) for i in pd]
                        pd_bbox = [loc_model(p) for p in pd_padded]
                        pd_bbox = T.stack(pd_bbox)
                        
                        # Calculate CIOU Loss
                        ciou_loss = loss_func(pd_bbox, gd_bbox, slide_size)

                        optzr.zero_grad()
                        ciou_loss.backward(retain_graph=True)
                        optzr.step()

                        ls_tr += ciou_loss.data.cpu().numpy()
                        ls_bbox = ciou_loss.data.cpu().numpy() if not ciou_loss == 0 else 0
                        
                        TQ.set_postfix(train_ls_bbox='%.3f'%(ls_bbox))
                        log['epoch %d'%(e)]['train'].append(float('%.6f'%(ls_bbox)))
                ls_tr /= len(dl_tr)
            
            loc_model.eval()
            ls_vl = 0
            with T.no_grad():
                for item, out in zip(dl_vl, val_outputs):
                    slide_size = [960, 720]
                    for section, gd_sec in zip(out['pd_obj'], item['out_bbox']):
                        for slide, gd in zip(section, gd_sec):
                            if(len(slide) == 1):
                                continue
                            slide, gd = slide[:-1], gd[:-1]

                            # eliminate other Nones from groundtruth
                            none_list = [i for i in range(len(gd)) if gd[i] is None]
                            slide = [slide[i] for i in range(len(slide)) if i not in none_list]
                            gd = [gd[i] for i in range(len(gd)) if i not in none_list]

                            pd = T.cat(slide, dim=0)
                            gd_bbox = T.tensor(gd, device=device)
                            pd_padded = [pad_tensor(i, 1024, 0) for i in pd]
                            pd_bbox = [loc_model(p) for p in pd_padded]
                            pd_bbox = T.stack(pd_bbox)
                            
                            # Calculate CIOU Loss
                            ciou_loss = loss_func(pd_bbox, gd_bbox, slide_size)
            
                            ls_vl += ciou_loss.data.cpu().numpy()
                            ls_bbox = ciou_loss.data.cpu().numpy() if not ciou_loss==0 else 0
                            
                            TQ.set_postfix(val_ls_bbox='%.3f'%(ls_bbox))
                            log['epoch %d'%(e)]['val'].append(float('%.6f'%(ls_bbox)))
                    ls_vl /= len(dl_vl)

            print('Ep %d: ls_tr=%.4f, ls_vl=%.4f' % (e+1, ls_tr, ls_vl))
            json.dump(log, open('%s/log.json'%(PATH), 'w'))

            # Step the learning rate scheduler
            scheduler.step(ls_vl)

            # Early Stopping Logic
            if ls_vl < best_val_loss:
                best_val_loss = ls_vl
                patience_counter = 0
                T.save(loc_model.state_dict(), '%s/ciou_storyloc_best_scratch.pt'%(PATH))  # Save the best model
            else:
                patience_counter += 1
                # if patience_counter >= patience:
                #     print("Early stopping triggered. Training stopped.")
                #     break
    T.save(loc_model.state_dict(), '%s/ciou_storyloc_final_scratch.pt'%(PATH))
