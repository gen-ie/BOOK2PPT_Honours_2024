# ORIGINAL CODE FROM DOC2PPT REPOSITORY - Progress Tracker
# Only comments have been added

import pickle, json, os
from tqdm import tqdm
from glob import glob

import numpy as np

import torch as T

from dataset import *

device = T.device("cuda" if T.cuda.is_available() else "cpu")

class Model(T.nn.Module):
    def __init__(self):
        super().__init__()
        # progress tracker for section
        self.rnn_sec = T.nn.GRU(2048, 256,
                                batch_first=True)
        # progress tracker for slide?
        self.rnn_page = T.nn.GRU(16, 256,
                                 batch_first=True)
        # progress tracker for object
        self.rnn_obj = T.nn.GRU(16, 256,
                                batch_first=True)
        # for contextualising sentence embeddings extracted from document
        self.rnn_text = T.nn.GRU(1024, 1024,
                                 batch_first=True, bidirectional=True)
        # figure and contextualised sentence embeddings are projected to two-layer MLP
        self.fc_fig = T.nn.Linear(1024+2048, 2048)
        # attention map for slides
        self.att_page = T.nn.Linear(256, 2048,
                                    bias=False)
        # attention map for objects
        self.att_obj = T.nn.Linear(256, 2048,
                                    bias=False)
        # used for binary decision making for slide and object
        self.fc_tok = T.nn.Sequential(*[T.nn.Linear(256+2048, 2)])

    def forward(self, item, teacher=True):
        ret = {}
        ret['pd_tok_page'], ret['pd_tok_obj'], ret['pd_obj'] = [], [], []
        ret['out_tok_page'], ret['out_tok_obj'] = [], []

        # PREPARE
        # get extracted figures as indicated by inp_emb_pix = location of of figures in document (coordinated by pixels) and put it in two-layer MLP
        if item['inp_emb_pix'] is not None:
            inp_fig = T.cat([T.from_numpy(item['inp_emb_pix']), T.from_numpy(item['inp_emb_cap'])], dim=1).unsqueeze(0).to(device)
            inp_fig = self.fc_fig(inp_fig)
        else:
            inp_fig = None

        inp, inp_sec = [], []

        # get extracted sentence embeddings as indicated by inp_emb_text
        for emb in item['inp_emb_text']:
            if emb is not None:
                emb = T.from_numpy(emb).unsqueeze(0).to(device)

                # get context from each sentence embedding
                emb, _ = self.rnn_text(emb)
                # adds forward and backwards output of bidirectional-GRU
                inp_sec.append(T.cat([emb[:, -1, :1024], emb[:, 0, 1024:]], dim=1).unsqueeze(1))
                # if figure embeddings exist for this section, concatenate it to contextualised sentence embeddings
                if inp_fig is not None:
                    o = T.cat([emb, inp_fig], dim=1)
                else:
                    o = emb
            else:
                # if there are no sentence embeddings present, just add zeros to inp_sec
                inp_sec.append(T.zeros((1, 1, 2048)).float().to(device))
                if inp_fig is not None:
                    o = inp_fig
                else:
                    o = None
            # add the section (sentences + figure embeddings) onto the list of inputs
            inp.append(o)

        # DECODING 
        # initialise section progress tracker 
        h_sec = None
        for sec_i in range(len(inp)):
            # PT section GRU takes in a sections and the section PT 
            hid_sec, h_sec = self.rnn_sec(inp_sec[sec_i], h_sec)

            # initialises arrays of predicted page tokens, object tokens, and objects?
            pd_tok_page, pd_tok_obj, pd_obj = [], [], []
            # initialises arrays of output page tokens and object tokens
            out_tok_page, out_tok_obj = [], []
            # initialises slide progress pointer (h_page) = hid_sec; page_i = slide index
            h_page, page_i = hid_sec[:, 0, :].unsqueeze(0), 0

            # slide_counter = 0
            while True:
                # PT slide GRU takes in matrix of zeroes (blank slide) and the slide PT 
                hid_page, h_page = self.rnn_page(T.zeros((1, 1, 16)).to(device), h_page)
                # get current ground-truth section
                ref = inp[sec_i]
                # if ground truth is nonexistent, there is no attention nor context 
                if ref is None:
                    att = None
                    cxt = T.zeros((1, 1, 2048)).float().to(device)
                else:
                    # multiply attention map of current slide (hid_page) and the ground-truth slide to create an attention map over section embeddings
                    att = T.bmm(self.att_page(hid_page), ref.transpose(1, 2))
                    # multiply the soft-max of attention map and ground-truth slide to compute bilinear compatibility
                    cxt = T.bmm(T.nn.functional.softmax(att, dim=2), ref)
                # concatenate the hidden page and the context
                tmp = T.cat([hid_page, cxt], dim=2).squeeze(1)
                # apply binary decision [NEW SLIDE] = 0  or [END SEC] = 1
                tok = self.fc_tok(tmp)
                # add resulting decision to predicted page token list (0,1)
                pd_tok_page.append(tok)

                # if teacher forcing is true, make the current ground-truth slide the next token
                if teacher==True:
                    tok = item['out_tok_page'][sec_i][page_i]

                # otherwise use the predicted slide (via softmax)
                else:
                    tok = np.argmax(tok.data.cpu().numpy()[0], axis=0)
                # add resulting generated slide to output page token list (0,1)
                out_tok_page.append(tok)

                # if decision is 1 = [END SEC], break the slide loop
                if tok==1:
                    break
                # if decision is 0 = [NEW SLIDE], add objects
                else:
                    # lists of token objects and output objects
                    t_obj, o_obj = [], []
                    # list for slide objects 
                    u_tok = []
                    # initialise object progress tracker (h_obj) and object index (obj_i)
                    h_obj, obj_i = hid_page[:, 0, :].unsqueeze(0), 0
                    while True:
                        # PT object GRU takes in matrix of zeroes and the object  PT 
                        hid_obj, h_obj = self.rnn_obj(T.zeros((1, 1, 16)).to(device), h_obj)

                        # get current ground-truth object
                        ref = inp[sec_i]
                        # if ground truth is nonexistent, there is no attention nor context 
                        if ref is None:
                            att = None
                            cxt = T.zeros((1, 1, 2048)).float().to(device)
                        else:
                            # multiply attention map of current onject (hid_obj) and the ground-truth slide to create an attention map over section embeddings
                            att = T.bmm(self.att_obj(hid_obj), ref.transpose(1, 2))
                            # multiply the soft-max of attention map and ground-truth slide to compute bilinear compatibility
                            cxt = T.bmm(T.nn.functional.softmax(att, dim=2), ref)
                        
                        # concatenate the hidden page and the context
                        tmp = T.cat([hid_obj, cxt], dim=2).squeeze(1)
                        # apply binary decision [NEW OBJECT] = 0  or [END SLIDE] = 1; save the attention map in obj
                        tok, obj = self.fc_tok(tmp), att.squeeze(0) if att is not None else None
                        
                        # add resulting decision made for object into t_obj
                        t_obj.append(tok)
                        # add attention map of object (used to find location of object) into o_obj
                        o_obj.append(obj)

                        # if teacher forcing is true, make the current ground-truth object the next token
                        if teacher==True:
                            tok = item['out_tok_obj'][sec_i][page_i][obj_i]

                        # otherwise use the predicted predicted (via softmax)
                        else:
                            tok = np.argmax(tok.data.cpu().numpy()[0], axis=0)
                            # if object index is 5, make tok = 1 (slide only allows 5 objects in it)
                            if obj_i==5:
                                tok = 1
                        # add generated object to u_tok
                        u_tok.append(tok)

                        # if decision is 1, then break the object loop
                        if tok==1:
                            break
                        
                        # increment object index
                        obj_i += 1

                    # add all instances of object decisions into predicted object token list (0,1)
                    pd_tok_obj.append(t_obj)
                    # add all instances of object locations into Predicted Object list
                    pd_obj.append(o_obj)
                    # add all instances of generated objects into output object token list (0,1)
                    out_tok_obj.append(u_tok)

                # increment slide index
                page_i += 1

            # decision made for section-slide [END SEC] or [NEW SLIDE]
            ret['pd_tok_page'].append(pd_tok_page)
            # decisions made for slide-object [END SLIDE] or [NEW OBJECT]
            ret['pd_tok_obj'].append(pd_tok_obj)
            # object locations for each slide
            ret['pd_obj'].append(pd_obj)
            # argmax of pd_tok_page
            ret['out_tok_page'].append(out_tok_page)
            # argmax of pd_tok_obj
            ret['out_tok_obj'].append(out_tok_obj)

        return ret

class Loss(T.nn.Module):
    def __init__(self):
        super().__init__()
        # initialise cross-entropy loss
        self.loss_ce = T.nn.CrossEntropyLoss()

    def forward(self, out, item):
        # initialise lists of structural lost (ls_tok) and content loss (ls_obj)
        ls_tok, ls_obj = [], []

        # MEASURES STRUCTURAL LOSS
        # for each section i and (ground-truth, predicted) pairs in predicted page token list and output page token list - (series of slide binary decisions)
        for sec_i, (tok_page_pd, tok_page_gd) in enumerate(zip(out['pd_tok_page'], item['out_tok_page'])):
            # convert both ground-truth and generated into comparable numpy arrays
            pd, gd = T.cat(tok_page_pd, dim=0), T.from_numpy(np.array(tok_page_gd)).long().to(device)
            # implement cross-entropy loss 
            ls = self.loss_ce(pd, gd)
            ls_tok.append(ls)

            # for each slide and (ground-truth, predicted) pairs in predicted object token list and output object token list - (series of object binary decisions)
            for page_i, (tok_obj_pd, tok_obj_gd) in enumerate(zip(out['pd_tok_obj'][sec_i], item['out_tok_obj'][sec_i])):
                # convert both ground-truth and generated into comparable numpy arrays
                pd, gd = T.cat(tok_obj_pd, dim=0), T.from_numpy(np.array(tok_obj_gd)).long().to(device)
                # implement cross-entropy loss 
                ls = self.loss_ce(pd, gd)
                ls_tok.append(ls)

            # CONTENT LOSS: for each slide and (ground-truth, predicted) pairs in predicted object list and output object list
            for page_i, (obj_pd, obj_gd) in enumerate(zip(out['pd_obj'][sec_i], item['out_obj'][sec_i])):
                if len(obj_pd)==1:
                    continue
                obj_pd, obj_gd = obj_pd[:-1], obj_gd[:-1]
                # eliminate other Nones from groundtruth
                none_list = [i for i in range(len(obj_gd)) if obj_gd[i] is None]
                obj_pd = [obj_pd[i] for i in range(len(obj_pd)) if i not in none_list]
                obj_gd = [obj_gd[i] for i in range(len(obj_gd)) if i not in none_list]
            
                # convert both ground-truth and generated into comparable numpy arrays
                pd, gd = T.cat(obj_pd, dim=0), T.from_numpy(np.array(obj_gd)).long().to(device)
                # implement cross-entropy loss 
                ls = self.loss_ce(pd, gd)
                ls_obj.append(ls)

        # find the mean of structural loss
        ls = 0
        for x in ls_tok:
            ls += x
        if len(ls_tok)>0:
            ls /= len(ls_tok)
        ls_tok = ls

        # find the mean of content loss
        ls = 0
        for x in ls_obj:
            ls += x
        if len(ls_obj)>0:
            ls /= len(ls_obj)
        ls_obj = ls

        # return final loss
        return ls_tok, ls_obj

if __name__=='__main__':

    dat = {}
    for conf in json.load(open('./data/v1.0/train_val_test.json', 'r')):
        pkl = pickle.load(open('./data/v1.0/%s.pkl'%(conf), 'rb'))
        dat[conf] = {}
        for item in pkl:
            idd = item['idd']
            dat[conf][idd] = item
    dl_tr = DLoader(dat, 'train')
    dl_vl = DLoader(dat, 'val')

    model = Model().to(device)
    loss_func = Loss().to(device)

    model.train()
    for i in tqdm(range(len(dl_tr)), ascii=True):
        item = dl_tr[i]
        out = model(item)
        ls_tok, ls_obj = loss_func(out, item)

    model.eval()
    for i in tqdm(range(len(dl_vl)), ascii=True):
        item = dl_vl[i]
        out = model(item)
        ls_tok, ls_obj = loss_func(out, item)
