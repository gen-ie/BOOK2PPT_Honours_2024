import pickle, json, os
from tqdm import tqdm
from glob import glob

import numpy as np

import torch as T

from dataset import *

class Model(T.nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn_sec = T.nn.GRU(2048, 256,
                                batch_first=True)
        self.rnn_page = T.nn.GRU(16, 256,
                                 batch_first=True)
        self.rnn_obj = T.nn.GRU(16, 256,
                                batch_first=True)

        self.rnn_text = T.nn.GRU(1024, 1024,
                                 batch_first=True, bidirectional=True)
        self.fc_fig = T.nn.Linear(1024+2048, 2048)

        self.att_page = T.nn.Linear(256, 2048,
                                    bias=False)
        self.att_obj = T.nn.Linear(256, 2048,
                                    bias=False)

        self.fc_tok = T.nn.Sequential(*[T.nn.Linear(256+2048, 2)])

    def forward(self, item, teacher=True):
        ret = {}

        ret['pd_tok_page'], ret['pd_tok_obj'], ret['pd_obj'] = [], [], []
        ret['out_tok_page'], ret['out_tok_obj'] = [], []

        # PREPARE
        if item['inp_emb_pix'] is not None:
            inp_fig = T.cat([T.from_numpy(item['inp_emb_pix']), T.from_numpy(item['inp_emb_cap'])], dim=1).unsqueeze(0).cuda()
            inp_fig = self.fc_fig(inp_fig)
        else:
            inp_fig = None

        inp, inp_sec = [], []
        for emb in item['inp_emb_text']:
            if emb is not None:
                emb = T.from_numpy(emb).unsqueeze(0).cuda()
                emb, _ = self.rnn_text(emb)
                inp_sec.append(T.cat([emb[:, -1, :1024], emb[:, 0, 1024:]], dim=1).unsqueeze(1))
                if inp_fig is not None:
                    o = T.cat([emb, inp_fig], dim=1)
                else:
                    o = emb
            else:
                inp_sec.append(T.zeros((1, 1, 2048)).float().cuda())
                if inp_fig is not None:
                    o = inp_fig
                else:
                    o = None
            inp.append(o)

        # DECODING
        h_sec = None
        for sec_i in range(len(inp)):
            hid_sec, h_sec = self.rnn_sec(inp_sec[sec_i], h_sec)

            pd_tok_page, pd_tok_obj, pd_obj = [], [], []
            out_tok_page, out_tok_obj = [], []
            h_page, page_i = hid_sec[:, 0, :].unsqueeze(0), 0
            while True:
                hid_page, h_page = self.rnn_page(T.zeros((1, 1, 16)).cuda(), h_page)

                ref = inp[sec_i]
                if ref is None:
                    att = None
                    cxt = T.zeros((1, 1, 2048)).float().cuda()
                else:
                    att = T.bmm(self.att_page(hid_page), ref.transpose(1, 2))
                    cxt = T.bmm(T.nn.functional.softmax(att, dim=2), ref)

                tmp = T.cat([hid_page, cxt], dim=2).squeeze(1)
                tok = self.fc_tok(tmp)
                pd_tok_page.append(tok)

                if teacher==True:
                    tok = item['out_tok_page'][sec_i][page_i]
                else:
                    tok = np.argmax(tok.data.cpu().numpy()[0], axis=0)
                out_tok_page.append(tok)

                if tok==1:
                    break
                else:
                    t_obj, o_obj = [], []
                    u_tok = []
                    h_obj, obj_i = hid_page[:, 0, :].unsqueeze(0), 0
                    while True:
                        hid_obj, h_obj = self.rnn_obj(T.zeros((1, 1, 16)).cuda(), h_obj)

                        ref = inp[sec_i]
                        if ref is None:
                            att = None
                            cxt = T.zeros((1, 1, 2048)).float().cuda()
                        else:
                            att = T.bmm(self.att_obj(hid_obj), ref.transpose(1, 2))
                            cxt = T.bmm(T.nn.functional.softmax(att, dim=2), ref)

                        tmp = T.cat([hid_obj, cxt], dim=2).squeeze(1)
                        tok, obj = self.fc_tok(tmp), att.squeeze(0) if att is not None else None
                        t_obj.append(tok)
                        o_obj.append(obj)

                        if teacher==True:
                            tok = item['out_tok_obj'][sec_i][page_i][obj_i]
                        else:
                            tok = np.argmax(tok.data.cpu().numpy()[0], axis=0)
                            if obj_i==5:
                                tok = 1
                        u_tok.append(tok)

                        if tok==1:
                            break

                        obj_i += 1

                    pd_tok_obj.append(t_obj)
                    pd_obj.append(o_obj)
                    out_tok_obj.append(u_tok)

                page_i += 1

            ret['pd_tok_page'].append(pd_tok_page)
            ret['pd_tok_obj'].append(pd_tok_obj)
            ret['pd_obj'].append(pd_obj)
            ret['out_tok_page'].append(out_tok_page)
            ret['out_tok_obj'].append(out_tok_obj)

        return ret

class Loss(T.nn.Module):
    def __init__(self):
        super().__init__()

        self.loss_ce = T.nn.CrossEntropyLoss()

    def forward(self, out, item):
        ls_tok, ls_obj = [], []

        for sec_i, (tok_page_pd, tok_page_gd) in enumerate(zip(out['pd_tok_page'], item['out_tok_page'])):
            pd, gd = T.cat(tok_page_pd, dim=0), T.from_numpy(np.array(tok_page_gd)).long().cuda()
            ls = self.loss_ce(pd, gd)
            ls_tok.append(ls)

            for page_i, (tok_obj_pd, tok_obj_gd) in enumerate(zip(out['pd_tok_obj'][sec_i], item['out_tok_obj'][sec_i])):
                pd, gd = T.cat(tok_obj_pd, dim=0), T.from_numpy(np.array(tok_obj_gd)).long().cuda()
                ls = self.loss_ce(pd, gd)
                ls_tok.append(ls)

            for page_i, (obj_pd, obj_gd) in enumerate(zip(out['pd_obj'][sec_i], item['out_obj'][sec_i])):
                if len(obj_pd)==1:
                    continue
                obj_pd, obj_gd = obj_pd[:-1], obj_gd[:-1]
                pd, gd = T.cat(obj_pd, dim=0), T.from_numpy(np.array(obj_gd)).long().cuda()
                ls = self.loss_ce(pd, gd)
                ls_obj.append(ls)

        ls = 0
        for x in ls_tok:
            ls += x
        if len(ls_tok)>0:
            ls /= len(ls_tok)
        ls_tok = ls

        ls = 0
        for x in ls_obj:
            ls += x
        if len(ls_obj)>0:
            ls /= len(ls_obj)
        ls_obj = ls

        return ls_tok, ls_obj

if __name__=='__main__':

    dat = {}
    for conf in json.load(open('../data/v1.0/train_val_test.json', 'r')):
        pkl = pickle.load(open('../data/v1.0/%s.pkl'%(conf), 'rb'))
        dat[conf] = {}
        for item in pkl:
            idd = item['idd']
            dat[conf][idd] = item
    dl_tr = DLoader(dat, 'train')
    dl_vl = DLoader(dat, 'val')

    model = Model().cuda()
    loss_func = Loss().cuda()

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
