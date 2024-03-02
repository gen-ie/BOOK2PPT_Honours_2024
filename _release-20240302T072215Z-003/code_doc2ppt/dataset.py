import pickle, json, os
from tqdm import tqdm
from glob import glob

import numpy as np

import torch as T

class DLoader:
    def __init__(self, dat, typ='train'):
        super().__init__()

        self.dat, self.typ = dat, typ

        self.tr_vl_ts = json.load(open('../data/v1.0/train_val_test.json', 'r'))
        self.lst = []
        if typ=='test':
            rec_human = pickle.load(open('../data/record_human.pkl', 'rb'))
            for conf in rec_human:
                for idd in rec_human[conf]:
                    self.lst.append([conf, idd])
        else:
            for conf in self.tr_vl_ts:
                for idd in self.tr_vl_ts[conf][typ]:
                    self.lst.append([conf, idd])

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, idx):
        ret = {}

        conf, idd = self.lst[idx]
        ret['conf'], ret['idd'] = conf, idd

        ret['inp_text'], ret['inp_emb_text'] = [], []
        for sec in self.dat[conf][idd]['paper']['sections']:
            if len(sec)>=1:
                txt = [item['text'] for item in sec]
                emb = np.array([item['embedding'] for item in sec]) # [len_sent, 1024]
            else:
                txt, emb = None, None
            ret['inp_text'].append(txt)
            ret['inp_emb_text'].append(emb)

        ret['inp_fig'] = []
        ret['inp_pix'], ret['inp_emb_pix'] = [], []
        ret['inp_cap'], ret['inp_emb_cap'] = [], []
        for fig in self.dat[conf][idd]['paper']['figures']:
            ret['inp_fig'].append(fig['name'])
            ret['inp_pix'].append(fig['pixel'])
            ret['inp_emb_pix'].append(fig['feature'])
            ret['inp_cap'].append(fig['caption'])
            ret['inp_emb_cap'].append(fig['embedding'])
        if len(ret['inp_emb_pix'])>=1:
            ret['inp_emb_pix'] = np.array(ret['inp_emb_pix']) # [len_fig, 2048]
            ret['inp_emb_cap'] = np.array(ret['inp_emb_cap']) # [len_fig, 1024]
        else:
            ret['inp_emb_pix'] = None
            ret['inp_emb_cap'] = None

        ret['out_tok_page'], ret['out_tok_obj'], ret['out_obj'], ret['out_gdt'] = [], [], [], []
        for sec_i, sec in enumerate(self.dat[conf][idd]['slide']['pages']):
            tok_page, tok_obj, out_obj, out_gdt = [], [], [], []
            for item in sec:
                tok_page.append(0)

                t_obj, o_obj, o_gdt = [], [], []
                for obj in item['page']:
                    t_obj.append(0)
                    o_obj.append(obj['label'])
                    o_gdt.append(obj['text'])
                for obj in item['figure']:
                    t_obj.append(0)
                    off = len(ret['inp_text'][sec_i]) if ret['inp_text'][sec_i] is not None else 0
                    o_obj.append(off+obj['label'])
                    o_gdt.append(obj['label'])
                t_obj.append(1)
                o_obj.append(None)
                o_gdt.append(None)

                tok_obj.append(t_obj)
                out_obj.append(o_obj)
                out_gdt.append(o_gdt)

            tok_page.append(1)
            ret['out_tok_page'].append(tok_page)
            ret['out_tok_obj'].append(tok_obj)
            ret['out_obj'].append(out_obj)
            ret['out_gdt'].append(out_gdt)

        return ret

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
    dl_ts = DLoader(dat, 'test')
    print('Train: %d, Val: %d, Test: %d'%(len(dl_tr), len(dl_vl), len(dl_ts)))

    for i in range(1000, 1006):
        item = dl_tr[i]
        print(item['conf'], item['idd'])

        print(item['out_tok_page'])
        print(item['out_tok_obj'])
        print(item['out_obj'])
        print('-----')
