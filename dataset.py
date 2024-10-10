# ORIGINAL CODE FROM DOC2PPT REPOSITORY - Docement Reader
# Slight modifications were added alongside comments

import pickle, json, os
from tqdm import tqdm
from glob import glob

import numpy as np

import torch as T

class DLoader:
    def __init__(self, dat, typ='train', domain='stories'):
        super().__init__()
        # initialise data and its type (train, validate, test)
        self.dat, self.typ = dat, typ
        if domain == 'stories':
            TR_VL_TEST_FILE = './data/v1.0/book_json.json'
            self.tr_vl_ts = json.load(open(TR_VL_TEST_FILE, 'r'))
        elif domain == 'original':
            TR_VL_FILE = './data/v1.0/train_val_test_2.json'
            TEST_FILE = './data/record_human.pkl'
            self.tr_vl_ts = json.load(open(TR_VL_FILE, 'r'))  

        self.lst = []
        # if type is test, extract human-made document-slide pairs
        if typ=='test':
            # for each paper-slide pair in research type
            if domain == 'original':
                rec_human = pickle.load(open(TEST_FILE, 'rb')) 
                # for each research paper type in dataset
                for conf in rec_human:
                    if conf in self.tr_vl_ts.keys():
                        for idd in rec_human[conf]:
                            self.lst.append([conf, idd])
            else:
                rec_human = json.load(open(TR_VL_TEST_FILE, 'r'))
                for conf in rec_human:
                    for idd in rec_human[conf][typ]:
                        self.lst.append([conf, idd])
        # else, use the data as indicated by the train_val_test.json file
        else:
            for conf in self.tr_vl_ts:
                for idd in self.tr_vl_ts[conf][typ]:
                    self.lst.append([conf, idd])

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, idx):
        # return dictionary
        ret = {}

        # get the doc-slide pairs
        conf, idd = self.lst[idx]
        ret['conf'], ret['idd'] = conf, idd

        ret['inp_text'], ret['inp_emb_text'] = [], []
        # for every section in a paper
        for sec in self.dat[conf][idd]['paper']['sections']:
            if len(sec)>=1:
                # get a list of sentences
                txt = [item['text'] for item in sec]
                # and their embeddings
                emb = np.array([item['embedding'] for item in sec]) # [len_sent, 1024]
            else:
                txt, emb = None, None
            # add sentences in 'inp_text' and their embeddings in 'inp_emb_text'
            ret['inp_text'].append(txt)
            ret['inp_emb_text'].append(emb)

        ret['inp_fig'] = []
        ret['inp_pix'], ret['inp_emb_pix'] = [], []
        ret['inp_cap'], ret['inp_emb_cap'] = [], []
        # for every figure in a paper
        for fig in self.dat[conf][idd]['paper']['figures']:
            # get the features of the figure as well as the features' embeddings 
            ret['inp_fig'].append(fig['name'])
            ret['inp_pix'].append(fig['pixel'])
            ret['inp_emb_pix'].append(fig['feature'])
            ret['inp_cap'].append(fig['caption'])
            ret['inp_emb_cap'].append(fig['embedding'])
        if len(ret['inp_emb_pix'])>=1:
            ret['inp_emb_pix'] = np.array(ret['inp_emb_pix'], dtype=np.float32) # [len_fig, 2048]
            if ret['inp_emb_cap'] == [[]]:
                # considering that there is no caption for these generated images, just pad 'inp_emb_cap' with zeroes (1, 1024)
                ret['inp_emb_cap'] = np.zeros((len(ret['inp_emb_cap']), 1024), dtype=np.float32) # [len_fig, 1024]
            else:
                ret['inp_emb_cap'] = np.array(ret['inp_emb_cap']) # [len_fig, 1024]
        else:
            ret['inp_emb_pix'] = None
            ret['inp_emb_cap'] = None

        ret['out_tok_page'], ret['out_tok_obj'], ret['out_obj'], ret['out_gdt'], ret['out_bbox'] = [], [], [], [], []
        # for every section page in paper and corresponding slides in powerpoint
        for sec_i, sec in enumerate(self.dat[conf][idd]['slide']['pages']):
            tok_page, tok_obj, out_obj, out_gdt, out_bbox= [], [], [], [], []
            # for every slide in section
            for item in sec:
                # append first slide decision [NEW SLIDE]
                tok_page.append(0)

                t_obj, o_obj, o_gdt, bbox = [], [], [], []
                # for every available text object in slide
                for obj in item['page']:
                    # [NEW OBJECT]
                    t_obj.append(0)
                    # add label and the text itself
                    o_obj.append(obj['label'])
                    o_gdt.append(obj['text'])
                    bbox.append(obj['bbox'])
                # for every available figure object in slide
                for obj in item['figure']:
                    # [NEW OBJECT]
                    t_obj.append(0)
                    # offset?
                    off = len(ret['inp_text'][sec_i]) if ret['inp_text'][sec_i] is not None else 0

                    o_obj.append(off+obj['label'])
                    o_gdt.append(obj['label'])
                    bbox.append(obj['bbox'])

                # append [END SLIDE] decision
                t_obj.append(1)
                o_obj.append(None)
                o_gdt.append(None)
                bbox.append(None)

                tok_obj.append(t_obj)
                out_obj.append(o_obj)
                out_gdt.append(o_gdt)
                out_bbox.append(bbox)

            # append [END SEC] decision
            tok_page.append(1)
            # decisions on making new slides for section or ending section
            ret['out_tok_page'].append(tok_page)
            # decisions on making new objects for section or ending slide
            ret['out_tok_obj'].append(tok_obj)
            # object labels
            ret['out_obj'].append(out_obj)
            # sentence and figure objects found in docs and slides
            ret['out_gdt'].append(out_gdt)
            ret['out_bbox'].append(out_bbox)

        # get the sizes of the ground-truth presentation slides
        if ('size' in self.dat[conf][idd]['slide'].keys()):
            ret['slide_size'] = self.dat[conf][idd]['slide']['size']

        return ret

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
    dl_ts = DLoader(dat, 'test')
    print('Train: %d, Val: %d, Test: %d'%(len(dl_tr), len(dl_vl), len(dl_ts)))

    for i in range(1000, 1006):
        item = dl_tr[i]
        print(item['conf'], item['idd'])

        print(item['out_tok_page'])
        print(item['out_tok_obj'])
        print(item['out_obj'])
        print('-----')
