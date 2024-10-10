import torch as T
import pickle

from dataset import *
from model import *

dat = {}
for conf in json.load(open('../../data/v1.0/book_json.json', 'r')):
    pkl = pickle.load(open(f'../../books/{conf}.pkl', 'rb'))
    dat[conf] = {}
    for item in pkl:
        idd = item['idd']
        dat[conf][idd] = item

dl_tr = DLoader(dat, typ='train')
dl_vl = DLoader(dat, typ='val')

device = T.device("cuda" if T.cuda.is_available() else "cpu")
loaded_model = T.load('../../models/model_hse-tf.pt', map_location=T.device(device), weights_only=True)
model = Model().to(device)
model.load_state_dict(loaded_model)

# Extract the last layers
params_change = ['att_page.weight', 'att_obj.weight', 'fc_tok.0.weight', 'fc_tok.0.bias']

# Freeze all layers except the last one
for name, param in model.named_parameters():
    if any(param_name in name for param_name in params_change):
        param.requires_grad = True
    else:
        param.requires_grad = False  

loss_func = Loss().to(device)
optzr = T.optim.Adam(model.parameters(), lr=0.0003)

# create directory for loss logs and model
PATH = 'story_base'
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
        with T.no_grad():  # Disable gradient calculation for validation
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
        json.dump(log, open('%s/log.json'%(PATH), 'w'))

T.save(model.state_dict(), '%s/storybook_model_base_final.pt'%(PATH))