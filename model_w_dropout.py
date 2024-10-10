# Progress Tracker with Dropout
import torch as T
import torch.nn as nn
import numpy as np

device = T.device("cuda" if T.cuda.is_available() else "cpu")

class DOC2PPTWithDropout(nn.Module):
    def __init__(self, original_model, dropout=0.3):
        super(DOC2PPTWithDropout, self).__init__()
        
        # Copy the original RNN and FC layers
        self.rnn_sec = original_model.rnn_sec
        self.rnn_page = original_model.rnn_page
        self.rnn_obj = original_model.rnn_obj
        self.rnn_text = original_model.rnn_text
        
        # Add dropout after each RNN layer
        self.dropout_rnn_sec = nn.Dropout(dropout)
        self.dropout_rnn_page = nn.Dropout(dropout)
        self.dropout_rnn_obj = nn.Dropout(dropout)
        
        # Copy the original fully connected layers and attention layers
        self.fc_fig = original_model.fc_fig
        self.fc_tok = original_model.fc_tok
        self.att_page = original_model.att_page
        self.att_obj = original_model.att_obj
    
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
            h_sec = self.dropout_rnn_sec(h_sec)

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
                h_page = self.dropout_rnn_page(h_page)

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
                        h_obj = self.dropout_rnn_obj(h_obj)

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