
# coding: utf-8

import os
os.chdir('Machine_Translation_NLP')


import json
import pandas as pd
import time

import jieba
import re

from config import fact_seperator, element_seperator
from config import vocab_pred, vocab_pred_size, vocab_prefix
from config import UNK_index, PAD_index, SOS_index, EOS_index 
from config import OOV_pred_index, PAD_pred_index, EOS_pred_index


def replaceMisspred(predicate):
    '''replace missing predicate
    '''
    if predicate == '_':
        return 'P'
    else:
        return predicate
    
def character_segmentation(string):
    res = []
    for part in list(jieba.cut(string, cut_all=False)):
        if re.match('^[\da-zA-Z]+$', part):
            res.append(part)
        else:
            res.extend(list(part))
    return res

def load_preprocess_data(data_add):
    saoke = []
    with open(data_add, 'r') as f:
        for line in f:
            saoke.append(json.loads(line))
    data = []
    # list of dict
    for sample in saoke:
        # remove some exceptions with empty facts
        if sample['logic'] == []:
            continue
        # tokenize src sentence
        sample_processed = dict()
        sample_processed['src_org'] = sample['natural'].replace('`', '')
        #sample_processed['src'] = list(jieba.cut(sample['natural'], cut_all=False))
        sample_processed['src'] = character_segmentation(sample_processed['src_org'])
        
        # transform fact list into str and tokenize
        # $ separates facts; @ separate elements for one fact; & separate objects for one fact
        sample_processed['tgt_org'] = sample['logic']
        logic_list = []
        logic_set = set()
        for fact in sample['logic']:
            fact = element_seperator.join([fact['subject'], replaceMisspred(fact['predicate']), '&'.join(fact['object'])])
            if not fact in logic_set:
                logic_set.add(fact)
                logic_list.append(fact)
        sample_processed['tgt_list'] = logic_list # remove repeated facts (606)
        logic_str = fact_seperator.join(logic_list)
        sample_processed['tgt'] = character_segmentation(logic_str)
        data.append(sample_processed)
    return data


import numpy as np
from collections import Counter
from itertools import dropwhile

class Lang:
    def __init__(self, name, emb_pretrained_add=None, max_vocab_size=None):
        self.name = name
        self.word2index = None 
        self.index2word = None 
        self.max_vocab_size = max_vocab_size  # Count SOS and EOS
        self.vocab_size = None
        self.emb_pretrained_add = emb_pretrained_add
        self.embedding_matrix = None

    def build_vocab(self, data):
        all_tokens = []
        for sample in data:
            all_tokens.extend(sample['src'])
            all_tokens.extend(sample['tgt'])  
        token_counter = Counter(all_tokens)
        print('The number of unique tokens totally in dataset: ', len(token_counter))
        # remove word with freq==1 
        for key, count in dropwhile(lambda key_count: key_count[1] > 1, token_counter.most_common()):
            del token_counter[key]
        
        if self.max_vocab_size:
            vocab, count = zip(*token_counter.most_common(self.max_vocab_size))
        else:
            vocab, count = zip(*token_counter.most_common())
        
        self.index2word = vocab_prefix + list(vocab)
        word2index = dict(zip(self.index2word, range(0, len(self.index2word)))) 
#         word2index = dict(zip(vocab, range(len(vocab_prefix),len(vocab_prefix)+len(vocab)))) 
#         for idx, token in enumerate(vocab_prefix):
#             word2index[token] = idx
        self.word2index = word2index
        self.vocab_size = len(self.index2word)
        return None 

    def build_emb_weight(self):
        words_emb_dict = load_emb_vectors(self.emb_pretrained_add)
        emb_weight = np.zeros([self.vocab_size, 300])
        for i in range(len(vocab_prefix), self.vocab_size):
            emb = words_emb_dict.get(self.index2word[i], None)
            if emb is not None:
                try:
                    emb_weight[i] = emb
                except:
                    pass
                    #print(len(emb), self.index2word[i], emb)
        self.embedding_matrix = emb_weight
        return None

def load_emb_vectors(fasttest_home):
    max_num_load = 500000
    words_dict = {}
    with open(fasttest_home) as f:
        for num_row, line in enumerate(f):
            if num_row >= max_num_load:
                break
            s = line.split()
            words_dict[s[0]] = np.asarray(s[1:])
    return words_dict


def text2index(data, key, word2index):
    '''
    transform tokens into index as input for both src and tgt
    '''
    indexdata = []
    for line in data:
        line = line[key]
        indexdata.append([word2index[c] if c in word2index.keys() else UNK_index for c in line])
    print('finish indexing')
    return indexdata

def text2index_tgt(data, key, word2index):
    '''
    transform tokens into index as input for tgt
    '''
    indexdata = []
    for line in data:
        line = line[key]
        line_index = []
        for fact in line:
            fact_list = character_segmentation(fact) + [fact_seperator]
            line_index.append([word2index.get(c, UNK_index) for c in fact_list])
        indexdata.append(line_index)
    print('finish indexing')
    return indexdata

def construct_Lang(name, data, emb_pretrained_add = None, max_vocab_size = None):
    lang = Lang(name, emb_pretrained_add, max_vocab_size)
    lang.build_vocab(data)
    if emb_pretrained_add:
        lang.build_emb_weight()
    return lang

def text2symbolindex(data, key, word2index):
    '''get generation label for tgt 
    '''
    indexdata = []
    aaa = -1
    for line in data:
        line_index = []
        line = line[key]
        for fact in line:
            fact_list = character_segmentation(fact) + [fact_seperator]
            line_index.append([word2index.get(c, OOV_pred_index) for c in fact_list])
        indexdata.append(line_index)
    print('symbol label finish')
    return indexdata

def copy_indicator(data, src_key='src', tgt_key='tgt_list'):
    '''get copy label for tgt
    '''
    indicator = []
    for sample in data:
        tgt_list = sample[tgt_key]
        src = sample[src_key]
        sample_list = []
        for fact in tgt_list:
            fact_list = character_segmentation(fact) + [fact_seperator]
            matrix = np.zeros((len(fact_list), len(src)), dtype=int)
            for m in range(len(fact_list)):
                for n in range(len(src)):
                    if fact_list[m] == src[n]:
                        matrix[m,n] = 1
            sample_list.append(matrix)
        indicator.append(sample_list)
    return indicator


from torch.utils.data import Dataset, DataLoader
from itertools import dropwhile

class VocabDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """
    def __init__(self, src_index, tgt_index, tgt_symbolindex, tgt_indicator, data, src_clip=None, tgt_clip=None):
        """
        @param data_list: list of character
        @param target_list: list of targets

        """
        self.src_clip = src_clip
        self.tgt_clip = tgt_clip
        self.src_list, self.tgt_list = src_index, tgt_index
        self.data = data
        self.tgt_symbolindex, self.tgt_indicator  = tgt_symbolindex, tgt_indicator
        
        assert (len(self.src_list) == len(self.tgt_list) == len(self.tgt_symbolindex)== len(self.tgt_indicator))
        #self.word2index = word2index

    def __len__(self):
        return len(self.src_list)

    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        src = self.src_list[key]
        tgt = self.tgt_list[key]
        src_org = self.data[key]['src']
        tgt_org = self.data[key]['tgt']
        tgt_org_fact_num = len(tgt)
        tgt_sym = self.tgt_symbolindex[key]
        tgt_ind = self.tgt_indicator[key]
        
        if self.src_clip is not None:
            src = src[:self.src_clip]
            tgt_ind = [f_ind[:,:self.src_clip] for f_ind in tgt_ind]
        src_length = len(src)

        tgt_length = sum([len(f) for f in tgt]) - 1
        if self.tgt_clip is not None:
            while tgt_length > self.tgt_clip and tgt_org_fact_num > 1:
                drop_fact_idx = np.random.choice(tgt_org_fact_num, 1)[0]
                tgt = [tgt[i] for i in range(tgt_org_fact_num) if i!=drop_fact_idx]
                tgt_sym = [tgt_sym[i] for i in range(tgt_org_fact_num) if i!=drop_fact_idx]
                tgt_ind = [tgt_ind[i] for i in range(tgt_org_fact_num) if i!=drop_fact_idx]
                tgt_org_fact_num -= 1
#                 if len(tgt)==0:
#                     print(tgt_org_fact_num, len(tgt))
                tgt_length = sum([len(f) for f in tgt]) - 1
        
        return src, src_length, tgt, tgt_length, tgt_sym, tgt_ind, src_org, tgt_org
        #return src_org, src_tensor, src_true_len, tgt_org, tgt_tensor, tgt_true_len, tgt_label_vocab, tgt_label_copy 

def vocab_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all
    data have the same length
    """
    src_list = []
    tgt_list = []
    #tgt_fact_num_list = []
    src_length_list = []
    tgt_length_list = []
    tgt_symbol_list = []
    tgt_indicator_list = []
    src_org_list = []
    tgt_org_list = []

    for datum in batch:
        src_length_list.append(datum[1]) # 不用加1；eos不算
        tgt_length_list.append(datum[3]+1)
        
    
    batch_max_src_length = np.max(src_length_list)
    # padding
    for datum in batch:
        #+[EOS_index] -1
        padded_vec = np.pad(np.array(datum[0]), 
                                pad_width=((0, batch_max_src_length-datum[1])),
                                mode="constant", constant_values=PAD_index)
        src_list.append(padded_vec)

        tgt_list.append(datum[2])
        tgt_symbol_list.append(datum[4])
        tgt_indicator_list.append(datum[5])
        src_org_list.append(datum[6])
        tgt_org_list.append(datum[7])
    
    # re-order
    ind_dec_order = np.argsort(src_length_list)[::-1]
    
    src_list = np.array(src_list)[ind_dec_order]
    src_length_list = np.array(src_length_list)[ind_dec_order]
    tgt_list = [tgt_list[i] for i in ind_dec_order]
    tgt_length_list = np.array(tgt_length_list)[ind_dec_order]
    tgt_symbol_list = [tgt_symbol_list[i] for i in ind_dec_order]
    #print(tgt_indicator_list[0].dtype, tgt_indicator_list[0][:5][:5])
    tgt_indicator_list = [tgt_indicator_list[i] for i in ind_dec_order]
    #print(tgt_indicator_list.dtype, tgt_indicator_list.shape)
    src_org_list = [src_org_list[i] for i in ind_dec_order]
    tgt_org_list = [tgt_org_list[i] for i in ind_dec_order]
    
    return [torch.from_numpy(src_list), 
            torch.from_numpy(src_length_list), 
            tgt_list, 
            tgt_length_list,
            tgt_symbol_list,
            tgt_indicator_list,
            src_org_list,
            tgt_org_list,           
           ]


# load data
data_add = '/scratch/tx443/NLU/project/SAOKE_DATA.json'
data = load_preprocess_data(data_add)


# split train val test
from sklearn.model_selection import train_test_split
train_data, val_test_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(val_test_data, test_size=0.5, random_state=45)


# build vocab from train for input indexing
trainLang = construct_Lang('train', train_data)

# build generation vocab for prediction
word2symbolindex = {}
for idx, token in enumerate(vocab_pred):
        word2symbolindex[token] = idx

# check
assert(UNK_index==trainLang.word2index['<UNK>'])
assert(PAD_index==trainLang.word2index['<PAD>'])
assert(SOS_index==trainLang.word2index['<SOS>'])
assert(EOS_index==trainLang.word2index['<EOS>'])

assert(OOV_pred_index==word2symbolindex['<OOV>'])
assert(PAD_pred_index==word2symbolindex['<PAD>'])
assert(EOS_pred_index==word2symbolindex['<EOS>'])


# input indexing for src
train_src_input_index = text2index(train_data, 'src', trainLang.word2index) 
val_src_input_index = text2index(val_data, 'src', trainLang.word2index) 

# input indexing for tgt
train_tgt_input_index = text2index_tgt(train_data, 'tgt_list', trainLang.word2index) 
val_tgt_input_index = text2index_tgt(val_data, 'tgt_list', trainLang.word2index) 

# get generation label
train_label_symbolindex = text2symbolindex(train_data, 'tgt_list', word2symbolindex)
val_label_symbolindex = text2symbolindex(val_data, 'tgt_list', word2symbolindex)

# get copy label
train_indicator = copy_indicator(train_data, 'src', 'tgt_list')
val_indicator = copy_indicator(val_data, 'src', 'tgt_list')

len(train_src_input_index),len(train_tgt_input_index),len(train_label_symbolindex),len(train_indicator),len(train_data)
len(val_src_input_index),len(val_tgt_input_index),len(val_label_symbolindex),len(val_indicator),len(val_data)


# # Train
import time
import os
import torch.nn as nn
import torch
from torch import optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
# from Data_utils import VocabDataset, vocab_collate_func
# from preprocessing_util import preposs_toekn, Lang, text2index, construct_Lang
from Multilayers_Encoder import EncoderRNN
from Multilayers_Decoder import DecoderAtten, sequence_mask
from config import device, embedding_freeze
import random
from evaluation_pm import similarity_score, check_fact_same, predict_facts, evaluate_prediction
import pickle


def permute_tgt(pm_order, tgt_input, tgt_gen, tgt_copy):
    tgt_input_new = []
    tgt_gen_new = []
    for i in pm_order:
        tgt_input_new.extend(tgt_input[i])
        tgt_gen_new.extend(tgt_gen[i])
    tgt_input_new[-1] = EOS_index
    tgt_gen_new[-1] = EOS_pred_index
    tgt_copy_new = np.concatenate([tgt_copy[i] for i in pm_order], axis=0)
    return np.array(tgt_input_new), np.array(tgt_gen_new), tgt_copy_new

def permute_tgt_batch(pm_order_batch, tgt_input_batch, tgt_gen_batch, 
                      tgt_copy_batch, tgt_len_batch, src_len_batch):
    tgt_input_list, tgt_gen_list, tgt_copy_list = [], [], []
    batch_max_tgt_length = tgt_len_batch.max()
    src_len_batch = src_len_batch.numpy()
    batch_max_src_length = src_len_batch.max()
#     assert(batch_max_src_length==src_len_batch[0])
    for pm_order, tgt_input, tgt_gen, tgt_copy, tgt_len, src_len in zip(
        pm_order_batch, tgt_input_batch, tgt_gen_batch, tgt_copy_batch, tgt_len_batch, src_len_batch):
        tgt_input, tgt_gen, tgt_copy = permute_tgt(pm_order, tgt_input, tgt_gen, tgt_copy)
        padded_vec = np.pad(tgt_input,
                                pad_width=((0, batch_max_tgt_length-tgt_len)),
                                mode="constant", constant_values=PAD_index)
        tgt_input_list.append(padded_vec)
        
        padded_vec = np.pad(tgt_gen,
                                pad_width=((0, batch_max_tgt_length-tgt_len)),
                                mode="constant", constant_values=PAD_pred_index)
        tgt_gen_list.append(padded_vec)
        
        padded_vec = np.pad(tgt_copy,
                            pad_width=((0, batch_max_tgt_length-tgt_len),(0, batch_max_src_length-src_len)),
                            mode="constant", constant_values=0)
        tgt_copy_list.append(padded_vec)
        
    return [torch.from_numpy(np.array(tgt_input_list)), 
            torch.LongTensor(tgt_len_batch),
            torch.from_numpy(np.array(tgt_gen_list)),
            torch.from_numpy(np.array(tgt_copy_list)),
           ]


def train(src_data, tgt_data, encoder, decoder, encoder_optimizer, decoder_optimizer, 
          teacher_forcing_ratio, vocab):
    src_org_batch, src_tensor, src_true_len = src_data
    tgt_org_batch, tgt_tensor, tgt_label_vocab, tgt_label_copy, tgt_true_len = tgt_data
    '''
    finish train for a batch
    '''
    encoder.train()
    decoder.train()
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    batch_size = src_tensor.size(0)
    encoder_hidden, encoder_cell = encoder.initHidden(batch_size)
    loss = 0
    encoder_outputs, encoder_hidden, encoder_cell = encoder(src_tensor, encoder_hidden, src_true_len, encoder_cell)

    decoder_input = torch.tensor([[SOS_index]*batch_size], device=device).transpose(0,1)
    decoder_hidden, decoder_cell = encoder_hidden, decoder.initHidden(batch_size)
    step_log_likelihoods = []
    #print(decoder_hidden.size())
    #print('encoddddddddddder finishhhhhhhhhhhhhhh')
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        ### Teacher forcing: Feed the target as the next input
        decoding_token_index = 0
        tgt_max_len_batch = tgt_true_len.cpu().max().item()
        assert(tgt_max_len_batch==tgt_tensor.size(1))
        while decoding_token_index < tgt_max_len_batch:
            decoder_output, decoder_hidden, decoder_attention, decoder_cell = decoder(
                decoder_input, decoder_hidden, src_true_len, encoder_outputs, decoder_cell)

            decoding_label_vocab = tgt_label_vocab[:, decoding_token_index]
            decoding_label_copy = tgt_label_copy[:, decoding_token_index, :]
            copy_log_probs = decoder_output[:, vocab_pred_size:]+(decoding_label_copy.float()+1e-45).log()
            #mask sample which is copied only
            gen_mask = ((decoding_label_vocab!=OOV_pred_index) | (decoding_label_copy.sum(-1)==0)).float() 
            log_gen_mask = (gen_mask + 1e-45).log().unsqueeze(-1)
            #mask log_prob value for oov_pred_index when label_vocab==oov_pred_index and is copied 
            generation_log_probs = decoder_output.gather(1, decoding_label_vocab.unsqueeze(1)) + log_gen_mask
            combined_gen_and_copy = torch.cat((generation_log_probs, copy_log_probs), dim=-1)
            step_log_likelihood = torch.logsumexp(combined_gen_and_copy, dim=-1)
            step_log_likelihoods.append(step_log_likelihood.unsqueeze(1))
            #loss += criterion(decoder_output, tgt_tensor[:,decoding_token_index])
            decoder_input = tgt_tensor[:,decoding_token_index].unsqueeze(1)  # Teacher forcing
            decoding_token_index += 1

    else:
        ### Without teacher forcing: use its own predictions as the next input
        decoding_token_index = 0
        tgt_max_len_batch = tgt_true_len.cpu().max().item()
        assert(tgt_max_len_batch==tgt_tensor.size(1))
        while decoding_token_index < tgt_max_len_batch:
            decoder_output, decoder_hidden, decoder_attention_weights, decoder_cell = decoder(
                decoder_input, decoder_hidden, src_true_len, encoder_outputs, decoder_cell)

            decoding_label_vocab = tgt_label_vocab[:, decoding_token_index]
            decoding_label_copy = tgt_label_copy[:, decoding_token_index, :]
            copy_log_probs = decoder_output[:, vocab_pred_size:]+(decoding_label_copy.float()+1e-45).log()
            #mask sample which is copied only
            gen_mask = ((decoding_label_vocab!=OOV_pred_index)|(decoding_label_copy.sum(-1)==0)).float() 
            log_gen_mask = (gen_mask + 1e-45).log().unsqueeze(-1)
            #mask log_prob value for oov_pred_index when label_vocab==oov_pred_index and is copied 
            generation_log_probs = decoder_output.gather(1, decoding_label_vocab.unsqueeze(1)) + log_gen_mask
            combined_gen_and_copy = torch.cat((generation_log_probs, copy_log_probs), dim=-1)
            step_log_likelihood = torch.logsumexp(combined_gen_and_copy, dim=-1)
            step_log_likelihoods.append(step_log_likelihood.unsqueeze(1))

            topv, topi = decoder_output.topk(1, dim=-1)
            next_input = topi.detach().cpu().squeeze(1)
            decoder_input = []
            for i_batch in range(batch_size):
                pred_list = vocab_pred+src_org_batch[i_batch]
                next_input_token = pred_list[next_input[i_batch].item()]
                decoder_input.append(vocab.word2index.get(next_input_token, UNK_index))
            decoder_input = torch.tensor(decoder_input, device=device).unsqueeze(1)
            decoding_token_index += 1

    # average loss
    log_likelihoods = torch.cat(step_log_likelihoods, dim=-1)
    # mask padding for tgt
    tgt_pad_mask = sequence_mask(tgt_true_len).float()
    log_likelihoods = log_likelihoods*tgt_pad_mask
    loss = -log_likelihoods.sum()/batch_size
    loss.backward()

    ### TODO
    # clip for gradient exploding 
    encoder_optimizer.step()
    decoder_optimizer.step()

    return (loss*batch_size/tgt_pad_mask.sum()).item() #torch.div(loss, tgt_true_len.type_as(loss).mean()).item()  #/tgt_true_len.mean()

def pm_random(tgt_fact_num):
    return [np.random.permutation(i) for i in tgt_fact_num]

def pm_last3(tgt_fact_num):
    pm_list = []
    for i in tgt_fact_num:
        order = list(range(i))
        if i>3:
            pm_list.append(order[-3:]+order[:-3])
        elif i==3:
            pm_list.append(order[-2:]+order[:-2])
        elif i==2:
            pm_list.append(order[::-1])
        else:
            pm_list.append(order)
    return pm_list

def trainIters(train_loader, val_loader, encoder, decoder, num_epochs, learning_rate, 
               teacher_forcing_ratio, tfr_decay_rate, model_save_info, tgt_max_len, 
               beam_size, vocab):

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    if model_save_info['model_path_for_resume'] is not None:
        check_point_state = torch.load(model_save_info['model_path_for_resume'])
        encoder.load_state_dict(check_point_state['encoder_state_dict'])
        encoder_optimizer.load_state_dict(check_point_state['encoder_optimizer_state_dict'])
        decoder.load_state_dict(check_point_state['decoder_state_dict'])
        decoder_optimizer.load_state_dict(check_point_state['decoder_optimizer_state_dict'])

    for epoch in range(num_epochs): 
        start_time = time.time()
        n_iter = -1
        losses = np.zeros((len(train_loader),))
        if tfr_decay_rate is not None:
            teacher_forcing_ratio *= tfr_decay_rate
        for src_tensor, src_true_len, tgt_batch, tgt_true_len, tgt_gen_batch, tgt_copy_batch, src_org_batch, tgt_org_batch in train_loader:
            tgt_fact_num = [len(item) for item in tgt_batch]
            #pm_order_batch = [list(range(i)) for i in tgt_fact_num]
            pm_order_batch = pm_last3(tgt_fact_num)
            #pm_order_batch = [list(range(i-1,-1,-1)) for i in tgt_fact_num]
            tgt_tensor, tgt_true_len, tgt_label_vocab, tgt_label_copy = permute_tgt_batch(
                pm_order_batch, tgt_batch, tgt_gen_batch, tgt_copy_batch, tgt_true_len, src_true_len)
            
            src_tensor, src_true_len = src_tensor.to(device), src_true_len.to(device)
            tgt_tensor, tgt_true_len = tgt_tensor.to(device), tgt_true_len.to(device)
            tgt_label_vocab, tgt_label_copy = tgt_label_vocab.to(device), tgt_label_copy.to(device)
            
            n_iter += 1
            #print('start_step: ', n_iter)
            src_data = (src_org_batch, src_tensor, src_true_len)
            tgt_data = (tgt_org_batch, tgt_tensor, tgt_label_vocab, tgt_label_copy, tgt_true_len)
            loss = train(src_data, tgt_data, encoder, decoder, encoder_optimizer, 
                         decoder_optimizer, teacher_forcing_ratio, vocab)
            losses[n_iter] = loss
            if n_iter % 500 == 0:
                pass
        val_loss, src_org, tgt_org, tgt_pred = predict_facts(val_loader, encoder, decoder, tgt_max_len, vocab)
        precision, recall, F_scores, matched_num, org_num, pred_num = evaluate_prediction(tgt_org, tgt_pred)
        precision_2 = matched_num.sum()/pred_num.sum()
        recall_2 = matched_num.sum()/org_num.sum()
        Fscore_2 = 2*precision_2*recall_2/(precision_2+recall_2+1e-10)
        epoch_train_time = (time.time()-start_time)/60
        print_str = 'epoch: [{0:d}/{1:d}]({2:.2f}m), step: [{3:d}/{4:d}], train_loss:{5:.6f}, val_precision: {6:.6f}/{7:.6f}, val_recall: {8:.6f}/{9:.6f}, val_Fscore: {10:.6f}/{11:.6f}, val_loss: {12:.6f}'.format(
            epoch, num_epochs, epoch_train_time, n_iter, len(train_loader), losses.mean(), precision.mean(), precision_2, 
            recall.mean(), recall_2, F_scores.mean(), Fscore_2,val_loss)
        print_info.append(print_str)
        print(print_str)

        if (epoch+1) % model_save_info['epochs_per_save_model'] == 0:
            check_point_state = {
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'decoder_optimizer_state_dict': decoder_optimizer.state_dict()
                }
            torch.save(check_point_state, '{}epoch_{}.pth'.format(model_save_info['model_path'], epoch))

    return None

paras = dict( 
    tgt_max_len = 220,
    max_src_len_dataloader =150, 
    max_tgt_len_dataloader =200,

    emb_size = 200,
    en_hidden_size = 128,
    en_num_layers = 2,
    en_num_direction = 2,
    de_hidden_size = 256,
    de_num_layers = 3,
    rnn_type = 'GRU', # {LSTM, GRU}
    attention_type = 'dot_prod', #'dot_prod', general, concat #dot-product need pre-process
    teacher_forcing_ratio = 1,
    tfr_decay_rate = None, #'None means no decay'

    learning_rate = 1e-3,
    num_epochs = 20,
    batch_size = 64, 
    beam_size = 5,
    dropout_rate = 0.0,

    model_save_info = dict(
        model_path = 'nmt_models/last3_order2/',
        epochs_per_save_model = 1,
        model_path_for_resume = None #'nmt_models/epoch_0.pth'
        )
    )

tgt_max_len = paras['tgt_max_len']
max_src_len_dataloader = paras['max_src_len_dataloader']
max_tgt_len_dataloader = paras['max_tgt_len_dataloader']

teacher_forcing_ratio = paras['teacher_forcing_ratio']
tfr_decay_rate = paras['tfr_decay_rate']
emb_size = paras['emb_size']
en_hidden_size = paras['en_hidden_size']
en_num_layers = paras['en_num_layers']
en_num_direction = paras['en_num_direction']
de_hidden_size = paras['de_hidden_size']
de_num_layers = paras['de_num_layers']

learning_rate = paras['learning_rate']
num_epochs = paras['num_epochs']
batch_size = paras['batch_size']
rnn_type = paras['rnn_type']
attention_type = paras['attention_type']
beam_size = paras['beam_size']
model_save_info = paras['model_save_info']
dropout_rate = paras['dropout_rate']


train_dataset = VocabDataset(train_src_input_index, train_tgt_input_index, 
                             train_label_symbolindex, train_indicator, train_data, 
                             max_src_len_dataloader, max_tgt_len_dataloader)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               collate_fn=vocab_collate_func,
                                               shuffle=True)

val_dataset = VocabDataset(val_src_input_index, val_tgt_input_index, 
                           val_label_symbolindex, val_indicator, val_data,
                           max_src_len_dataloader, max_tgt_len_dataloader)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=batch_size,
                                               collate_fn=vocab_collate_func,
                                               shuffle=False)


# make dir for saving models
if not os.path.exists(model_save_info['model_path']):
    os.makedirs(model_save_info['model_path'])
### save model hyperparameters
with open(model_save_info['model_path']+'model_params.pkl', 'wb') as f:
    model_hyparams = paras
    pickle.dump(model_hyparams, f)
print(model_hyparams)

encoder = EncoderRNN(trainLang.vocab_size, emb_size, en_hidden_size, en_num_layers, 
                     en_num_direction, (de_num_layers, de_hidden_size), rnn_type=rnn_type, 
                     dropout_rate=dropout_rate)
decoder = DecoderAtten(trainLang.vocab_size, vocab_pred_size, emb_size, de_hidden_size, 
                       de_num_layers, (en_num_layers, en_num_direction, en_hidden_size), 
                       rnn_type=rnn_type, atten_type=attention_type, 
                       dropout_rate=dropout_rate)

encoder, decoder = encoder.to(device), decoder.to(device)
print('Encoder:')
print(encoder)
print('Decoder:')
print(decoder)

print_info = []
trainIters(train_loader, val_loader, encoder, decoder, num_epochs, learning_rate, 
           teacher_forcing_ratio, tfr_decay_rate, model_save_info, tgt_max_len, 
           beam_size, trainLang)

with open(model_save_info['model_path']+'print_info.pkl', 'wb') as f:
    pickle.dump('\n'.join(print_info), f)