import numpy as np
from config import *
from collections import Counter
import re
import json
import jieba

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
        sample_processed['src_org'] = sample['natural']
        #sample_processed['src'] = list(jieba.cut(sample['natural'], cut_all=False))
        sample_processed['src'] = character_segmentation(sample['natural'])
        
        # transform fact list into str and tokenize
        # $ separates facts; @ separate elements for one fact; & separate objects for one fact
        sample_processed['tgt_org'] = sample['logic']
        logic_list = []
        for fact in sample['logic']:
            logic_list.append('@'.join([fact['subject'], replaceMisspred(fact['predicate']), 
                                       '&'.join(fact['object'])]))
        logic_str = '$'.join(logic_list)
        sample_processed['tgt'] = character_segmentation(logic_str)
        #list(jieba.cut(logic_str, cut_all=False))

        data.append(sample_processed)
    return data

    def text2index(data, key, word2index):
    '''
    transform tokens into index as input for both src and tgt
    '''
    indexdata = []
    for line in data:
        line = line[key]
        indexdata.append([word2index[c] if c in word2index.keys() else UNK_index for c in line])
        #indexdata[-1].append(EOS_index)
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
    for line in data:
        line = line[key]
        indexdata.append([word2index[c] if c in word2index.keys() else OOV_pred_index for c in line])
        #indexdata[-1].append(EOS_index)
    print('symbol label finish')
    return indexdata

def copy_indicator(data, src_key='src', tgt_key='tgt'):
    '''get copy label for tgt
    '''
    indicator = []
    for sample in data:
        tgt = sample[tgt_key]
        src = sample[src_key]
        matrix = np.zeros((len(tgt), len(src)), dtype=int)
        for m in range(len(tgt)):
            for n in range(len(src)):
                if tgt[m] == src[n]:
                    matrix[m,n] = 1
        indicator.append(matrix)
    return indicator
