import json
import pandas as pd
import time
import numpy as np 
from sklearn.model_selection import train_test_split
import jieba
import re
import os
import torch.nn as nn
import torch
from torch import optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from config import vocab_pred, vocab_pred_size, vocab_prefix
from config import UNK_index, PAD_index, SOS_index, EOS_index 
from config import OOV_pred_index, PAD_pred_index, EOS_pred_index

# from Data_utils import VocabDataset, vocab_collate_func
from preprocessing_util import load_preprocess_data, construct_Lang, text2symbolindex, text2index, copy_indicator
from Multilayers_Encoder import EncoderRNN
from Multilayers_Decoder import DecoderAtten, sequence_mask
from config import device, embedding_freeze
import random
from evaluation import similarity_score, check_fact_same, predict_facts, evaluate_prediction
import pickle

####################Define Global Variable#########################

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

####################Define Global Variable#########################




def start_train(transtype, paras):
    src_max_vocab_size = paras['src_max_vocab_size']
    tgt_max_vocab_size = paras['tgt_max_vocab_size']
    tgt_max_len = paras['tgt_max_len']
    max_src_len_dataloader = paras['max_src_len_dataloader']
    max_tgt_len_dataloader = paras['max_tgt_len_dataloader']

    teacher_forcing_ratio = paras['teacher_forcing_ratio']
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

    address_book=dict(
        train_src = 'Machine_Translation_NLP/iwsltzhen/iwslt-{}-{}/train.tok.{}'.format(transtype[0], transtype[1], transtype[0]),
        train_tgt = 'Machine_Translation_NLP/iwsltzhen/iwslt-{}-{}/train.tok.{}'.format(transtype[0], transtype[1], transtype[1]),
        val_src = 'Machine_Translation_NLP/iwsltzhen/iwslt-{}-{}/dev.tok.{}'.format(transtype[0], transtype[1], transtype[0]),
        val_tgt = 'Machine_Translation_NLP/iwsltzhen/iwslt-{}-{}/dev.tok.{}'.format(transtype[0], transtype[1], transtype[1]),
        src_emb = 'embedding/wiki.{}.vec'.format(transtype[0]),
        tgt_emb = 'embedding/wiki.{}.vec'.format(transtype[1])
        )
    #print(address_book)
    train_src_add = address_book['train_src']
    train_tgt_add = address_book['train_tgt']
    val_src_add = address_book['val_src']
    val_tgt_add = address_book['val_tgt']
	
    # make dir for saving models
    if not os.path.exists(model_save_info['model_path']):
        os.makedirs(model_save_info['model_path'])
    ### save model hyperparameters
    with open(model_save_info['model_path']+'model_params.pkl', 'wb') as f:
        model_hyparams = paras
        model_hyparams['address_book'] = address_book
        pickle.dump(model_hyparams, f)
    print(model_hyparams)

    # read all data
    train_src = []
    with open(train_src_add) as f:
        for line in f:
            train_src.append(preposs_toekn(line[:-1].strip().split(' ')))

    train_tgt = []
    with open(train_tgt_add) as f:
        for line in f:
            train_tgt.append(preposs_toekn(line[:-1].strip().split(' ')))
        
    val_src = []
    with open(val_src_add) as f:
        for line in f:
            val_src.append(preposs_toekn(line[:-1].strip().split(' ')))

    val_tgt = []
    with open(val_tgt_add) as f:
        for line in f:
            val_tgt.append(preposs_toekn(line[:-1].strip().split(' ')))


    print('The number of train samples: ', len(train_src))
    print('The number of val samples: ', len(val_src))
    # build a common vocab both for src and tgt 
    srcLang = construct_Lang('src', src_max_vocab_size, address_book['src_emb'], train_src)
    tgtLang = construct_Lang('tgt', tgt_max_vocab_size, address_book['tgt_emb'], train_tgt)
    train_input_index = text2index(train_src, srcLang.word2index) #add EOS token here 
    train_output_index = text2index(train_tgt, tgtLang.word2index)
    val_input_index = text2index(val_src, srcLang.word2index)
    val_output_index = text2index(val_tgt, tgtLang.word2index)
    ### save srcLang and tgtLang

    #for src; keep original src_org and index based on vocab src_tensor

    #for tgt; vocab_pred_label, copy_label

    train_dataset = VocabDataset(train_input_index,train_output_index, max_src_len_dataloader, max_tgt_len_dataloader)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               collate_fn=vocab_collate_func,
                                               shuffle=True)

    val_dataset = VocabDataset(val_input_index,val_output_index, None, None)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=batch_size,
                                            collate_fn=vocab_collate_func,
                                            shuffle=False)

    # test_dataset = VocabDataset(test_data)
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
    #                                            batch_size=BATCH_SIZE,
    #                                            collate_fn=vocab_collate_func,
    #                                            shuffle=False)

    embedding_src_weight = torch.from_numpy(srcLang.embedding_matrix).type(torch.FloatTensor).to(device)
    embedding_tgt_weight = torch.from_numpy(tgtLang.embedding_matrix).type(torch.FloatTensor).to(device)
    print(embedding_src_weight.size(), embedding_tgt_weight.size())

    encoder = EncoderRNN(srcLang.vocab_size, emb_size, en_hidden_size, en_num_layers, en_num_direction, (de_num_layers, de_hidden_size), rnn_type=rnn_type, embedding_weight=embedding_src_weight, dropout_rate=dropout_rate)
    decoder = DecoderAtten(tgtLang.vocab_size, emb_size, de_hidden_size, de_num_layers, (en_num_layers, en_num_direction, en_hidden_size), rnn_type=rnn_type, embedding_weight=embedding_tgt_weight, atten_type=attention_type, dropout_rate=dropout_rate)

    
    encoder, decoder = encoder.to(device), decoder.to(device)
    print('Encoder:')
    print(encoder)
    print('Decoder:')
    print(decoder)
    trainIters(train_loader, val_loader, encoder, decoder, num_epochs, learning_rate, teacher_forcing_ratio, srcLang, tgtLang, model_save_info, tgt_max_len, beam_size)
    

if __name__ == "__main__":
    transtype = ('vi', 'en')
    paras = dict( 
        src_max_vocab_size = 26109, #47127, #26109, zh, vi
        tgt_max_vocab_size = 24418, #31553, #24418,
        tgt_max_len = 100,
        max_src_len_dataloader =72, #67, #72, 
        max_tgt_len_dataloader = 71, #72, #71, 

        emb_size = 300,
        en_hidden_size = 256,
        en_num_layers = 2,
        en_num_direction = 2,
        de_hidden_size = 128,
        de_num_layers = 3,
        #deal_bi = 'linear', #'linear', #{'linear', 'sum'} #linear layer en-hid to de_hid
        rnn_type = 'GRU', # {LSTM, GRU}
        attention_type = 'dot_prod', #'dot_prod', general, concat #dot-product need pre-process
        teacher_forcing_ratio = 0.5,

        learning_rate = 1e-3,
        num_epochs = 22,
        batch_size = 128, 
        beam_size = 5,
        dropout_rate = 0.1,

        model_save_info = dict(
            model_path = 'nmt_models/vi-en-1/',
            epochs_per_save_model = 1,
            model_path_for_resume = None #'nmt_models/epoch_0.pth'
            )
        )
    #print('paras: ', paras)
    start_train(transtype, paras)



