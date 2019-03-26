import numpy as np
import time
import os
import torch.nn as nn
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from Data_utils import VocabDataset, vocab_collate_func
from preprocessing_util import preposs_toekn, Lang, text2index, construct_Lang
from Multilayers_Encoder import EncoderRNN
from Multilayers_Decoder import DecoderRNN, DecoderAtten
from config import device, PAD_token, SOS_token, EOS_token, UNK_token, embedding_freeze, vocab_prefix
import random
from evaluation import evaluate_batch, evaluate_beam_batch
import pickle 

####################Define Global Variable#########################

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

####################Define Global Variable#########################

def train(src_data, tgt_data, encoder, decoder, encoder_optimizer, decoder_optimizer, teacher_forcing_ratio):
    src_org_batch, src_tensor, src_true_len = src_data
    tgt_org_batch, tgt_tensor, tgt_label_vocab, tgt_label_copy, tgt_true_len = tgt_data
    '''
    finish train for a batch
    '''
    batch_size = src_tensor.size(0)
    encoder_hidden, encoder_cell = encoder.initHidden(batch_size)
    
    encoder.train()
    decoder.train()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0
    encoder_outputs, encoder_hidden, encoder_cell = encoder(src_tensor, encoder_hidden, src_true_len, encoder_cell)

    decoder_input = torch.tensor([[SOS_token]*batch_size], device=device).transpose(0,1)
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
            copy_log_probs = decoder_output[:, vocab_size_pred:]+(decoding_label_copy.float()+1e-45).log()
            #mask sample which is copied only
            gen_mask = ((decoding_label_vocab!=oov_pred_index) | (decoding_label_copy.sum(-1)==0)).float() 
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
            copy_log_probs = decoder_output[:, vocab_size_pred:]+(decoding_label_copy.float()+1e-45).log()
            #mask sample which is copied only
            gen_mask = ((decoding_label_vocab!=oov_pred_index) | (decoding_label_copy.sum(-1)==0)).float() 
            log_gen_mask = (gen_mask + 1e-45).log().unsqueeze(-1)
            #mask log_prob value for oov_pred_index when label_vocab==oov_pred_index and is copied 
            generation_log_probs = decoder_output.gather(1, decoding_label_vocab.unsqueeze(1)) + log_gen_mask
            combined_gen_and_copy = torch.cat((generation_log_probs, copy_log_probs), dim=-1)
            step_log_likelihood = torch.logsumexp(combined_gen_and_copy, dim=-1)
            step_log_likelihoods.append(step_log_likelihood.unsqueeze(1))

            topv, topi = copy_log_probs.topk(1, dim=-1)
            next_input = topi.detach().cpu().squeeze(1)
            decoder_input = []
            for i_batch in range(batch_size):
                pred_list = vocab_pred+src_org_list[i_batch]
                next_input_token = pred_list[next_input[i_batch].item()]
                decoder_input.append(??get_index_vocab(next_input_token))
            decoder_input = torch.tensor(decoder_input, device=device).unsqueeze(1)
            #loss += criterion(decoder_output, tgt_tensor[:,decoding_token_index])
            #topv, topi = decoder_output.topk(1)
            #decoder_input = topi.detach()  # detach from history as input
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


def trainIters(train_loader, val_loader, encoder, decoder, num_epochs, 
               learning_rate, teacher_forcing_ratio, srcLang, tgtLang, model_save_info, tgt_max_len, beam_size):

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    if model_save_info['model_path_for_resume'] is not None:
        check_point_state = torch.load(model_save_info['model_path_for_resume'])
        encoder.load_state_dict(check_point_state['encoder_state_dict'])
        encoder_optimizer.load_state_dict(check_point_state['encoder_optimizer_state_dict'])
        decoder.load_state_dict(check_point_state['decoder_state_dict'])
        decoder_optimizer.load_state_dict(check_point_state['decoder_optimizer_state_dict'])

    #criterion = nn.NLLLoss() #nn.NLLLoss(ignore_index=PAD_token)
    max_val_bleu = 0

    for epoch in range(num_epochs): 
        n_iter = -1
        start_time = time.time()
        for src_org_batch, src_tensor, src_true_len, tgt_org_batch, tgt_tensor, tgt_label_vocab, tgt_label_copy, tgt_true_len in train_loader:
            n_iter += 1
            #print('start_step: ', n_iter)
            src_data = (src_org_batch, src_tensor, src_true_len)
            tgt_data = (tgt_org_batch, tgt_tensor, tgt_label_vocab, tgt_label_copy, tgt_true_len)
            loss = train(src_data, tgt_data, encoder, decoder, encoder_optimizer, decoder_optimizer, teacher_forcing_ratio)
            if n_iter % 500 == 0:
                #print('Loss:', loss)
                #eva_start = time.time()
                val_bleu_sacre, val_bleu_nltk, val_loss = evaluate_batch(val_loader, encoder, decoder, criterion, tgt_max_len, tgtLang.index2word, srcLang.index2word)
                #print((time.time()-eva_start)/60)
                print('epoch: [{}/{}], step: [{}/{}], train_loss:{}, val_bleu_sacre: {}, val_bleu_nltk: {}, val_loss: {}'.format(
                    epoch, num_epochs, n_iter, len(train_loader), loss, val_bleu_sacre[0], val_bleu_nltk, val_loss))
               # print('Decoder parameters grad:')
               # for p in decoder.named_parameters():
               #     print(p[0], ': ',  p[1].grad.data.abs().mean().item(), p[1].grad.data.abs().max().item(), p[1].data.abs().mean().item(), p[1].data.abs().max().item(), end=' ')
               # print('\n')
               # print('Encoder Parameters grad:')
               # for p in encoder.named_parameters():
               #     print(p[0], ': ',  p[1].grad.data.abs().mean().item(), p[1].grad.data.abs().max().item(), p[1].data.abs().mean().item(), p[1].data.abs().max().item(), end=' ')
               # print('\n')
        val_bleu_sacre, val_bleu_nltk, val_loss = evaluate_batch(val_loader, encoder, decoder, criterion, tgt_max_len, tgtLang.index2word, srcLang.index2word)
        print('epoch: [{}/{}] (Running time {:.3f} min), val_bleu_sacre: {}, val_bleu_nltk: {}, val_loss: {}'.format(epoch, num_epochs, (time.time()-start_time)/60, val_bleu_sacre, val_bleu_nltk, val_loss))
        #val_bleu_sacre_beam, _, _ = evaluate_beam_batch(beam_size, val_loader, encoder, decoder, criterion, tgt_max_len, tgtLang.index2word)
        #print('epoch: [{}/{}] (Running time {:.3f} min), val_bleu_sacre_beam: {}'.format(epoch, num_epochs, (time.time()-start_time)/60, val_bleu_sacre_beam))
        if max_val_bleu < val_bleu_sacre.score:
            max_val_bleu = val_bleu_sacre.score
            ### TODO save best model
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



