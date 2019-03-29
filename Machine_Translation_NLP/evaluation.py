# config:
import torch
import numpy as np
from nltk.translate import bleu_score 
from config import SOS_token, UNK_token, EOS_token, PAD_token, oov_pred_index, vocab_pred
import beam
import difflib
from Multilayers_Decoder import sequence_mask

def evaluate_batch(loader, encoder, decoder, tgt_max_length, vocab, vocab_pred_size):
    """
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    encoder.eval()
    decoder.eval()

    tgt_pred = []
    src_org = []
    tgt_org = []
    loss = 0

    for src_tensor, src_true_len, tgt_tensor, tgt_true_len, tgt_label_vocab, tgt_label_copy, src_org_batch, tgt_org_batch in loader:
        batch_size = src_tensor.size(0)
        encoder_hidden, encoder_cell = encoder.initHidden(batch_size)
        encoder_outputs, encoder_hidden, encoder_cell = encoder(src_tensor, encoder_hidden, src_true_len, encoder_cell)
        decoder_input = torch.tensor([SOS_token]*batch_size, device=device).unsqueeze(1)
        decoder_hidden, decoder_cell = encoder_hidden, decoder.initHidden(batch_size)

        decoding_token_index = 0
        step_log_likelihoods = []
        tgt_pred_batch = [[]]*batch_size
        tgt_true_len_max = tgt_true_len.cpu().numpy().max()
        #sent_not_end_index = list(range(batch_size))
        while decoding_token_index < tgt_max_length:
            decoder_output, decoder_hidden, _, decoder_cell = decoder(decoder_input, decoder_hidden, src_true_len, encoder_outputs, decoder_cell)
            
            # compute loss 
            if decoding_token_index < tgt_true_len_max:
                decoding_label_vocab = tgt_label_vocab[:, decoding_token_index]
                decoding_label_copy = tgt_label_copy[:, decoding_token_index, :]
                copy_log_probs = decoder_output[:, vocab_pred_size:]+(decoding_label_copy.float()+1e-45).log()
                #mask sample which is copied only
                gen_mask = ((decoding_label_vocab!=oov_pred_index) | (decoding_label_copy.sum(-1)==0)).float() 
                log_gen_mask = (gen_mask + 1e-45).log().unsqueeze(-1)
                #mask log_prob value for oov_pred_index when label_vocab==oov_pred_index and is copied 
                generation_log_probs = decoder_output.gather(1, decoding_label_vocab.unsqueeze(1)) + log_gen_mask
                combined_gen_and_copy = torch.cat((generation_log_probs, copy_log_probs), dim=-1)
                step_log_likelihood = torch.logsumexp(combined_gen_and_copy, dim=-1)
                step_log_likelihoods.append(step_log_likelihood.unsqueeze(1))

            #
            topv, topi = copy_log_probs.topk(1, dim=-1)
            next_input = topi.detach().cpu().squeeze(1)
            decoder_input = []
            stop_flag = [False]*batch_size
            for i_batch in range(batch_size):
                pred_list = vocab_pred+src_org_batch[i_batch]
                next_input_token = pred_list[next_input[i_batch].item()]
                if next_input_token == vocab_pred[EOS_token]:
                    stop_flag[i_batch] = True
                tgt_pred_batch[i_batch].append(next_input_token)
                decoder_input.append(vocab.word2index.get(next_input_token, UNK_token))
            decoder_input = torch.tensor(decoder_input, device=device).unsqueeze(1)
            decoding_token_index += 1
            if all(stop_flag):
                break
        log_likelihoods = torch.cat(step_log_likelihoods, dim=-1)
        # mask padding for tgt
        tgt_pad_mask = sequence_mask(tgt_true_len).float()
        log_likelihoods = log_likelihoods*tgt_pad_mask[:,:log_likelihoods.size(1)]g
        loss += -(log_likelihoods.sum()/tgt_pad_mask.sum()).item()
        tgt_pred.extend(tgt_pred_batch)
        src_org.extend(src_org_batch)
        tgt_org.extend(tgt_org_batch)
    eval_len = len(tgt_pred)
    precision = np.zeros((eval_len,))
    recall = np.zeros((eval_len,))
    for i in range(eval_len):
        org_facts = tgt_org[i].split('$')
        pred_facts = tgt_pred[i].split('$')
        org_facts_num = len(org_facts)
        pred_facts_num = len(pred_facts)
        org_match_num = np.zeros((org_facts_num))
        pred_match_num = np.zeros((pred_facts_num))
        for org_i in enumerate(org_facts):
            for pred_i in enumerate(pred_facts):
                org_fact = org_facts[org_i]
                pred_fact = pred_facts[pred_i]
                org_fact_ele = org_fact.split('@')
                pred_fact_ele = pred_fact.split('@')
                if len(org_fact_ele) == len(pred_fact_ele):
                    ele_num = len(org_fact_ele)
                    if difflib.SequenceMatcher(None,org_fact,pred_fact).ratio() > 0.85:
                        org_match_num[org_i] += 1
                        pred_match_num[pred_i] += 1
                        break
                    ele_sim = np.zeros((ele_num,))
                    for ele_i in range(ele_num):
                        ele_sim[ele_i] = difflib.SequenceMatcher(None,org_fact_ele[ele_i],pred_fact_ele[ele_i]).ratio()
                    if ele_sim.mean() > 0.85:
                        org_match_num[org_i] += 1
                        pred_match_num[pred_i] += 1
        precision[i] = pred_match_num.mean()
        recall[i] = org_match_num.mean()
    if True:
        random_sample = np.random.randint(eval_len)
        print('src:', src_org[random_sample])
        print('Ref: ', tgt_org[random_sample])
        print('pred: ', tgt_pred[random_sample])
    return precision, recall, loss


def evaluate_beam_batch(beam_size, loader, encoder, decoder, criterion, tgt_max_length, tgt_idx2words):
    """
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.eval()
    decoder.eval()

    have_cell = True
    if encoder.get_rnn_type == 'GRU':
        have_cell = False
    de_hidden_size = decoder.hidden_size
    de_num_layers = decoder.num_layers
    #have_cell = True
    #loss_all = []
    #tgt_sents_nltk = []
    tgt_sents_sacre = []
    #tgt_pred_sents_nltk = []
    tgt_pred_sents_sacre = []
    with torch.no_grad():
        for input_tensor, input_lengths, target_tensor, target_lengths in loader:
            batch_size = input_tensor.size(0) #int 
            encoder_hidden,encoder_cell = encoder.initHidden(batch_size)
            encoder_outputs,encoder_hidden,encoder_cell = encoder(input_tensor, encoder_hidden, input_lengths, encoder_cell)
            decoder_hidden, decoder_cell = encoder_hidden, decoder.initHidden(batch_size)

            beamers = [beam.beam(beam_size, min_length=0, n_best=1) for i in range(batch_size)]
            encoder_max_len, en_output_hz= encoder_outputs.size(1), encoder_outputs.size(2)
            assert(encoder_max_len == input_lengths.max())
            assert(en_output_hz == encoder.hidden_size*encoder.num_direction)
            decoder_hidden = decoder_hidden.unsqueeze(2).expand(de_num_layers, batch_size, beam_size, de_hidden_size).contiguous().view(de_num_layers, batch_size*beam_size, de_hidden_size)
            if have_cell:
                decoder_cell = decoder_cell.unsqueeze(2).expand(de_num_layers, batch_size, beam_size, de_hidden_size).contiguous().view(de_num_layers, batch_size*beam_size, de_hidden_size)
            input_lengths_beam = input_lengths.unsqueeze(1).expand(batch_size, beam_size).contiguous().view(batch_size*beam_size)
            encoder_outputs_beam = encoder_outputs.unsqueeze(1).expand(batch_size, beam_size, encoder_max_len, en_output_hz).contiguous().view(batch_size*beam_size, 
                encoder_max_len, en_output_hz)

            #loss = 0
            for decoding_token_index in range(tgt_max_length):
                decoder_input = torch.stack([beamer.next_ts[-1] for beamer in beamers], dim=0).unsqueeze(-1).view(batch_size*beam_size, 1).to(device)
                decoder_output, decoder_hidden, _, decoder_cell = decoder(decoder_input, decoder_hidden, input_lengths_beam, encoder_outputs_beam, decoder_cell)
                vocab_size = decoder_output.size(1)
                decoder_output_beam, decoder_hidden_beam = decoder_output.view(batch_size, beam_size, vocab_size), decoder_hidden.view(de_num_layers, batch_size, beam_size, de_hidden_size)
                if have_cell:
                    decoder_cell_beam = decoder_cell.view(de_num_layers, batch_size, beam_size, de_hidden_size)
                decoder_input_list = []
                decoder_hidden_list = []
                decoder_cell_list = []
                flag_stop = True
                for i_batch in range(batch_size):
                    beamer = beamers[i_batch]
                    if beamer.stopByEOS == False:
                        beamer.advance(decoder_output_beam[i_batch])
                        decoder_hidden_list.append(decoder_hidden_beam[:, i_batch, :, :].index_select(dim=1,index=beamer.prev_ps[-1]))
                        if have_cell:
                            decoder_cell_list.append(decoder_cell_beam[:, i_batch, :, :].index_select(dim=1,index=beamer.prev_ps[-1]))
                        decoder_input_list.append(beamer.next_ts[-1])
                        flag_stop = False
                    else:
                        decoder_hidden_list.append(decoder_hidden_beam[:,i_batch,:,:])
                        if have_cell:
                            decoder_cell_list.append(decoder_cell_beam[:,i_batch,:,:])
                        decoder_input_list.append(torch.LongTensor(beam_size).fill_(PAD_token).to(device))
                if flag_stop:
                    break
                decoder_input = torch.stack(decoder_input_list, 0).view(batch_size*beam_size, 1)
                decoder_hidden = torch.stack(decoder_hidden_list, 1).view(de_num_layers, batch_size*beam_size, de_hidden_size)
                if have_cell:
                    decoder_cell = torch.stack(decoder_cell_list, 1).view(de_num_layers, batch_size*beam_size, de_hidden_size)

            target_tensor_numpy = target_tensor.cpu().numpy()
            for i_batch in range(batch_size):
                beamer = beamers[i_batch]
                paths_sort = sorted(beamer.finish_paths, key=lambda x: x[0], reverse=True)
                if len(paths_sort) == 0:
                    best_path = (beamer.scores[0], len(beamer.prev_ps), 0)
                else:
                    best_path = paths_sort[0]
                score_best_path, tokens_best_path = beamer.get_pred_sentence(best_path)
                # ground true
                tgt_sent_tokens = fun_index2token(target_tensor_numpy[i_batch].tolist(), tgt_idx2words)
                tgt_sents_sacre.append(' '.join(tgt_sent_tokens))
                # prediction
                tgt_pred_sent_tokens = fun_index2token(tokens_best_path, tgt_idx2words)
                tgt_pred_sents_sacre.append(' '.join(tgt_pred_sent_tokens))

    #nltk_bleu_score = bleu_score.corpus_bleu(tgt_sents_nltk, tgt_pred_sents_nltk)
    sacre_bleu_score = sacrebleu.corpus_bleu(tgt_pred_sents_sacre, [tgt_sents_sacre], smooth='exp', smooth_floor=0.0, force=False, lowercase=False,
        tokenize='none', use_effective_order=True)
    if True:
        random_sample = 300 #np.random.randint(len(tgt_pred_sents_sacre))
        print('Ref: ', tgt_sents_sacre[random_sample])
        print('pred: ', tgt_pred_sents_sacre[random_sample])
    return sacre_bleu_score, None, None


def evaluate_single(data, encoder, decoder, criterion, tgt_max_length, tgt_idx2words, src_idx2words):
    """
    """
    input_tensor, input_lengths, target_tensor, target_lengths = data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    loss_all = []
    #tgt_sents_nltk = []
    src_sents = []
    tgt_sents_sacre = []
    #tgt_pred_sents_nltk = []
    tgt_pred_sents_sacre = []
    with torch.no_grad():
        batch_size = input_tensor.size(0)
        encoder_hidden = encoder.initHidden(batch_size)

        encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden, input_lengths)
        decoder_input = torch.tensor([[SOS_token]*batch_size], device=device).transpose(0,1)
        decoder_hidden = encoder_hidden

        decoding_token_index = 0
        loss = 0 
        tgt_true_len_np = target_lengths.cpu().numpy()
        #sent_not_end_index = list(range(batch_size))
        idx_token_pred = np.zeros((batch_size, tgt_max_length))
        while decoding_token_index < tgt_max_length:
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, input_lengths, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.detach()  # detach from history as input
            idx_token_pred_step = decoder_input.cpu().squeeze(1).numpy()
            idx_token_pred[:, decoding_token_index] = idx_token_pred_step
            if decoding_token_index < tgt_true_len_np.max():
                loss += criterion(decoder_output, target_tensor[:,decoding_token_index])
            decoding_token_index += 1
            end_or_not = idx_token_pred_step != EOS_token
            sent_not_end_index = list(np.where(end_or_not)[0])
            if len(sent_not_end_index) == 0:
                break

        target_tensor_numpy = target_tensor.cpu().numpy()
        input_tensor_numpy = input_tensor.cpu().numpy()
        for i_batch in range(batch_size):
            tgt_sent_tokens = fun_index2token(target_tensor_numpy[i_batch].tolist(), tgt_idx2words) #:tgt_true_len_np[i_batch]
            #tgt_sents_nltk.append([tgt_sent_tokens])
            tgt_sents_sacre.append(' '.join(tgt_sent_tokens))
            tgt_pred_sent_tokens = fun_index2token(idx_token_pred[i_batch].tolist(), tgt_idx2words)
            #tgt_pred_sents_nltk.append(tgt_pred_sent_tokens)
            tgt_pred_sents_sacre.append(' '.join(tgt_pred_sent_tokens))
            src_sent_tokens = fun_index2token(input_tensor_numpy[i_batch].tolist(), src_idx2words)
            src_sents.append(' '.join(src_sent_tokens))
        # if decoding_token_index == 0:
        #     print('dddddddddd',src_sents[-1],tgt_sents[-1])
        # if target_lengths == 0:
        #     print('fffffffffff',src_sents[-1],tgt_sents[-1])
        loss_all.append(loss.item()/decoding_token_index)
    #nltk_bleu_score = bleu_score.corpus_bleu(tgt_sents_nltk, tgt_pred_sents_nltk)
    sacre_bleu_score = sacrebleu.corpus_bleu(tgt_pred_sents_sacre, [tgt_sents_sacre], smooth='exp', smooth_floor=0.0, force=False, lowercase=False,
        tokenize='none', use_effective_order=True)
    loss = np.mean(loss_all)
    return sacre_bleu_score, None, loss, tgt_sents_sacre, tgt_pred_sents_sacre, src_sents, None # atten

