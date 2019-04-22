# config:
import torch
import numpy as np
from config import fact_seperator, element_seperator
from config import SOS_index, UNK_index, EOS_index, PAD_index, OOV_pred_index, EOS_pred_index, vocab_pred, vocab_pred_size
#import beam
import difflib
from seq2seq.models.transformer import sequence_mask
from seq2seq.models.modules.state import State
from scipy.optimize import linear_sum_assignment

def similarity_score(fact1, fact2):
    elem1 = fact1.split(element_seperator)
    elem2 = fact2.split(element_seperator)
    n1 = len(elem1)
    n2 = len(elem2)
    sim = 0
    for i in range(min(n1,n2)):
        sim += difflib.SequenceMatcher(None,elem1[i],elem2[i]).ratio()
    return sim/max(n1,n2)

def check_fact_same(org_fact, pred_fact):
    org_fact_ele = org_fact.split(element_seperator)
    pred_fact_ele = pred_fact.split(element_seperator)
    if len(org_fact_ele) == len(pred_fact_ele):
        ele_num = len(org_fact_ele)
        if difflib.SequenceMatcher(None,org_fact,pred_fact).ratio() > 0.85:
            return True       
        ele_sim = np.zeros((ele_num,))
        for ele_i in range(ele_num):
            ele_sim[ele_i] = difflib.SequenceMatcher(None,org_fact_ele[ele_i],pred_fact_ele[ele_i]).ratio()
        if ele_sim.mean() > 0.85:
            return True
    return False

def bridge(context):
    return State(context=context,batch_first=True)

def predict_facts(loader, encoder, decoder, tgt_max_length, vocab):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    encoder.eval()
    decoder.eval()

    tgt_pred = []
    src_org = []
    tgt_org = []
    loss = 0

    for src_tensor, src_true_len, tgt_tensor, tgt_true_len, tgt_label_vocab, tgt_label_copy, src_org_batch, tgt_org_batch in loader:
        batch_size = src_tensor.size(0)
        encoder_context = encoder(src_tensor)
        state = bridge(encoder_context)

        decoder_input = torch.tensor([SOS_index]*batch_size, device=device).unsqueeze(1)

        decoding_token_index = 0
        stop_flag = [False]*batch_size
        step_log_likelihoods = []
        tgt_pred_batch = [[] for i_batch in range(batch_size)]
        tgt_true_len_max = tgt_true_len.cpu().numpy().max()
        while decoding_token_index < tgt_max_length:
            decoder_output, _ = decoder(decoder_input, state) # state update at each step

            # # compute loss 
            # if decoding_token_index < tgt_true_len_max:
            #     decoding_label_vocab = tgt_label_vocab[:, decoding_token_index]
            #     decoding_label_copy = tgt_label_copy[:, decoding_token_index, :]
            #     copy_log_probs = decoder_output[:, vocab_pred_size:]+(decoding_label_copy.float()+1e-45).log()
            #     #mask sample which is copied only
            #     gen_mask = ((decoding_label_vocab!=OOV_pred_index) | (decoding_label_copy.sum(-1)==0)).float() 
            #     log_gen_mask = (gen_mask + 1e-45).log().unsqueeze(-1)
            #     #mask log_prob value for oov_pred_index when label_vocab==oov_pred_index and is copied 
            #     generation_log_probs = decoder_output.gather(1, decoding_label_vocab.unsqueeze(1)) + log_gen_mask
            #     combined_gen_and_copy = torch.cat((generation_log_probs, copy_log_probs), dim=-1)
            #     step_log_likelihood = torch.logsumexp(combined_gen_and_copy, dim=-1)
            #     step_log_likelihoods.append(step_log_likelihood.unsqueeze(1))

            # prediction
            topv, topi = decoder_output.topk(1, dim=-1)
            next_input = topi.detach().cpu().squeeze(1)
            decoder_input = []
            for i_batch in range(batch_size):
                pred_list = vocab_pred+src_org_batch[i_batch]
                next_input_token = pred_list[next_input[i_batch].item()]
                if next_input_token == vocab_pred[EOS_pred_index]:
                    stop_flag[i_batch] = True
                if not stop_flag[i_batch]:
                    tgt_pred_batch[i_batch].append(next_input_token)
                decoder_input.append(vocab.word2index.get(next_input_token, UNK_index))
            decoder_input = torch.tensor(decoder_input, device=device).unsqueeze(1)
            decoding_token_index += 1
            if all(stop_flag):
                break
        # log_likelihoods = torch.cat(step_log_likelihoods, dim=-1)
        # # mask padding for tgt
        # tgt_pad_mask = sequence_mask(tgt_true_len).float()
        # log_likelihoods = log_likelihoods*tgt_pad_mask[:,:log_likelihoods.size(1)]
        # loss += -(log_likelihoods.sum()/tgt_pad_mask.sum()).item()

        tgt_pred.extend(tgt_pred_batch)
        src_org.extend(src_org_batch)
        tgt_org.extend(tgt_org_batch)
    # loss = loss/len(loader)
    return loss, src_org, tgt_org, tgt_pred

def evaluate_prediction(tgt_org, tgt_pred):
    eval_len = len(tgt_pred)
    precision = np.zeros((eval_len,))
    recall = np.zeros((eval_len,))
    F_scores = np.zeros((eval_len,))
    matched_num = np.zeros((eval_len,))
    org_num = np.zeros((eval_len,))
    pred_num = np.zeros((eval_len,))
    for i in range(eval_len):
        org_facts = ''.join(tgt_org[i]).split(fact_seperator)
        pred_facts = ''.join(tgt_pred[i]).split(fact_seperator)
        # remove duplicates
        pred_facts = list(set(pred_facts))
        org_facts_num = len(org_facts)
        pred_facts_num = len(pred_facts)
        org_match_num = np.zeros((org_facts_num))
        pred_match_num = np.zeros((pred_facts_num))
        similarity_ma = np.zeros((org_facts_num, pred_facts_num))
        for org_i in range(org_facts_num):
            for pred_i in range(pred_facts_num):
                similarity_ma[org_i, pred_i] = similarity_score(org_facts[org_i], pred_facts[pred_i])
        row_ind, col_ind = linear_sum_assignment(-similarity_ma)
        
        for org_i, pred_i in zip(row_ind, col_ind):
            org_fact = org_facts[org_i]
            pred_fact = pred_facts[pred_i]
            fact_same = check_fact_same(org_fact, pred_fact)
            if fact_same:
                org_match_num[org_i] = 1
                pred_match_num[pred_i] = 1
        matched_num[i] = pred_match_num.sum()
        assert(matched_num[i]==org_match_num.sum())
        org_num[i] = org_facts_num
        pred_num[i] = pred_facts_num
        precision[i] = pred_match_num.mean()
        recall[i] = org_match_num.mean()
        F_scores[i] = 2*precision[i]*recall[i]/(precision[i]+recall[i]+1e-10)
    return precision, recall



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
        decoder_input = torch.tensor([SOS_index]*batch_size, device=device).unsqueeze(1)
        decoder_hidden, decoder_cell = encoder_hidden, decoder.initHidden(batch_size)

        decoding_token_index = 0
        step_log_likelihoods = []
        tgt_pred_batch = [[] for i_batch in range(batch_size)]
        tgt_true_len_max = tgt_true_len.cpu().numpy().max()
        stop_flag = [False]*batch_size
        #sent_not_end_index = list(range(batch_size))
        while decoding_token_index < tgt_max_length:
            decoder_output, decoder_hidden, _, decoder_cell = decoder(decoder_input, decoder_hidden, src_true_len, encoder_outputs, decoder_cell)
            
            # compute loss 
            if decoding_token_index < tgt_true_len_max:
                decoding_label_vocab = tgt_label_vocab[:, decoding_token_index]
                decoding_label_copy = tgt_label_copy[:, decoding_token_index, :]
                copy_log_probs = decoder_output[:, vocab_pred_size:]+(decoding_label_copy.float()+1e-45).log()
                #mask sample which is copied only
                gen_mask = ((decoding_label_vocab!=OOV_pred_index) | (decoding_label_copy.sum(-1)==0)).float() 
                log_gen_mask = (gen_mask + 1e-45).log().unsqueeze(-1)
                #mask log_prob value for OOV_pred_index when label_vocab==OOV_pred_index and is copied 
                generation_log_probs = decoder_output.gather(1, decoding_label_vocab.unsqueeze(1)) + log_gen_mask
                combined_gen_and_copy = torch.cat((generation_log_probs, copy_log_probs), dim=-1)
                step_log_likelihood = torch.logsumexp(combined_gen_and_copy, dim=-1)
                step_log_likelihoods.append(step_log_likelihood.unsqueeze(1))

            #
            topv, topi = decoder_output.topk(1, dim=-1)
            next_input = topi.detach().cpu().squeeze(1)
            decoder_input = []
            for i_batch in range(batch_size):
                #print(''.join(src_org_batch[i_batch]), ''.join(tgt_org_batch[i_batch]))
                pred_list = vocab_pred+src_org_batch[i_batch]+['<EOS>']
                next_input_token = pred_list[next_input[i_batch].item()]
                if next_input_token == vocab_pred[EOS_pred_index]:
                    stop_flag[i_batch] = True
                if stop_flag[i_batch] is False:
                    tgt_pred_batch[i_batch].append(next_input_token)
                decoder_input.append(vocab.word2index.get(next_input_token, UNK_index))
            decoder_input = torch.tensor(decoder_input, device=device).unsqueeze(1)
            decoding_token_index += 1
            if all(stop_flag):
                break
        #print(src_org_batch[0], '\n', tgt_org_batch[0], '\n', tgt_pred_batch[0])
        log_likelihoods = torch.cat(step_log_likelihoods, dim=-1)
        # mask padding for tgt
        tgt_pad_mask = sequence_mask(tgt_true_len).float()
        log_likelihoods = log_likelihoods*tgt_pad_mask[:,:log_likelihoods.size(1)]
        loss += -(log_likelihoods.sum()/tgt_pad_mask.sum()).item()
        tgt_pred.extend(tgt_pred_batch)
        src_org.extend(src_org_batch)
        tgt_org.extend(tgt_org_batch)
    loss = loss/len(loader)
    eval_len = len(tgt_pred)
    precision = np.zeros((eval_len,))
    recall = np.zeros((eval_len,))
    for i in range(eval_len):
        org_facts = ''.join(tgt_org[i]).split('$')
        pred_facts = ''.join(tgt_pred[i]).split('$')
        org_facts_num = len(org_facts)
        pred_facts_num = len(pred_facts)
        org_match_num = np.zeros((org_facts_num))
        pred_match_num = np.zeros((pred_facts_num))
        for org_i in range(org_facts_num):
            for pred_i in range(pred_facts_num):
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
    if False:
        random_sample = np.random.randint(eval_len)
        print('src:', src_org[random_sample])
        print('Ref: ', tgt_org[random_sample])
        print('pred: ', tgt_pred[random_sample])
    return precision, recall, loss
            

