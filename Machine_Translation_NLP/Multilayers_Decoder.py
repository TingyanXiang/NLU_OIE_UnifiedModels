import torch.nn as nn
import torch
import torch.nn.functional as F
from config import embedding_freeze, att_concat_hz, device
import numpy as np 

class DecoderRNN(nn.Module):
    def __init__(self, emb_size, hidden_size, vocab_size, num_layers, rnn_type = 'GRU', embedding_weight = None, dropout_rate = 0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_rate)
        if embedding_weight is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze = embedding_freeze)
        else:
            self.embedding = nn.Embedding(vocab_size,emb_size)
        self.rnn_type = rnn_type
        if rnn_type == 'GRU':
            self.gru = nn.GRU(emb_size, hidden_size, num_layers, batch_first=True, dropout = dropout_rate)
        elif rnn_type == 'LSTM':
            self.lstm = nn.LSTM(emb_size, hidden_size, num_layers, batch_first=True, dropout = dropout_rate)
        else:
            print('RNN TYPE ERROR')
        self.out = nn.Linear(hidden_size, vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, tgt_input, hidden, true_len = None, encoder_outputs = None, cell = None):
        output = self.embedding(tgt_input)
        #print(output.size())
        if self.rnn_type == 'GRU':
            output, hidden = self.gru(output, hidden)
        else:
            output, (hidden, cell) = self.lstm(output,(hidden, cell))
        logits = self.out(output.squeeze(1))
        output = self.logsoftmax(logits)
        return output, hidden, None, cell

    # def initHidden(self, encoder_hidden):
    #     batch_size = encoder_hidden.size(1)
    #     return encoder_hidden.expand(self.num_layers, batch_size, self.hidden_size).contiguous()
    

class DecoderAtten(nn.Module):
    # encoder_params = (en_num_layers, en_num_direction, en_hidden_size)
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers, encoder_params, rnn_type='GRU', embedding_weight=None, atten_type='dot_prod', dropout_rate=0.1):
        super(DecoderAtten, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.num_layers = num_layers
        #self.transform_en_hid = nn.Linear(np.prod(encoder_params), num_layers*hidden_size)
        en_output_hz = encoder_params[1]*encoder_params[2]
        if embedding_weight is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, emb_size)
        self.rnn_type = rnn_type
        if rnn_type == 'GRU':
            self.gru = nn.GRU(emb_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == 'LSTM':
            self.lstm = nn.LSTM(emb_size, hidden_size, num_layers, batch_first=True)
        else:
            print('RNN TYPE ERROR')
        self.atten = AttentionLayer(hidden_size, en_output_hz, atten_type=atten_type)
        
        self.linear = nn.Linear(en_output_hz+hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, tgt_input, hidden, true_len, encoder_outputs, cell=None):
        output = self.embedding(tgt_input)
        output = self.dropout(output)
        #print(output.size())
        if self.rnn_type == 'GRU':
            output, hidden = self.gru(output, hidden)
        else:
            output, (hidden, cell) = self.lstm(output, (hidden, cell))
        ### add attention
        atten_output, atten_weight = self.atten(output, encoder_outputs, true_len)
        out1 = torch.cat((output, atten_output),-1)
        out2 = self.linear(out1.squeeze(1))
        out2 = F.relu(out2)
        logits = self.out(out2)
        output = self.logsoftmax(logits)
        return output, hidden, atten_weight, cell
    
    def initHidden(self, batch_size):
        # batch_size = encoder_hidden.size(1)
        #(en_num_layers*num_direction, bz, en_hidden_size) >> (bz, en_num_layers*num_direction*en_hidden_size)
        # encoder_hidden = encoder_hidden.transpose(0,1).contiguous().view(batch_size, -1)
        # hidden = self.transform_en_hid(encoder_hidden) #(bz, de_num_layers*de_hidden_size)
        # hidden = hidden.view(batch_size, self.num_layers, self.hidden_size).transpose(0,1).contiguous()
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return cell #(de_num_layers, bz, de_hidden_size)

class AttentionLayer(nn.Module):
    def __init__(self, q_hidden_size, m_hidden_size, atten_type):
        super(AttentionLayer, self).__init__()
        self.q_hidden_size = q_hidden_size
        self.m_hidden_size = m_hidden_size
        self.mode = atten_type
        if atten_type == 'dot_prod':
            if q_hidden_size != m_hidden_size:
                print((q_hidden_size, m_hidden_size), 'query and memory must have the same hidden size; use general way automatically')
                self.mode = 'general'
                self.general_linear = nn.Linear(q_hidden_size, m_hidden_size, bias=False)
            else:
                print('dot_prod')
        elif atten_type == 'general':
            print('general')
            self.general_linear = nn.Linear(q_hidden_size, m_hidden_size, bias=False)
        elif atten_type == 'concat':
            print('concat')
            self.content_linear = nn.Linear(q_hidden_size+m_hidden_size, att_concat_hz, bias=True)
            self.score_linear = nn.Linear(att_concat_hz, 1, bias = False)
        else:
            print('mode out of bound')

    def forward(self, query, memory_bank, true_len):
        #batch_size, src_len, hidden_size = memory_bank.size()
        #query_len = query.size(1)
        scores = self.atten_score(query, memory_bank)
        
        mask_matrix = sequence_mask(true_len).unsqueeze(1)
        scores.masked_fill_(1-mask_matrix, float('-inf'))
        scores_normalized = F.softmax(scores, dim=-1)
        #scores_normalized = F.softmax(scores.view(batch_size * query_len, seq_len), dim=-1).view(batch_size, query_len, seq_len)
        context = torch.bmm(scores_normalized, memory_bank)
        
        return context, scores_normalized #(bz, query_len, m_hidden_size) (bz, query_len, src_len)
    
    def atten_score(self, query, memory_bank):
        """
        query: (batch, tgt_length, q_hidden_size)
        memory_bank: (batch, src_length, m_hidden_size)
        return: (batch, tgt_length, src_length)
        """
        batch_size, src_len, m_hidden_size = memory_bank.size()
        query_len = query.size(1)
        if self.mode == 'dot_prod':
            out = torch.bmm(query, memory_bank.transpose(1, 2))
        elif self.mode == 'general':
            temp = self.general_linear(query.view(batch_size * query_len, self.q_hidden_size))
            out = torch.bmm(temp.view(batch_size,query_len,self.m_hidden_size),memory_bank.transpose(1, 2))
        elif self.mode == 'concat':
            query_temp = query.unsqueeze(2).expand(batch_size,query_len,src_len,self.q_hidden_size)
            memory_temp = memory_bank.unsqueeze(1).expand(batch_size,query_len,src_len,self.m_hidden_size)
            content_out = self.content_linear(torch.cat((query_temp,memory_temp),-1).view(batch_size * query_len * src_len, self.m_hidden_size+self.q_hidden_size))
            content_out = torch.tanh(content_out)
            out = self.score_linear(content_out)
            out = out.squeeze(-1).view(batch_size, query_len, src_len)
        else:
            print('mode out of bound')
        return out #(bz, query_len, src_len)

def sequence_mask(lengths):
    batch_size = lengths.numel()
    max_len = lengths.max()
    return (torch.arange(0, max_len).type_as(lengths).repeat(batch_size,1).lt(lengths.unsqueeze(1)))
