import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from itertools import dropwhile

from config import vocab_pred, vocab_pred_size, vocab_prefix
from config import UNK_index, PAD_index, SOS_index, EOS_index 
from config import OOV_pred_index, PAD_pred_index, EOS_pred_index


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
        tgt_sym = self.tgt_symbolindex[key]
        tgt_ind = self.tgt_indicator[key]
        
        if self.src_clip is not None:
            src = src[:self.src_clip]
            src_org = src_org[:self.src_clip]
            tgt_ind = tgt_ind[:,:self.src_clip]
        src_length = len(src)

        if self.tgt_clip is not None:
            tgt = tgt[:self.tgt_clip]
            tgt_org = tgt_org[:self.tgt_clip]
            tgt_sym = tgt_sym[:self.tgt_clip]
            tgt_ind = tgt_ind[:self.tgt_clip,:]
        tgt_length = len(tgt)
        
        return src, src_length, tgt, tgt_length, tgt_sym, tgt_ind, src_org, tgt_org
        
        #return src_org, src_tensor, src_true_len, tgt_org, tgt_tensor, tgt_true_len, tgt_label_vocab, tgt_label_copy 

def vocab_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all
    data have the same length
    """
    src_list = []
    tgt_list = []
    src_length_list = []
    tgt_length_list = []
    tgt_symbol_list = []
    tgt_indicator_list = []
    src_org_list = []
    tgt_org_list = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for datum in batch:
        src_length_list.append(datum[1]) # 不用加1；eos不算
        tgt_length_list.append(datum[3]+1) 
    
    batch_max_src_length = np.max(src_length_list)
    batch_max_tgt_length = np.max(tgt_length_list)
    # padding
    for datum in batch:
        #+[EOS_index] -1
        padded_vec = np.pad(np.array(datum[0]), 
                                pad_width=((0, batch_max_src_length-datum[1])),
                                mode="constant", constant_values=PAD_index)
        src_list.append(padded_vec)
        
        padded_vec = np.pad(np.array(datum[2]+[EOS_index]),
                                pad_width=((0, batch_max_tgt_length-datum[3]-1)),
                                mode="constant", constant_values=PAD_index)
        tgt_list.append(padded_vec)
        
        padded_vec = np.pad(np.array(datum[4]+[EOS_pred_index]),
                                pad_width=((0, batch_max_tgt_length-datum[3]-1)),
                                mode="constant", constant_values=PAD_pred_index)
        tgt_symbol_list.append(padded_vec)
        
        indicator = np.pad(datum[5], pad_width=((0,1),(0,0)), 
                           mode='constant', constant_values=0)
        #indicator[-1,-1] = 1  -1
        padded_vec = np.pad(indicator,
                            pad_width=((0, batch_max_tgt_length-datum[3]-1),((0, batch_max_src_length-datum[1]))),
                            mode="constant", constant_values=0)
        #print(padded_vec.dtype, padded_vec.shape)
        tgt_indicator_list.append(padded_vec)
        
        src_org_list.append(datum[6])
        tgt_org_list.append(datum[7])
    
    # re-order
    ind_dec_order = np.argsort(src_length_list)[::-1]
    
    src_list = np.array(src_list)[ind_dec_order]
    src_length_list = np.array(src_length_list)[ind_dec_order]
    tgt_list = np.array(tgt_list)[ind_dec_order]
    tgt_length_list = np.array(tgt_length_list)[ind_dec_order]
    tgt_symbol_list = np.array(tgt_symbol_list)[ind_dec_order]
    #print(tgt_indicator_list[0].dtype, tgt_indicator_list[0][:5][:5])
    tgt_indicator_list = np.array(tgt_indicator_list)[ind_dec_order]
    #print(tgt_indicator_list.dtype, tgt_indicator_list.shape)
    src_org_list = [src_org_list[i] for i in ind_dec_order]
    tgt_org_list = [tgt_org_list[i] for i in ind_dec_order]
    
    #print(type(np.array(data_list)),type(np.array(label_list)))
    
    return [torch.from_numpy(src_list).to(device), 
            torch.LongTensor(src_length_list).to(device), 
            torch.from_numpy(tgt_list).to(device), 
            torch.LongTensor(tgt_length_list).to(device),
            torch.from_numpy(tgt_symbol_list).to(device),
            torch.from_numpy(tgt_indicator_list).to(device),
            src_org_list,
            tgt_org_list,           
           ]


    




