import torch

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3
vocab_prefix = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_freeze = False

#src_vocab_size = 40000
#tgt_vocab_size = 20000
#tgt_max_length = 60
#max_src_len_dataloader, max_tgt_len_dataloader = 60, 60
#embedding_freeze = False
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#address_book1 = dict(
#    train_src = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/train_sortByEn.tok.zh',
#    train_tgt = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/train_sortByEn.tok.en',
#    val_src = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/dev.tok.zh',
#    val_tgt = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/dev.tok.en',
#    src_emb = 'embedding/wiki.zh.vec',
#    tgt_emb = 'embedding/wiki.en.vec'
#)
#
address_book1 = dict(
    train_src = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/train.tok.zh',
    train_tgt = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/train.tok.en',
    val_src = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/dev.tok.zh',
    val_tgt = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/dev.tok.en',
    src_emb = 'embedding/wiki.zh.vec',
    tgt_emb = 'embedding/wiki.en.vec'
)
#
#address_book1 = dict(
#    train_src = 'data/zh-en/train_sortByEn_10w.tok.zh',
#    train_tgt = 'data/zh-en/train_sortByEn_10w.tok.en',
#    val_src = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/dev.tok.zh',
#    val_tgt = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/dev.tok.en',
#    src_emb = 'embedding/wiki.zh.vec',
#    tgt_emb = 'embedding/wiki.en.vec'
#)
#
address_book = dict(
    train_src = 'Machine_Translation_NLP/iwsltzhen/iwslt-vi-en/train.tok.vi',
    train_tgt = 'Machine_Translation_NLP/iwsltzhen/iwslt-vi-en/train.tok.en',
    val_src = 'Machine_Translation_NLP/iwsltzhen/iwslt-vi-en/dev.tok.vi',
    val_tgt = 'Machine_Translation_NLP/iwsltzhen/iwslt-vi-en/dev.tok.en',
    src_emb = 'embedding/wiki.vi.vec',
    tgt_emb = 'embedding/wiki.en.vec'
)
#address_book1 = dict(
#    train_src = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/dev.tok.zh',
#    train_tgt = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/dev.tok.en',
#    val_src = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/dev.tok.zh',
#    val_tgt = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/dev.tok.en',
#    src_emb = 'embedding/wiki.zh.vec',
#    tgt_emb = 'embedding/wiki.en.vec'
#)
#address_book1 = dict(
#    train_src = 'Machine_Translation_NLP/iwsltzhen/iwslt-vi-en/train.tok.en',
#    train_tgt = 'Machine_Translation_NLP/iwsltzhen/iwslt-vi-en/train.tok.en',
#    val_src = 'Machine_Translation_NLP/iwsltzhen/iwslt-vi-en/dev.tok.en',
#    val_tgt = 'Machine_Translation_NLP/iwsltzhen/iwslt-vi-en/dev.tok.en',
#    src_emb = 'embedding/wiki.en.vec',
#    tgt_emb = 'embedding/wiki.en.vec'
#)
#
