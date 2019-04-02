import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_prefix = ['<PAD>', '<UNK>', '<EOS>', '<SOS>']
PAD_index = vocab_prefix.index('<PAD>')
UNK_index = vocab_prefix.index('<UNK>')
EOS_index = vocab_prefix.index('<EOS>')
SOS_index = vocab_prefix.index('<SOS>') 

#vocab_pred = ['<PAD>','<OOV>','<EOS>','ISA','DESC','IN','BIRTH',"DEATH", 
#"=", "$", "[", "]", "|", "X", "Y", "Z", "P", "@", "&"]
vocab_pred = ['<PAD>','<OOV>','<EOS>','ISA','DESC','IN','BIRTH',"DEATH",
"=", "$", "[", "]", "|", "X", "Y", "Z", "P", "@", "&", "_"]
vocab_pred_size = len(vocab_pred)

OOV_pred_index = vocab_pred.index('<OOV>')
PAD_pred_index = vocab_pred.index('<PAD>')
EOS_pred_index = vocab_pred.index('<EOS>')

embedding_freeze = False
att_concat_hz = 64

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
# #
# address_book1 = dict(
#     train_src = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/train.tok.zh',
#     train_tgt = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/train.tok.en',
#     val_src = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/dev.tok.zh',
#     val_tgt = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/dev.tok.en',
#     src_emb = 'embedding/wiki.zh.vec',
#     tgt_emb = 'embedding/wiki.en.vec'
# )
# #
#address_book1 = dict(
#    train_src = 'data/zh-en/train_sortByEn_10w.tok.zh',
#    train_tgt = 'data/zh-en/train_sortByEn_10w.tok.en',
#    val_src = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/dev.tok.zh',
#    val_tgt = 'Machine_Translation_NLP/iwsltzhen/iwslt-zh-en/dev.tok.en',
#    src_emb = 'embedding/wiki.zh.vec',
#    tgt_emb = 'embedding/wiki.en.vec'
#)
#
# address_book = dict(
#     train_src = 'Machine_Translation_NLP/iwsltzhen/iwslt-vi-en/train.tok.vi',
#     train_tgt = 'Machine_Translation_NLP/iwsltzhen/iwslt-vi-en/train.tok.en',
#     val_src = 'Machine_Translation_NLP/iwsltzhen/iwslt-vi-en/dev.tok.vi',
#     val_tgt = 'Machine_Translation_NLP/iwsltzhen/iwslt-vi-en/dev.tok.en',
#     src_emb = 'embedding/wiki.vi.vec',
#     tgt_emb = 'embedding/wiki.en.vec'
# )
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

