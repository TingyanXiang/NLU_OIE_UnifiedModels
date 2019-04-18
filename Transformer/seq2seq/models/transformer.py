import torch
import torch.nn as nn
import math
from copy import deepcopy
from .seq2seq_base import Seq2Seq
from .modules.state import State
from .modules.transformer_blocks import EncoderBlock, DecoderBlock, EncoderBlockPreNorm, DecoderBlockPreNorm, positional_embedding, CharWordEmbedder
from config import PAD_index, vocab_pred_size

class TransformerAttentionEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=512, embedding_size=None,
                 num_layers=6, num_heads=8, inner_linear=2048, inner_groups=1, prenormalized=False,
                 mask_symbol=PAD_index, layer_norm=True, weight_norm=False, dropout=0, embedder=None):

        super(TransformerAttentionEncoder, self).__init__()
        embedding_size = embedding_size or hidden_size
        if embedding_size != hidden_size:
            self.input_projection = nn.Parameter(
                torch.empty(embedding_size, hidden_size))
            nn.init.kaiming_uniform_(self.input_projection, a=math.sqrt(5))
        self.hidden_size = hidden_size
        self.batch_first = True
        self.mask_symbol = mask_symbol
        self.embedder = embedder or nn.Embedding(
            vocab_size, embedding_size, padding_idx=PAD_index)
        self.scale_embedding = hidden_size ** 0.5
        self.dropout = nn.Dropout(dropout, inplace=True)
        if prenormalized:
            block = EncoderBlockPreNorm
        else:
            block = EncoderBlock
        self.blocks = nn.ModuleList([block(hidden_size,
                                           num_heads=num_heads,
                                           inner_linear=inner_linear,
                                           inner_groups=inner_groups,
                                           layer_norm=layer_norm,
                                           weight_norm=weight_norm,
                                           dropout=dropout)
                                     for _ in range(num_layers)
                                     ])
        if layer_norm and prenormalized:
            self.lnorm = nn.LayerNorm(hidden_size)

    def forward(self, inputs, hidden=None):
        if self.mask_symbol is not None:
            padding_mask = inputs.eq(self.mask_symbol)
        else:
            padding_mask = None
        x = self.embedder(inputs).mul_(self.scale_embedding)
        if hasattr(self, 'input_projection'):
            x = x @ self.input_projection
        x.add_(positional_embedding(x))
        x = self.dropout(x)

        for block in self.blocks:
            block.set_mask(padding_mask)
            x = block(x)

        if hasattr(self, 'lnorm'):
            x = self.lnorm(x)
        return State(outputs=x, mask=padding_mask, batch_first=True)


class TransformerAttentionDecoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=512, embedding_size=None, num_layers=6,
                 num_heads=8, dropout=0, inner_linear=2048, inner_groups=1, prenormalized=False, stateful=False, state_dim=None,
                 mask_symbol=PAD_index, tie_embedding=True, layer_norm=True, weight_norm=False, embedder=None, classifier_type=None):

        super(TransformerAttentionDecoder, self).__init__()
        embedding_size = embedding_size or hidden_size
        if embedding_size != hidden_size:
            self.input_projection = nn.Parameter(
                torch.empty(embedding_size, hidden_size))
            nn.init.kaiming_uniform_(self.input_projection, a=math.sqrt(5))
        self.batch_first = True
        self.mask_symbol = mask_symbol
        self.embedder = embedder or nn.Embedding(
            vocab_size, embedding_size, padding_idx=PAD_index)
        self.scale_embedding = hidden_size ** 0.5
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.stateful = stateful
        if prenormalized:
            block = DecoderBlockPreNorm
        else:
            block = DecoderBlock
        self.blocks = nn.ModuleList([block(hidden_size,
                                           num_heads=num_heads,
                                           inner_linear=inner_linear,
                                           inner_groups=inner_groups,
                                           layer_norm=layer_norm,
                                           weight_norm=weight_norm,
                                           dropout=dropout,
                                           stateful=stateful,
                                           state_dim=state_dim)
                                     for _ in range(num_layers)
                                     ])
        if layer_norm and prenormalized:
            self.lnorm = nn.LayerNorm(hidden_size)
        
        self.classifier_type = classifier_type
        if classifier_type == 'normal':
            self.classifier = nn.Linear(embedding_size, vocab_size)
            if tie_embedding:
                self.embedder.weight = self.classifier.weight

            if embedding_size != hidden_size:
                if tie_embedding:
                    self.output_projection = self.input_projection
                else:
                    self.output_projection = nn.Parameter(
                        torch.empty(embedding_size, hidden_size))
                    nn.init.kaiming_uniform_(
                        self.output_projection, a=math.sqrt(5))
            self.logsoftmax = nn.LogSoftmax(dim=-1)
        elif classifier_type == 'copy':
            self.classifier = CopyMechanism(hidden_size, hidden_size, vocab_pred_size)
        else:
            self.classifier = None


    def forward(self, inputs, state, get_attention=False):
        context = state.context
        time_step = 0
        if self.stateful:
            block_state = state.hidden
            if block_state is None:
                self.time_step = 0
            time_step = self.time_step
        else:
            block_state = state.inputs
            time_step = 0 if block_state is None else block_state[0].size(1)

        if block_state is None:
            block_state = [None] * len(self.blocks)

        if self.mask_symbol is not None:
            padding_mask = inputs.eq(self.mask_symbol)
        else:
            padding_mask = None
        x = self.embedder(inputs).mul_(self.scale_embedding)
        if hasattr(self, 'input_projection'):
            x = x @ self.input_projection
        x.add_(positional_embedding(x, offset=time_step))
        x = self.dropout(x)

        attention_scores = []
        updated_state = []
        for i, block in enumerate(self.blocks):
            block.set_mask(padding_mask, context.mask)
            x, attn_enc, block_s = block(x, context.outputs, block_state[i])
            updated_state.append(block_s)
            if get_attention:
                attention_scores.append(attn_enc)
            else:
                del attn_enc

        if hasattr(self, 'lnorm'):
            x = self.lnorm(x)

        if hasattr(self, 'output_projection'):
            x = x @ self.output_projection.t()
        if self.classifier_type == 'normal'
            x = self.classifier(x)
            x = self.logsoftmax(x)
        elif self.classifier_type == 'copy':
            x = self.classifier(x, context)
        else:
            pass
        
        if self.stateful:
            state.hidden = tuple(updated_state)
            self.time_step += 1
        else:
            state.inputs = tuple(updated_state)
        if get_attention:
            state.attention_score = attention_scores
        return x, state

def sequence_mask(lengths):
    batch_size = lengths.numel()
    max_len = lengths.max()
    return (torch.arange(0, max_len).type_as(lengths).repeat(batch_size,1).lt(lengths.unsqueeze(1)))

class CopyMechanism(nn.Module):
    def __init__(self, de_logits_hz, en_output_hz, vocab_size_pred):
        super(CopyMechanism, self).__init__()
        self.generate_linear = nn.Linear(de_logits_hz, vocab_size_pred)
        self.copy_linear = nn.Linear(en_output_hz, de_logits_hz)
        self.LogSoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, logits, context):
        encoder_outputs = context.outputs
        generation_scores = self.generate_linear(logits) #(bz, de_logits_hz)>>(bz, vocab_size_pred)
        # remove sos and eos
        encoder_out = torch.tanh(self.copy_linear(encoder_outputs)) #(bz, src_sen_len, en_output_hz)>>(bz, src_sen_len, de_logits_hz)
        copy_scores = torch.bmm(encoder_out, logits.unsqueeze(-1)).squeeze(-1) #(bz, src_sen_len)
        # mask copy_scores for padding
        mask_matrix = context.mask
        copy_scores.masked_fill_(mask_matrix, float('-inf'))
        scores = torch.cat((generation_scores, copy_scores), dim=-1)
        log_prob_scores = self.LogSoftmax(scores) #(bz, vocab_size_pred+src_sen_len)
        return log_prob_scores


class Transformer(Seq2Seq):

    def __init__(self, vocab_size, hidden_size=512, embedding_size=None, num_layers=6, num_heads=8,
                 inner_linear=2048, inner_groups=1, dropout=0.1, prenormalized=False, tie_embedding=True,
                 encoder=None, decoder=None, layer_norm=True, weight_norm=False, stateful=None):
        super(Transformer, self).__init__()
        embedding_size = embedding_size or hidden_size
        # keeping encoder, decoder None will result with default configuration
        encoder = encoder or {}
        decoder = decoder or {}
        encoder = deepcopy(encoder)
        decoder = deepcopy(decoder)
        encoder.setdefault('embedding_size', embedding_size)
        encoder.setdefault('hidden_size', hidden_size)
        encoder.setdefault('num_layers', num_layers)
        encoder.setdefault('num_heads', num_heads)
        encoder.setdefault('vocab_size', vocab_size)
        encoder.setdefault('layer_norm', layer_norm)
        encoder.setdefault('weight_norm', weight_norm)
        encoder.setdefault('dropout', dropout)
        encoder.setdefault('inner_linear', inner_linear)
        encoder.setdefault('inner_groups', inner_groups)
        encoder.setdefault('prenormalized', prenormalized)

        decoder.setdefault('embedding_size', embedding_size)
        decoder.setdefault('hidden_size', hidden_size)
        decoder.setdefault('num_layers', num_layers)
        decoder.setdefault('num_heads', num_heads)
        decoder.setdefault('tie_embedding', tie_embedding)
        decoder.setdefault('vocab_size', vocab_size)
        decoder.setdefault('layer_norm', layer_norm)
        decoder.setdefault('weight_norm', weight_norm)
        decoder.setdefault('dropout', dropout)
        decoder.setdefault('inner_linear', inner_linear)
        decoder.setdefault('inner_groups', inner_groups)
        decoder.setdefault('prenormalized', prenormalized)
        decoder.setdefault('stateful', stateful)

        if isinstance(vocab_size, tuple):
            embedder = CharWordEmbedder(
                vocab_size[1], embedding_size, hidden_size)
            encoder.setdefault('embedder', embedder)
            decoder.setdefault('embedder', embedder)
            decoder['classifier'] = False

        self.batch_first = True
        self.encoder = TransformerAttentionEncoder(**encoder)
        self.decoder = TransformerAttentionDecoder(**decoder)

        if tie_embedding and not isinstance(vocab_size, tuple):
            assert self.encoder.embedder.weight.shape == self.decoder.classifier.weight.shape
            self.encoder.embedder.weight = self.decoder.classifier.weight
            if embedding_size != hidden_size:
                self.encoder.input_projection = self.decoder.input_projection
