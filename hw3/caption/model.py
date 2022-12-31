import copy
import math

import timm
import torch.nn as nn
import torch.nn.functional as F
import torch

class image_caption(nn.Module):
    def __init__(self, model_name, test=False, voc_size=18022, d_model = 1024, nhead = 8, num_decoder_layers = 6, dim_feedforward = 2048, 
                 dropout = 0.1, activation = F.relu, layer_norm_eps = 1e-5, batch_first = True, norm_first = False, max_len = 55):
        super(image_caption, self).__init__()
        self.test = test
        self.embedding = nn.Embedding(voc_size, d_model)
        self.pos = positionembedding(d_model, dropout=0, max_len=max_len)

        if test:
            self.encoder = timm.create_model(model_name, pretrained=False)
        else:
            self.encoder = timm.create_model(model_name, pretrained=True)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first)
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.predictor = nn.Linear(d_model, voc_size)

        if not test:
            _reset_parameters(self.decoder)

        self.d_model = d_model
        self.nhead = nhead
        self.max_len = max_len - 1
        self.freeze()       

    def forward(self, src, tgt, tgt_mask=None):
        src = self.encoder.forward_features(src)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos(tgt)
        x = self.decoder(tgt, src, tgt_key_padding_mask=tgt_mask, tgt_mask=generate_square_subsequent_mask(self.max_len, device=tgt.device))
        return x
    
    def freeze(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def generate(self, src, tgt):
        src = self.encoder.forward_features(src)
        for j in range(self.max_len-1):
            tgts = self.embedding(tgt) * math.sqrt(self.d_model)
            tgts = self.pos(tgts)
            pred = self.decoder(tgts, src, tgt_mask=generate_square_subsequent_mask(tgt.size(1), device=tgt.device))

            pred = self.predictor(pred[:, -1, :])

            y = torch.argmax(pred, dim=1)

            tgt = torch.concat([tgt, y.unsqueeze(0)], dim=1)

            if y == 3:
                break
        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm


    def forward(self, tgt, memory, tgt_mask = None, memory_mask = None,
                tgt_key_padding_mask = None, memory_key_padding_mask = None):

        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward = 2048, dropout = 0.1, activation = F.relu,
                 layer_norm_eps = 1e-5, batch_first = False, norm_first = False, device=None, dtype=None):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)


    def forward(self, tgt, memory, tgt_mask = None, memory_mask = None, tgt_key_padding_mask = None, memory_key_padding_mask = None):

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        return x


    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

class positionembedding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(positionembedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

def _reset_parameters(x):
    for p in x.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

def generate_square_subsequent_mask(sz, device='cpu'):
    return torch.triu(torch.full((sz, sz), float('-inf'), device=device, dtype=torch.bool), diagonal=1)
