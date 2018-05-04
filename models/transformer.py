''' Define the Transformer model '''
import math
import torch as tc
import torch.nn as nn
import torch.nn.init as init
import numpy as np

import wargs
from tools.utils import *
from models.losser import *

__author__ = "Yu-Hsiang Huang"
np.set_printoptions(threshold=np.nan)

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(self, n_src_vocab, n_tgt_vocab, n_max_seq, n_layers=6, n_head=8,
                 d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1,
                 proj_share_weight=True, embs_share_weight=True, use_attcnn=False):

        wlog('Transformer Model ========================= ')
        wlog('\tn_src_vocab:        {}'.format(n_src_vocab))
        wlog('\tn_trg_vocab:        {}'.format(n_tgt_vocab))
        wlog('\tn_max_seq:          {}'.format(n_max_seq))
        wlog('\tn_layers:           {}'.format(n_layers))
        wlog('\tn_head:             {}'.format(n_head))
        wlog('\td_word_vec:         {}'.format(d_word_vec))
        wlog('\td_model:            {}'.format(d_model))
        wlog('\td_inner_hid:        {}'.format(d_inner_hid))
        wlog('\tdropout:            {}'.format(dropout))
        wlog('\tproj_share_weight:  {}'.format(proj_share_weight))
        wlog('\tembs_share_weight:  {}'.format(embs_share_weight))
        wlog('\tuse_attcnn:         {}'.format(use_attcnn))

        super(Transformer, self).__init__()
        self.encoder = Encoder(n_src_vocab, n_max_seq, n_layers=n_layers, n_head=n_head,
                               d_word_vec=d_word_vec, d_model=d_model,
                               d_inner_hid=d_inner_hid, dropout=dropout, use_attcnn=use_attcnn)
        self.decoder = Decoder(n_tgt_vocab, n_max_seq, n_layers=n_layers, n_head=n_head,
                               d_word_vec=d_word_vec, d_model=d_model,
                               d_inner_hid=d_inner_hid, dropout=dropout,
                               proj_share_weight=proj_share_weight, use_attcnn=use_attcnn)

        assert d_model == d_word_vec, 'To facilitate the residual connections, \
                the dimensions of all module output shall be the same.'
        if embs_share_weight is True:
            # Share the weight matrix between src and tgt word embeddings
            assert n_src_vocab == n_tgt_vocab, \
            "To share word embedding table, the vocabulary size of src and tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def get_trainable_parameters(self):
        ''' Avoid updating the position encoding '''
        enc_freezed_param_ids = set(map(id, self.encoder.position_enc.parameters()))
        dec_freezed_param_ids = set(map(id, self.decoder.position_enc.parameters()))
        freezed_param_ids = enc_freezed_param_ids | dec_freezed_param_ids
        return ((n, p) for (n, p) in self.named_parameters() if id(p) not in freezed_param_ids)

    def forward(self, src, tgt):

        src_seq, src_pos = src
        tgt_seq, tgt_pos = tgt

        tgt_seq = tgt_seq[:, :-1]
        tgt_pos = tgt_pos[:, :-1]

        enc_outputs, enc_slf_attn, enc_slf_one_attn = self.encoder(src_seq, src_pos)
        enc_output = enc_outputs[-1]
        dec_output, dec_slf_attns, dec_enc_attns, dec_enc_one_attn = \
                self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)

        return dec_output

class BatchConv1d(nn.Module):

    def __init__(self, q_size, k_size, kernel_width = 3, use_mask=False):

        super(BatchConv1d, self).__init__()
        self.q_size = q_size
        self.k_size = k_size
        self.kernel_width = kernel_width
        self.padding_width = (self.kernel_width - 1) / 2
        # (batch_size * q_len) * q_size -> (batch_size * q_len) * k_size * kernel_width
        self.q_to_kernel = nn.Linear(q_size, k_size * self.kernel_width, bias=True)
        self.q_to_bias = nn.Linear(q_size, 1, bias=True)
        #nn.init.xavier_normal(self.q_to_kernel.weight)
        self.bias_b = nn.Parameter(tc.FloatTensor(1))
        nn.init.normal(self.bias_b)
        #self.bias_b.fill_(0.0)
        self.use_mask = use_mask
        if self.use_mask is True:
            self.kernel_mask = tc.zeros(kernel_width)
            self.kernel_mask[:(self.kernel_width + 1) / 2] = 1
            self.kernel_mask = Variable(self.kernel_mask, requires_grad=False).cuda()

    def forward(self, q, k):

        batch_size, q_len, q_size = q.size()
        batch_size, k_len, k_size = k.size()

        # conv1d(input, weight)
        #### original parameter dimensions ####
        # input.size = batch_size * in_channels * input_width
        # weight.size = out_channels * in_channels * kernel_width
        # output.size = batch_size * out_channels * output_width
        #### use groups to make every sentence in batch should have its own kernel #####
        # groups = batch_size
        # input.size = 1 * (batch_size * k_size) * kv_len
        # weight.size = batch_size *  k_size * kernel_width
        # bias.size = batch_size
        # output.size = 1 * batch_size * kv_len
        # q: (B*n_head, L_q, d_k), k: (B*n_head, L_k, d_k)
        inp = k.permute(0, 2, 1).contiguous().view(1, batch_size * k_size, k_len)
        #q_flat = q.view(batch_size * q_len, q_size)
        #kernel = self.q_to_kernel(q_flat).view(batch_size * q_len, k_size, self.kernel_width)
        #bias   = self.q_to_bias  (q_flat).view(batch_size * q_len)
        kernel = self.q_to_kernel(q).view(batch_size * q_len, k_size, self.kernel_width)
        bias   = self.q_to_bias  (q).view(batch_size * q_len)
        if self.use_mask is True: kernel = kernel * self.kernel_mask[None, None, :]
        conv_res = F.conv1d(inp,
                            kernel,
                            bias=bias,
                            groups=batch_size,
                            padding=self.padding_width) # (1, batch_size * q_len, k_len)
        conv_res_b = conv_res + self.bias_b
        a_ij = conv_res_b.view(batch_size, q_len, k_len) # kv_len * batch_size

        return a_ij

class MultiHeadAttention(nn.Module):
    '''
        Multi-Head Attention module from <Attention is All You Need>
        Args:
            n_head(int):    number of parallel heads.
            d_model(int):   the dimension of keys/values/queries in this MultiHeadAttention
                d_model % n_head == 0
            d_k(int):       the dimension of queries and keys
            d_v(int):       the dimension of values
    '''
    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1, use_attcnn=True, use_mask=False):

        super(MultiHeadAttention, self).__init__()

        assert d_model % n_head == 0, 'd_model {} divided by n_head {}.'.format(d_model, n_head)
        self.d_model, self.n_head, self.d_k, self.d_v = d_model, n_head, d_k, d_v
        self.dim_per_head = d_model // n_head

        self.linear_q = XavierLinear(d_model, n_head * self.dim_per_head, bias=True)
        self.linear_k = XavierLinear(d_model, n_head * self.dim_per_head, bias=True)
        self.linear_v = XavierLinear(d_model, n_head * self.dim_per_head, bias=True)
        self.temper = self.dim_per_head ** 0.5

        self.mSoftMax = MaskSoftmax()
        self.dropout = nn.Dropout(dropout)
        self.use_attcnn = use_attcnn
        if self.use_attcnn is True:
            #self.kernel_width = 7
            self.kws = [1,1,1,1,3,3,5,7]
            self.kernels = nn.ModuleList(
                [
                    BatchConv1d(d_k, d_k, kw, use_mask=use_mask)
                    for kw in self.kws
                ]
            )
            self.use_mask = use_mask

        self.proj = XavierLinear(d_model, d_model, bias=True)

    def forward(self, q, k, v, attn_mask=None):

        B_q, L_q, d_model_q = q.size()
        B_k, L_k, d_model_k = k.size()
        B_v, L_v, d_model_v = v.size()
        assert B_k == B_v and L_k == L_v and d_model_k == d_model_v == self.d_model
        assert B_q == B_k and d_model_q == d_model_k == self.d_model
        if attn_mask is not None:
            _B, _L_q, _L_k = attn_mask.size()
            assert _B == B_q and _L_q == L_q and _L_k == L_k

        n_h, residual = self.n_head, q
        assert d_model_q % n_h == 0, 'd_model {} divided by n_head {}.'.format(d_model_q, n_head)

        def shape(x):
            return x.view(x.size(0), -1, self.n_head, self.dim_per_head).permute(0, 2, 1, 3)

        def unshape(x):
            return x.permute(0, 2, 1, 3).contiguous().view(x.size(0), -1, self.n_head * self.dim_per_head)

        # (B, L_q, d_model) -> (B, L_q, n_head*dim_per_head) -> (B, n_head, L_q, dim_per_head)
        q_s = shape(self.linear_q(q))
        k_s = shape(self.linear_k(k))
        v_s = shape(self.linear_v(v))

        '''
        q_s_r = q_s[:, :, :, None].repeat(1, 1, 1, self.kernel_width) # -> (n_head*B, L_q, L_k, kernel_width)
        q_s_r[:, :1, :, 1] = 0
        q_s_r[:, 1:, :, 1] = q_s_r[:, :-1, :, 0]
        q_s_r[:, :2, :, 2] = 0
        q_s_r[:, 2:, :, 2] = q_s_r[:, :-2, :, 0]
        '''

        if self.use_attcnn is True:
            '''
            k_s_mask = k_s.repeat(1, 1, 1)
            if self.use_mask is True and attn_mask is not None:   # (B, L_q, L_k)
                attn_mask_repeat = attn_mask.repeat(n_h, 1, 1) # -> (n_head*B, L_q, L_k)
                attn_mask_repeat = attn_mask_repeat.permute(0, 2, 1)[:,:,0][:,:,None] # -> (n_head*B, L_k, 1)
                attn_mask_repeat = attn_mask_repeat.repeat(1, 1, self.d_k) # -> (n_head*B, L_k, d_k)
                k_s_mask.data.masked_fill_(attn_mask_repeat, 0.0)
            else:
                pass
            '''
            '''
            attn = self.bconv1d(q_s.view(B_q*n_h, L_q, d_model_q/n_h),
                                k_s.view(B_k*n_h, L_k, d_model_k/n_h)
                                ).view(n_h, B_k, L_q, L_k) / self.temper
            '''
            attns = list(range(n_h))
            for i in range(n_h):
                attns[i] = self.kernels[i](q_s[:, i, :, :], k_s[:, i, :, :])
            attn = tc.stack(attns, dim=1) / self.temper

        else:
            q_s = q_s / self.temper
            # (B, n_head, L_q, dim_per_head) * (B, n_head, dim_per_head, L_k)
            attn = tc.matmul(q_s, k_s.permute(0, 1, 3, 2))  # (B, n_head, L_q, L_k)

        if attn_mask is not None:   # (B, L_q, L_k)
            attn_mask = tc.stack([attn_mask for k in range(n_h)], dim=1) # -> (B, n_head, L_q, L_k)
            assert attn_mask.size() == attn.size(), 'Attention mask shape {} mismatch ' \
                    'with Attention logit tensor shape {}.'.format(attn_mask.size(), attn.size())
            attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.mSoftMax(attn, dim=-1)
        # one attention
        one_head_attn = attn[:, 0, :, :].contiguous()

        attn = self.dropout(attn)   # (B, n_head, L_q, L_k) note: L_k == L_v
        output = tc.matmul(attn, v_s)  # (B, n_head, L_q, dim_per_head)
        output = unshape(output)
        output = self.proj(output)          # (B_q, L_q, d_model)

        return output, attn, one_head_attn

class PositionwiseFeedForward(nn.Module):
    '''
        A two-layer Feed-Forward Network
        Args:
            size(int): the size of input for the first-layer of the FFN.
            hidden_size(int): the hidden layer size of the second-layer
                              of the FNN.
            droput(float): dropout probability(0-1.0).
    '''
    def __init__(self, d_hid, d_inner_hid, dropout=0.1):

        super(PositionwiseFeedForward, self).__init__()
        self.linear_1 = XavierLinear(d_hid, d_inner_hid, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.linear_2 = XavierLinear(d_inner_hid, d_hid, bias=True)

    def forward(self, x):
        # x: (B_q, L_q, d_model)
        tmp = self.dropout(self.relu(self.linear_1(x)))

        return self.linear_2(tmp)

class EncoderLayer(nn.Module):
    '''
        Args:
            size(int): the dimension of keys/values/queries in
                       MultiHeadAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
            droput(float): dropout probability(0-1.0).
            n_head(int): the number of head for MultiHeadAttention.
            hidden_size(int): the second-layer of the PositionwiseFeedForward.
    '''
    def __init__(self, d_model, n_head=8, d_k=64, d_v=64, d_inner_hid=2048, dropout=0.1,
                 use_attcnn=False):

        super(EncoderLayer, self).__init__()
        self.ln_1 = Layer_Norm(d_model)
        self.src_slf_attn = MultiHeadAttention(d_model, n_head, d_k, d_v, dropout=dropout,
                                               use_attcnn=use_attcnn)
        self.dropout_1 = nn.Dropout(dropout, inplace=True)
        self.ln_2 = Layer_Norm(d_model)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
        self.dropout_2 = nn.Dropout(dropout, inplace=True)

    def forward(self, enc_input, slf_attn_mask=None):

        enc_input_norm = self.ln_1(enc_input)    # n
        # q - k - v
        enc_output, enc_slf_attn, enc_slf_one_attn = self.src_slf_attn(
            enc_input_norm, enc_input_norm, enc_input_norm, attn_mask=slf_attn_mask)
        ff_in = self.dropout_1(enc_output) + enc_input    # da

        ff_in_norm = self.ln_2(ff_in)   # n
        # enc_output: (B_q, L_q, d_model), enc_slf_attn: (B*n_head, L_q, L_k)
        ff_out = self.pos_ffn(ff_in_norm)
        enc_output = self.dropout_2(ff_out) + ff_in   # da

        return enc_output, enc_slf_attn, enc_slf_one_attn

''' A encoder model with self attention mechanism. '''
class Encoder(nn.Module):

    def __init__(self, n_src_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1, use_attcnn=False):

        super(Encoder, self).__init__()

        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model

        self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=PAD)
        self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=PAD)
        wlog('src position emb: {}'.format(self.position_enc.weight.data.size()))
        wlog('src emb: {}'.format(self.src_word_emb.weight.data.size()))

        self.dropout = nn.Dropout(dropout)
        #print 'src: ', n_src_vocab, n_position
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_k, d_v, d_inner_hid,
                         dropout=dropout, use_attcnn=use_attcnn)
            for _ in range(n_layers)])

        self.ln = Layer_Norm(d_model)

    def forward(self, src_seq, src_pos):

        B, L = src_seq.size()
        # Word embedding look up
        enc_output = self.src_word_emb(src_seq)
        # Position Encoding addition
        enc_output += self.position_enc(src_pos)
        enc_outputs, enc_slf_attns = [], []

        #src_slf_attn_mask = src_seq.data.ne(PAD).unsqueeze(1).expand(B, L, L)
        src_slf_attn_mask = src_seq.data.eq(PAD).unsqueeze(1).expand(B, L, L)
        #src_slf_attn_mask = get_attn_padding_mask(src_seq, src_seq)
        enc_output = self.dropout(enc_output)

        for enc_layer in self.layer_stack:
            # enc_output: (B_q, L_q, d_model), enc_slf_attn: (B*n_head, L_q, L_k)
            enc_output, enc_slf_attn, enc_slf_one_attn = enc_layer(enc_output, src_slf_attn_mask)
            enc_outputs += [enc_output]
            enc_slf_attns += [enc_slf_attn]

        enc_outputs[-1] = self.ln(enc_outputs[-1])

        return (enc_outputs, enc_slf_attns, enc_slf_one_attn)

class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, n_head, d_k=64, d_v=64, d_inner_hid=2048, dropout=0.1,
                 use_attcnn=False):

        super(DecoderLayer, self).__init__()
        self.ln_1 = Layer_Norm(d_model)
        self.trg_slf_attn = MultiHeadAttention(d_model, n_head, d_k, d_v, dropout=dropout,
                                               use_attcnn=use_attcnn, use_mask=True)
        self.dropout_1 = nn.Dropout(dropout, inplace=True)
        self.ln_2 = Layer_Norm(d_model)
        self.trg_src_attn = MultiHeadAttention(d_model, n_head, d_k, d_v, dropout=dropout,
                                               use_attcnn=use_attcnn)
        self.dropout_2 = nn.Dropout(dropout, inplace=True)
        self.ln_3 = Layer_Norm(d_model)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
        self.dropout_3 = nn.Dropout(dropout, inplace=True)

    def forward(self, dec_input, enc_output, trg_slf_attn_mask=None, trg_src_attn_mask=None):

        dec_input_norm = self.ln_1(dec_input)    # n
        # trg_slf_attn_mask: (B, trg_L, trg_L), trg_src_attn_mask: (B, trg_L, src_L)
        dec_output, dec_slf_attn, dec_slf_one_attn = self.trg_slf_attn(
            dec_input_norm, dec_input_norm, dec_input_norm, attn_mask=trg_slf_attn_mask)
        # (L_q, L_k, L_v) == (trg_L, trg_L, trg_L)
        # dec_output: (B_q, L_q, d_model) == (B, trg_L, d_model)
        # dec_slf_attn: (B*n_head, L_q, L_k) == (B*n_head, trg_L, trg_L)
        att_input = self.dropout_1(dec_output) + dec_input
        #x = self.dropout(dec_output) + x  # da

        att_input_norm = self.ln_2(att_input)   # n
        dec_output, dec_enc_attn, dec_enc_one_attn = self.trg_src_attn(
            att_input_norm, enc_output, enc_output, attn_mask=trg_src_attn_mask)
        # (L_q, L_k, L_v) == (trg_L, src_L, src_L)
        # dec_output: (B_q, L_q, d_model) == (B, trg_L, d_model)
        # dec_enc_attn: (B*n_head, L_q, L_k) == (B*n_head, trg_L, src_L)
        ff_in = self.dropout_2(dec_output) + att_input   # da

        ff_in_norm = self.ln_3(ff_in)   # n
        ff_out = self.pos_ffn(ff_in_norm)
        dec_output = self.dropout_3(ff_out) + ff_in # da

        return dec_output, dec_slf_attn, dec_enc_attn, dec_enc_one_attn

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''
    def __init__(self, n_tgt_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1, proj_share_weight=False,
                 use_attcnn=False):

        super(Decoder, self).__init__()
        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model

        self.position_enc = nn.Embedding(n_position + 2, d_word_vec, padding_idx=PAD)
        self.position_enc.weight.data = position_encoding_init(n_position + 2, d_word_vec)

        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec, padding_idx=PAD)
        wlog('trg position emb: {}'.format(self.position_enc.weight.data.size()))
        wlog('trg emb: {}'.format(self.tgt_word_emb.weight.data.size()))
        self.dropout = nn.Dropout(dropout)

        trg_lookup_table = self.tgt_word_emb if proj_share_weight is True else None
        self.classifier = Classifier(d_model, n_tgt_vocab, trg_lookup_table,
                                     trg_wemb_size=d_word_vec)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, n_head, d_k, d_v, d_inner_hid,
                         dropout=dropout, use_attcnn=use_attcnn)
            for _ in range(n_layers)])

        self.ln = Layer_Norm(d_model)

    #def forward(self, tgt_seq, tgt_pos, src_seq, enc_outputs):
    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output):

        src_B, src_L = src_seq.size()
        trg_B, trg_L = tgt_seq.size()
        #print trg_L, tgt_pos.size(-1)
        assert src_B == trg_B
        # Word embedding look up
        dec_out = self.tgt_word_emb(tgt_seq)
        dec_out += self.position_enc(tgt_pos)

        '''
        Get an attention mask to avoid using the subsequent info.
        array([[[0, 1, 1],
                [0, 0, 1],
                [0, 0, 0]]], dtype=uint8)
        '''
        trg_src_attn_mask = src_seq.data.eq(PAD).unsqueeze(1).expand(src_B, trg_L, src_L)

        trg_slf_attn_mask = tgt_seq.data.eq(PAD).unsqueeze(1).expand(trg_B, trg_L, trg_L)
        subsequent_mask = np.triu(np.ones((trg_B, trg_L, trg_L)), k=1).astype('uint8')
        subsequent_mask = tc.from_numpy(subsequent_mask)
        if tgt_seq.is_cuda: subsequent_mask = subsequent_mask.cuda()
        trg_slf_attn_mask = tc.gt(trg_slf_attn_mask + subsequent_mask, 0)
        # Decode
        #dec_slf_attn_pad_mask = get_attn_padding_mask(tgt_seq, tgt_seq)
        #dec_slf_attn_sub_mask = get_attn_subsequent_mask(tgt_seq)
        #trg_slf_attn_mask = tc.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)

        #trg_src_attn_mask = get_attn_padding_mask(tgt_seq, src_seq)
        # (mb_size, len_q, len_k)  len_q == len_k == len_trg
        # (mb_size, len_q, len_k)  len_q == len_trg, len_k == len_src
        dec_slf_attns, dec_enc_attns = [], []

        dec_out = self.dropout(dec_out)
        #for dec_layer, enc_output in zip(self.layer_stack, enc_outputs):
        for dec_layer in self.layer_stack:
            dec_out, dec_slf_attn, dec_enc_attn, dec_enc_one_attn = dec_layer(dec_out, enc_output,
                trg_slf_attn_mask=trg_slf_attn_mask, trg_src_attn_mask=trg_src_attn_mask)
            dec_slf_attns += [dec_slf_attn]
            dec_enc_attns += [dec_enc_attn]

        dec_out = self.ln(dec_out)

        return (dec_out, dec_slf_attns, dec_enc_attns, dec_enc_one_attn)

def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return tc.from_numpy(position_enc).type(tc.FloatTensor)


