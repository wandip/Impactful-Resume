import torch, time, argparse, os, codecs, h5py, random
import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from cpgn_utils_inpattern_length import *
import pickle as cPickle

# seq2seq w/ decoder attention
# transformation embeddings concatenated with decoder word inputs
# attention conditioned on transformation via bilinear product
class CPGN(nn.Module):
    def __init__(self, d_word, d_hid, d_nt_1, d_nt_2, d_trans_1, d_trans_2,
        len_voc, len_trans_voc_1, len_trans_voc_2, use_input_parse):

        super(CPGN, self).__init__()
        self.d_word = d_word
        self.d_hid = d_hid
        self.d_trans_1 = d_trans_1
        self.d_trans_2 = d_trans_2
        self.d_nt_1 = d_nt_1 + 1
        self.d_nt_2 = d_nt_2 + 1
        self.len_voc = len_voc
        self.len_trans_voc_1 = len_trans_voc_1
        self.len_trans_voc_2 = len_trans_voc_2
        self.use_input_parse = use_input_parse

        # embeddings
        self.word_embs = nn.Embedding(len_voc, d_word)
        self.trans_embs_1 = nn.Embedding(len_trans_voc_1, d_nt_1)
        self.trans_embs_2 = nn.Embedding(len_trans_voc_2, d_nt_2)

        # lstms
        if use_input_parse:
            self.encoder = nn.LSTM(d_word + d_trans_1 + d_trans_2, d_hid, num_layers=1, bidirectional=True, batch_first=True)
        else:
            self.encoder = nn.LSTM(d_word, d_hid, num_layers=1, bidirectional=True, batch_first=True)

        self.encoder_proj = nn.Linear(d_hid * 2, d_hid)
        self.decoder = nn.LSTM(d_word + d_hid + d_hid, d_hid, num_layers=2, batch_first=True)
        self.trans_encoder_1 = nn.LSTM(d_nt_1, d_trans_1, num_layers=1, batch_first=True)
        self.trans_hid_init_1 = Variable(torch.zeros(1, 1, d_trans_1).cuda())
        self.trans_cell_init_1 = Variable(torch.zeros(1, 1, d_trans_1).cuda())
        self.trans_encoder_2 = nn.LSTM(d_nt_2, d_trans_2, num_layers=1, batch_first=True)
        self.trans_hid_init_2 = Variable(torch.zeros(1, 1, d_trans_2).cuda())
        self.trans_cell_init_2 = Variable(torch.zeros(1, 1, d_trans_2).cuda())
        self.e_hid_init = Variable(torch.zeros(2, 1, d_hid).cuda())
        self.e_cell_init = Variable(torch.zeros(2, 1, d_hid).cuda())
        self.d_cell_init = Variable(torch.zeros(2, 1, d_hid).cuda())

        # output softmax
        self.out_dense_1 = nn.Linear(d_hid * 2, d_hid)
        self.out_dense_2 = nn.Linear(d_hid, len_voc)
        self.att_nonlin = nn.Softmax()
        self.out_nonlin = nn.LogSoftmax()

        # attention params
        self.att_parse_proj_1 = nn.Linear(d_trans_1, d_hid)
        self.att_parse_proj_2 = nn.Linear(d_trans_2, d_hid)
        self.att_W = nn.Parameter(torch.Tensor(d_hid, d_hid).cuda())
        self.att_parse_W = nn.Parameter(torch.Tensor(d_hid, d_hid).cuda())
        nn.init.xavier_uniform(self.att_parse_W.data)
        nn.init.xavier_uniform(self.att_W.data)

        # copy prob params
        self.copy_hid_v = nn.Parameter(torch.Tensor(d_hid, 1).cuda())
        self.copy_att_v = nn.Parameter(torch.Tensor(d_hid, 1).cuda())
        self.copy_inp_v = nn.Parameter(torch.Tensor(d_word + d_hid + d_hid, 1).cuda())
        nn.init.xavier_uniform(self.copy_hid_v.data)
        nn.init.xavier_uniform(self.copy_att_v.data)
        nn.init.xavier_uniform(self.copy_inp_v.data)


    # create matrix mask from length vector
    def compute_mask(self, lengths):
        max_len = torch.max(lengths)
        range_row = torch.arange(0, max_len).long().cuda()[None, :].expand(lengths.size()[0], max_len)
        mask = lengths[:, None].expand_as(range_row)
        mask = range_row < mask
        return Variable(mask.float().cuda())


    # masked softmax for attention
    def masked_softmax(self, vector, mask):
        result = torch.nn.functional.softmax(vector)
        result = result * mask
        result = result / (result.sum(dim=1, keepdim=True) + 1e-13)
        return result


    # compute masked attention over enc hiddens with bilinear product
    def compute_decoder_attention(self, hid_previous, enc_hids, in_lens):
        mask = self.compute_mask(in_lens)
        b_hn = hid_previous.mm(self.att_W)
        scores = b_hn[:, None, :] * enc_hids
        scores = torch.sum(scores, 2)
        scores = self.masked_softmax(scores, mask)
        return scores


    # compute masked attention over parse sequence with bilinear product
    def compute_transformation_attention(self, hid_previous, trans_embs, trans_lens):

        mask = self.compute_mask(trans_lens)
        b_hn = hid_previous.mm(self.att_parse_W)
        scores = b_hn[:, None, :] * trans_embs
        scores = torch.sum(scores, 2)
        scores = self.masked_softmax(scores, mask)
        return scores


    # return encoding for an input batch
    def encode_batch(self, inputs, trans, lengths): #TODO: It's ok that we have one implementation becasue trans is not used

        bsz, max_len = inputs.size()
        in_embs = self.word_embs(inputs)
        lens, indices = torch.sort(lengths, 0, True)

        # concat word embs with trans hid
        if self.use_input_parse:
            in_embs = torch.cat([in_embs, trans.unsqueeze(1).expand(bsz, max_len, self.d_trans)], 2)

        e_hid_init = self.e_hid_init.expand(2, bsz, self.d_hid).contiguous()
        e_cell_init = self.e_cell_init.expand(2, bsz, self.d_hid).contiguous()
        all_hids, (enc_last_hid, _) = self.encoder(pack(in_embs[indices],
                    lens.tolist(), batch_first=True), (e_hid_init, e_cell_init))
        _, _indices = torch.sort(indices, 0)
        all_hids = unpack(all_hids, batch_first=True)[0][_indices]
        all_hids = self.encoder_proj(all_hids.view(-1, self.d_hid * 2)).view(bsz, max_len, self.d_hid)

        enc_last_hid = torch.cat([enc_last_hid[0], enc_last_hid[1]], 1)
        enc_last_hid = self.encoder_proj(enc_last_hid)[_indices]

        return all_hids, enc_last_hid

    # return encoding for an input batch
    def encode_transformations_1(self, trans, lengths, return_last=True):
        bsz, _ = trans.size()
        # print 'bsz: {}'.format(bsz)

        lens, indices = torch.sort(lengths, 0, True)
        in_embs = self.trans_embs_1(trans)

        t_hid_init = self.trans_hid_init_1.expand(1, bsz, self.d_trans_1).contiguous()
        t_cell_init = self.trans_cell_init_1.expand(1, bsz, self.d_trans_1).contiguous()
        all_hids, (enc_last_hid, _) = self.trans_encoder_1(pack(in_embs[indices],
            lens.tolist(), batch_first=True), (t_hid_init, t_cell_init))
        _, _indices = torch.sort(indices, 0)

        if return_last:
            return enc_last_hid.squeeze(0)[_indices]
        else:
            all_hids = unpack(all_hids, batch_first=True)[0]
            return all_hids[_indices]

    def encode_transformations_2(self, trans, lengths, return_last=True):
            bsz, _ = trans.size()
            # print 'bsz: {}'.format(bsz)

            lens, indices = torch.sort(lengths, 0, True)
            in_embs = self.trans_embs_2(trans)

            t_hid_init = self.trans_hid_init_2.expand(1, bsz, self.d_trans_2).contiguous()
            t_cell_init = self.trans_cell_init_2.expand(1, bsz, self.d_trans_2).contiguous()
            all_hids, (enc_last_hid, _) = self.trans_encoder_2(pack(in_embs[indices],
                lens.tolist(), batch_first=True), (t_hid_init, t_cell_init))
            _, _indices = torch.sort(indices, 0)

            if return_last:
                return enc_last_hid.squeeze(0)[_indices]
            else:
                all_hids = unpack(all_hids, batch_first=True)[0]
                return all_hids[_indices]

    # decode one timestep
    def decode_step(self, idx, prev_words, prev_hid, prev_cell,
        enc_hids, trans_embs_1, trans_embs_2, in_sent_lens, trans_lens_1, trans_lens_2, bsz, max_len):

        # initialize with zeros
        if idx == 0:
            word_input = Variable(torch.zeros(bsz, 1, self.d_word).cuda())

        # get previous ground truth word embed and concat w/ transformation emb
        else:
            word_input = self.word_embs(prev_words)
            word_input = word_input.view(bsz, 1, self.d_word)

        trans_weights_1 = self.compute_transformation_attention(prev_hid[1], trans_embs_1, trans_lens_1)
        trans_ctx_1 = torch.sum(trans_weights_1[:, :, None] * trans_embs_1, 1)
        trans_weights_2 = self.compute_transformation_attention(prev_hid[1], trans_embs_2, trans_lens_2)
        trans_ctx_2 = torch.sum(trans_weights_2[:, :, None] * trans_embs_2, 1)

        decoder_input = torch.cat([word_input, trans_ctx_1.unsqueeze(1), trans_ctx_2.unsqueeze(1)], 2)

        # feed to decoder lstm
        _, (hn, cn) = self.decoder(decoder_input, (prev_hid, prev_cell))

        # compute attention for next time step and att weighted ave of encoder hiddens
        attn_weights = self.compute_decoder_attention(hn[1], enc_hids, in_sent_lens)
        attn_ctx = torch.sum(attn_weights[:, :, None] * enc_hids, 1)

        # compute copy prob as function of lotsa shit
        p_copy = decoder_input.squeeze(1).mm(self.copy_inp_v)
        p_copy += attn_ctx.mm(self.copy_att_v)
        p_copy += hn[1].mm(self.copy_hid_v)
        p_copy = torch.sigmoid(p_copy).squeeze(1)

        return hn, cn, attn_weights, attn_ctx, p_copy

    def forward(self, inputs, outputs, in_trans_1, in_trans_2, out_trans_1, out_trans_2,
        in_sent_lens, out_sent_lens, in_trans_lens_1, in_trans_lens_2, out_trans_lens_1, out_trans_lens_2, max_decode):
        bsz, max_len = inputs.size()

        # encode transformations
        in_trans_hids = None
        if self.use_input_parse:
            in_trans_hids = self.encode_transformations_1(in_trans_1, in_trans_lens_1)

        out_trans_hids_1 = self.encode_transformations_1(out_trans_1, out_trans_lens_1, return_last=False)
        out_trans_hids_1 = self.att_parse_proj_1(out_trans_hids_1)

        out_trans_hids_2 = self.encode_transformations_2(out_trans_2, out_trans_lens_2, return_last=False)
        out_trans_hids_2 = self.att_parse_proj_2(out_trans_hids_2)

        # encode input sentence
        enc_hids, enc_last_hid = self.encode_batch(inputs, in_trans_hids, in_sent_lens)

        # store decoder hiddens and attentions for copy
        decoder_states = Variable(torch.zeros(max_decode, bsz, self.d_hid * 2).cuda())
        decoder_copy_dists = Variable(torch.zeros(max_decode, bsz, self.len_voc).cuda())
        copy_probs = Variable(torch.zeros(max_decode, bsz).cuda())

        # initialize decoder hidden to last encoder hidden
        hn = enc_last_hid.unsqueeze(0).expand(2, bsz, self.d_hid).contiguous()
        cn = self.d_cell_init.expand(2, bsz, self.d_hid).contiguous()

        # loop til max_decode, do lstm tick using previous prediction
        for idx in range(max_decode):
            prev_words = None
            if idx > 0:
                prev_words = outputs[:, idx - 1]

            # concat prev word emb and trans emb and feed to decoder lstm
            hn, cn, attn_weights, attn_ctx, p_copy = self.decode_step(idx, prev_words,
                hn, cn, enc_hids, out_trans_hids_1, out_trans_hids_2, in_sent_lens, out_trans_lens_1, out_trans_lens_2, bsz, max_len)

            # compute copy attn by scattering attn into vocab space
            vocab_scores = Variable(torch.zeros(bsz, self.len_voc).cuda())
            vocab_scores = vocab_scores.scatter_add_(1, inputs, attn_weights)

            # store decoder hiddens and copy probs in log domain
            decoder_states[idx] = torch.cat([hn[1], attn_ctx], 1)
            decoder_copy_dists[idx] = torch.log(vocab_scores + 1e-20)
            copy_probs[idx] = p_copy

        # now do prediction over decoder states (reshape to 2d first)
        decoder_states = decoder_states.transpose(0, 1).contiguous().view(-1, self.d_hid * 2)
        decoder_preds = self.out_dense_1(decoder_states)
        decoder_preds = self.out_dense_2(decoder_preds)
        decoder_preds = self.out_nonlin(decoder_preds)
        decoder_copy_dists = decoder_copy_dists.transpose(0, 1).contiguous().view(-1, self.len_voc)
        copy_probs = copy_probs.view(-1)
        final_dists = copy_probs[:, None] * decoder_copy_dists + \
            (1. - copy_probs[:, None]) * decoder_preds
        return final_dists

    # beam search given a single input / transformation
    def beam_search(self, inputs, in_trans, out_trans_1, out_trans_2, in_sent_lens, in_trans_lens,
        out_trans_lens_1, out_trans_lens_2, eos_idx, beam_size=4, max_steps=40):
        bsz, max_len = inputs.size()

        # chop input
        inputs = inputs[:, :in_sent_lens[0]]

        # encode transformations
        in_trans_hids = None
        if self.use_input_parse:
            in_trans_hids = self.encode_transformations_1(in_trans, in_trans_lens)
        out_trans_hids_1 = self.encode_transformations_1(out_trans_1, out_trans_lens_1, return_last=False)
        out_trans_hids_1 = self.att_parse_proj_1(out_trans_hids_1)

        out_trans_hids_2 = self.encode_transformations_2(out_trans_2, out_trans_lens_2, return_last=False)
        out_trans_hids_2 = self.att_parse_proj_2(out_trans_hids_2)

        # encode input sentence
        enc_hids, enc_last_hid = self.encode_batch(inputs, in_trans_hids, in_sent_lens)

        # initialize decoder hidden to last encoder hidden
        hn = enc_last_hid.unsqueeze(0).expand(2, bsz, self.d_hid).contiguous()
        cn = self.d_cell_init.expand(2, bsz, self.d_hid).contiguous()

        # initialize beams
        beams = [(0.0, hn, cn, [])]
        nsteps = 0

        # loop til max_decode, do lstm tick using previous prediction
        while True:

            # loop over everything in the beam
            beam_candidates = []
            for b in beams:
                curr_prob, prev_h, prev_c, seq = b

                # start with last word in sequence, if eos end the beam
                if len(seq) > 0:
                    prev_word = seq[-1]
                    if prev_word == eos_idx:
                        beam_candidates.append(b)
                        continue

                    # load into torch var so we can do decoding
                    prev_word = Variable(torch.LongTensor([prev_word]).cuda())

                else:
                    prev_word = None

                # concat prev word emb and prev attn input and feed to decoder lstm
                hn, cn, attn_weights, attn_ctx, p_copy = self.decode_step(len(seq), prev_word,
                    prev_h, prev_c, enc_hids, out_trans_hids_1, out_trans_hids_2, in_sent_lens, out_trans_lens_1, out_trans_lens_2, bsz, max_len)

                # compute copy attn by scattering attn into vocab space
                vocab_scores = Variable(torch.zeros(bsz, self.len_voc).cuda())
                vocab_scores = vocab_scores.scatter_add_(1, inputs, attn_weights)
                vocab_scores = torch.log(vocab_scores + 1e-20).squeeze()

                # compute prediction over vocab for a single time step
                pred_inp = torch.cat([hn[1], attn_ctx], 1)
                preds = self.out_dense_1(pred_inp)
                preds = self.out_dense_2(preds)
                preds = self.out_nonlin(preds).squeeze()

                final_preds = p_copy * vocab_scores + (1 - p_copy) * preds

                # sort in descending order (log domain)
                _, top_indices = torch.sort(-final_preds)

                # add top n candidates
                for z in range(beam_size):
                    word_idx = top_indices[z].data.item()
                    # word_idx = top_indices[z].data[0]
                    beam_candidates.append((curr_prob + final_preds[word_idx].data.item(),
                        hn, cn, seq + [word_idx]))

            beam_candidates.sort(reverse=True)
            beams = beam_candidates[:beam_size]

            nsteps += 1
            if nsteps > max_steps:
                break

        return beams

    def batch_beam_search(self, inputs, out_trans_1, out_trans_2, in_sent_lens,
        out_trans_lens_1, out_trans_lens_2, eos_idx, beam_size=5, max_steps=70):
        bsz, max_len = inputs.size()

        # chop input
        inputs = inputs[:, :in_sent_lens[0]]

        # encode transformations
        out_trans_hids_1 = self.encode_transformations_1(out_trans_1, out_trans_lens_1, return_last=False)
        out_trans_hids_1 = self.att_parse_proj_1(out_trans_hids_1)

        out_trans_hids_2 = self.encode_transformations_2(out_trans_2, out_trans_lens_2, return_last=False)
        out_trans_hids_2 = self.att_parse_proj_2(out_trans_hids_2)

        # encode input sentence
        enc_hids, enc_last_hid = self.encode_batch(inputs, None, in_sent_lens)

        # initialize decoder hidden to last encoder hidden
        hn = enc_last_hid.unsqueeze(0).expand(2, bsz, self.d_hid).contiguous()
        cn = self.d_cell_init

        beam_dict = OrderedDict()
        for b_idx in range(out_trans_1.size()[0]):
            beam_dict[b_idx] = [(0.0, hn, cn, [])]
        nsteps = 0

        # loop til max_decode, do lstm tick using previous prediction
        while True:

            prev_words = []
            prev_hs = []
            prev_cs = []

            for b_idx in beam_dict:

                beams = beam_dict[b_idx]
                # loop over everything in the beam
                beam_candidates = []
                for b in beams:
                    curr_prob, prev_h, prev_c, seq = b

                    # start with last word in sequence, if eos end the beam
                    if len(seq) > 0:
                        prev_words.append(seq[-1])

                    else:
                        prev_words = None

                    prev_hs.append(prev_h)
                    prev_cs.append(prev_c)

            # now batch decoder computations
            hs = torch.cat(prev_hs, 1)
            cs = torch.cat(prev_cs, 1)
            num_examples = hs.size()[1]
            if prev_words is not None:
                prev_words = Variable(torch.from_numpy(np.array(prev_words, dtype='int32')).long().cuda())

            # expand out parse states if necessary
            if num_examples != out_trans_hids_1.size()[0]:
                d1, d2, d3 = out_trans_hids_1.size()

                rep_factor = int(num_examples / d1)
                # rep_factor = num_examples / d1
                curr_out_1 = out_trans_hids_1.unsqueeze(1).expand(d1, rep_factor,
                    d2, d3).contiguous().view(-1, d2, d3)
                curr_out_lens_1 = out_trans_lens_1.unsqueeze(1).expand(d1, rep_factor).contiguous().view(-1)

            else:
                curr_out_1 = out_trans_hids_1
                curr_out_lens_1 = out_trans_lens_1

            if num_examples != out_trans_hids_2.size()[0]:
                d1, d2, d3 = out_trans_hids_2.size()
                rep_factor = int(num_examples / d1)

                curr_out_2 = out_trans_hids_2.unsqueeze(1).expand(d1, rep_factor,
                    d2, d3).contiguous().view(-1, d2, d3)

                curr_out_lens_2 = out_trans_lens_2.unsqueeze(1).expand(d1, rep_factor).contiguous().view(-1)

            else:
                curr_out_2 = out_trans_hids_2
                curr_out_lens_2 = out_trans_lens_2

            # expand out inputs and encoder hiddens
            _, in_len, hid_d = enc_hids.size()

            curr_enc_hids = enc_hids.expand(num_examples, in_len, hid_d)

            curr_enc_lens = in_sent_lens.expand(num_examples)
            curr_inputs = inputs.expand(num_examples, in_sent_lens[0])

            # concat prev word emb and prev attn input and feed to decoder lstm
            hn, cn, attn_weights, attn_ctx, p_copy = self.decode_step(nsteps, prev_words,
                hs, cs, curr_enc_hids, curr_out_1, curr_out_2, curr_enc_lens, curr_out_lens_1, curr_out_lens_2,
                                                                      num_examples, max_len)

            # compute copy attn by scattering attn into vocab space
            vocab_scores = Variable(torch.zeros(num_examples, self.len_voc).cuda())
            vocab_scores = vocab_scores.scatter_add_(1, curr_inputs, attn_weights)
            vocab_scores = torch.log(vocab_scores + 1e-20).squeeze()

            # compute prediction over vocab for a single time step
            pred_inp = torch.cat([hn[1], attn_ctx], 1)
            preds = self.out_dense_1(pred_inp)
            preds = self.out_dense_2(preds)
            preds = self.out_nonlin(preds).squeeze()
            final_preds = p_copy[:, None] * vocab_scores + (1 - p_copy[:, None]) * preds

            # now loop over the examples and sort each separately
            for b_idx in beam_dict:
                beam_candidates = []

                # no words previously predicted
                if num_examples == len(beam_dict):
                    ex_hn = hn[:,b_idx,:].unsqueeze(1)
                    ex_cn = cn[:,b_idx,:].unsqueeze(1)
                    preds = final_preds[b_idx]
                    _, top_indices = torch.sort(-preds)
                    # add top n candidates

                    for z in range(beam_size):
                        word_idx = top_indices[z].data.item()
                        # word_idx = top_indices[z].data[0]
                        beam_candidates.append((preds[word_idx].item(), ex_hn, ex_cn, [word_idx]))
                        # beam_candidates.append((preds[word_idx].data[0], ex_hn, ex_cn, [word_idx]))
                    beam_dict[b_idx] = beam_candidates
                else:
                    origin_beams = beam_dict[b_idx]
                    start = b_idx * beam_size
                    end = (b_idx + 1) * beam_size
                    ex_hn = hn[:,start:end,:]
                    ex_cn = cn[:,start:end,:]
                    ex_preds = final_preds[start:end]

                    for o_idx, ob in enumerate(origin_beams):
                        curr_prob, _, _, seq = ob

                        # if one of the beams is already complete, add it to candidates
                        if seq[-1] == eos_idx:
                            beam_candidates.append(ob)

                        preds = ex_preds[o_idx]
                        curr_hn = ex_hn[:,o_idx,:]
                        curr_cn = ex_cn[:,o_idx,:]
                        _, top_indices = torch.sort(-preds)
                        for z in range(beam_size):
                            word_idx = top_indices[z].data.item()
                            # word_idx = top_indices[z].data[0]

                            beam_candidates.append((curr_prob + float(preds[word_idx].cpu().data.item()),
                            # beam_candidates.append((curr_prob + float(preds[word_idx].cpu().data[0]),
                                curr_hn.unsqueeze(1), curr_cn.unsqueeze(1), seq + [word_idx]))

                    s_inds = np.argsort([x[0] for x in beam_candidates])[::-1]
                    beam_candidates = [beam_candidates[x] for x in s_inds]
                    beam_dict[b_idx] = beam_candidates[:beam_size]

            nsteps += 1
            if nsteps > max_steps:
                break

        return beam_dict

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Syntactically Controlled Paraphrase Network')
    parser.add_argument('--gpu', type=str, default='0',
            help='GPU id')
    parser.add_argument('--data', type=str, default='data/parsed_data.h5',
            help='hdf5 location')
    parser.add_argument('--vocab', type=str, default='data/parse_vocab.pkl',
            help='word vocabulary')
    parser.add_argument('--parse_vocab_2', type=str, default='data/unique_lenght.txt',
            help='tag vocabulary 2')
    parser.add_argument('--model', type=str, default='CPGN2_L_pattern_new_4k.pt',
            help='model save path')
    parser.add_argument('--batch_size', type=int, default=64,
            help='batch size')
    parser.add_argument('--min_sent_length', type=int, default=5,
            help='min number of tokens per batch')
    parser.add_argument('--d_word', type=int, default=300,
            help='word embedding dimension')
    parser.add_argument('--d_trans', type=int, default=128,
            help='transformation hidden dimension')
    parser.add_argument('--d_nt', type=int, default=56,
            help='nonterminal embedding dimension')
    parser.add_argument('--d_hid', type=int, default=512,
            help='lstm hidden dimension')
    parser.add_argument('--n_epochs', type=int, default=5,
            help='n_epochs')
    parser.add_argument('--lr', type=float, default=0.00005,
            help='learning rate')
    parser.add_argument('--grad_clip', type=float, default=5.0,
            help='clip if grad norm exceeds this threshold')
    parser.add_argument('--save_freq', type=int, default=500,
            help='how many minibatches to save model')
    parser.add_argument('--lr_decay_factor', type=int, default=0.5,
            help='how much to decrease LR every epoch')
    parser.add_argument('--eval_mode', type=bool, default=False,
            help='run beam search for some examples using a trained model')
    parser.add_argument('--init_trained_model', type=int, default=0,
            help='continue training a cached model')
    parser.add_argument('--tree_dropout', type=float, default=0.,
            help='dropout rate for dropping tree terminals')
    parser.add_argument('--tree_level_dropout', type=float, default=0.,
            help='dropout rate for dropping entire levels of a tree')
    parser.add_argument('--short_batch_downsampling_freq', type=float, default=0.0,
            help='dropout rate for dropping entire levels of a tree')
    parser.add_argument('--short_batch_threshold', type=int, default=20,
            help='if sentences are shorter than this, they will be downsampled')
    parser.add_argument('--seed', type=int, default=1000,
            help='random seed')
    parser.add_argument('--use_input_parse', type=int, default=0,
            help='whether or not to use the input parse')
    parser.add_argument('--dev_batches', type=int, default=200,
            help='how many minibatches to use for validation')

    args = parser.parse_args()

    batch_size = args.batch_size
    d_word = args.d_word
    d_hid = args.d_hid
    d_trans_1 = args.d_trans
    d_nt_1 = args.d_nt
    d_trans_2 = 64
    d_nt_2 = 40
    n_epochs = args.n_epochs
    lr = args.lr
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    out_file = 'models/' + args.model
    std_f = open('models/logs/std_' + args.model + '.txt', 'w')

    # set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # load data, word vocab, and parse vocab
    h5f = h5py.File(args.data, 'r')
    inp = h5f['inputs']
    out = h5f['outputs']
    in_lens = h5f['in_lengths']
    out_lens = h5f['out_lengths']

    vocab, rev_vocab = \
        cPickle.load(open(args.vocab, 'rb'))
    rev_vocab[len(rev_vocab)] = '___'
    vocab['___'] = len(vocab)

    label_voc_1 = vocab
    rev_label_voc_1 = rev_vocab

    tag_file_2 = codecs.open(args.parse_vocab_2, 'r', 'utf-8')
    label_voc_2 = {}
    for idx, line in enumerate(tag_file_2):
        line = line.strip()
        # if line != 'EOP':
        label_voc_2[line] = idx
    rev_label_voc_2 = dict((v,k) for (k,v) in label_voc_2.items())

    len_voc = len(vocab)
    len_parse_voc_1 = len(label_voc_1)
    len_parse_voc_2 = len(label_voc_2)
    max_decode = inp.shape[1]
    minibatches = [(start, start + batch_size) \
        for start in range(0, inp.shape[0], batch_size)][:-1]

    random.Random(7).shuffle(minibatches)

    dev_minibatches = minibatches[:args.dev_batches]
    test_minibatches = minibatches[args.dev_batches:args.dev_batches+100]
    train_minibatches = minibatches[args.dev_batches+100:400000]
    print ('train_minibatches: {}\t dev_minibatches: {}\ttest_minibatches: {}'.format(len(train_minibatches), len(dev_minibatches), len(test_minibatches)))

    # build network
    net = CPGN(d_word, d_hid, d_nt_1, d_nt_2, d_trans_1, d_trans_2,
        len_voc, len_parse_voc_1, len_parse_voc_2, args.use_input_parse)
    net.cuda()

    # load saved model if evaluating
    if args.eval_mode:
        saved_model = torch.load(out_file)
        net.load_state_dict(saved_model['state_dict'])
        train_minibatches = saved_model['trained_minibatches']
        dev_batches = saved_model['dev_minibatches']
        net.eval()

    if args.init_trained_model:
        print ('starting from cached model')
        net.load_state_dict(torch.load(out_file)['state_dict'])

    # ignore zero targets in loss function (they are just padding)
    criterion = nn.NLLLoss(ignore_index=0)
    params = net.parameters()
    optimizer = optim.Adam(params, lr=lr)

    if args.init_trained_model:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * args.lr_decay_factor / 2.
            print ('new LR:', param_group['lr'])

    for ep in range(n_epochs):
        random.shuffle(train_minibatches)
        ep_loss = 0.
        start_time = time.time()
        num_batches = 0

        for b_idx, (start, end) in enumerate(train_minibatches):
            # read data from hdf5
            in_p_l = in_lens[start:end]
            out_p_l = out_lens[start:end]

            in_p_2 = []
            out_p_2 = []
            for i in range(len(in_p_l)):
                in_p_2.append(str(in_p_l[i]))
                out_p_2.append(str(out_p_l[i]))

            new_in_p = []
            new_out_p = []

            inp_train = inp[start:end]
            out_train = out[start:end]

            for i in range(len(inp_train)):
                input_i = inp_train[i]
                output_i = out_train[i]
                flag_j = False
                flag_k = False
                for j in range(out_p_l[i]):
                    if output_i[j] not in input_i:
                        flag_j = True
                        for k in range(j+1, out_p_l[i]):
                            if output_i[k] not in input_i:
                                flag_k = True

                                if j == 0:
                                    if k == j+1:
                                        res = rev_vocab[output_i[j]] + ' ' + rev_vocab[output_i[k]] + ' ___'
                                    else:
                                        if k == out_p_l[i] - 1:
                                            res = rev_vocab[output_i[j]] + ' ___ ' + rev_vocab[output_i[k]]
                                        else:
                                            res = rev_vocab[output_i[j]] + ' ___ ' + rev_vocab[output_i[k]] + ' ___'
                                else:
                                    if k == j+1 and k == out_p_l[i] - 1:
                                        res = '___ ' + rev_vocab[output_i[j]] + ' ' + rev_vocab[output_i[k]]
                                    elif k == j+1:
                                        res = '___ ' + rev_vocab[output_i[j]] + ' ' + rev_vocab[output_i[k]] + ' ___'
                                    else:
                                        if k == out_p_l[i] - 1:
                                            res = '___ ' + rev_vocab[output_i[j]] + ' ___ ' + rev_vocab[output_i[k]]
                                        else:
                                            res = '___ ' + rev_vocab[output_i[j]] + ' ___ ' + rev_vocab[output_i[k]] + ' ___'
                                break

                        if not flag_k:
                            if j==0:
                                res = rev_vocab[output_i[j]] + ' ___'
                            elif j == out_p_l[i]-1:
                                res = '___ ' + rev_vocab[output_i[j]]
                            else:
                                res = '___ ' + rev_vocab[output_i[j]] + ' ___'
                        break

                if not flag_j:
                    res = '___'

                new_out_p.append(str(res))

                flag_j = False
                flag_k = False
                for j in range(in_p_l[i]):
                    if input_i[j] not in output_i:
                        flag_j = True
                        for k in range(j+1, in_p_l[i]):
                            if output_i[k] not in input_i:
                                flag_k = True

                                if j == 0:
                                    if k == j+1:
                                        res = rev_vocab[input_i[j]] + ' ' + rev_vocab[input_i[k]] + ' ___'
                                    else:
                                        if k == in_p_l[i] - 1:
                                            res = rev_vocab[input_i[j]] + ' ___ ' + rev_vocab[input_i[k]]
                                        else:
                                            res = rev_vocab[input_i[j]] + ' ___ ' + rev_vocab[input_i[k]] + ' ___'
                                else:
                                    if k == j+1 and k == in_p_l[i] - 1:
                                        res = '___ ' + rev_vocab[input_i[j]] + ' ' + rev_vocab[input_i[k]]
                                    if k == j+1:
                                        res = '___ ' + rev_vocab[input_i[j]] + ' ' + rev_vocab[input_i[k]] + ' ___'
                                    else:
                                        if k == in_p_l[i] - 1:
                                            res = '___ ' + rev_vocab[input_i[j]] + ' ___ ' + rev_vocab[input_i[k]]
                                        else:
                                            res = '___ ' + rev_vocab[input_i[j]] + ' ___ ' + rev_vocab[input_i[k]] + ' ___'
                                break

                        if not flag_k:
                            if j==0:
                                res = rev_vocab[input_i[j]] + ' ___'
                            elif j == in_p_l[i]-1:
                                res = '___ ' + rev_vocab[input_i[j]]
                            else:
                                res = '___ ' + rev_vocab[input_i[j]] + ' ___'
                        break

                if not flag_j:
                    res = '___'

                new_in_p.append(str(res))


            # get valid instances of transformations
            z = indexify_transformations(new_in_p, new_out_p, in_p_2, out_p_2, label_voc_1, label_voc_2, args)
            if z == None:
                continue

            in_trans_np_1, in_trans_np_2, out_trans_np_1, out_trans_np_2, mismatch_inds, in_trans_len_np_1, in_trans_len_np_2, \
            out_trans_len_np_1, out_trans_len_np_2 = z

            # only store valid input instances
            inp_np = inp[start:end][mismatch_inds]
            out_np = out[start:end][mismatch_inds]
            in_len_np = in_lens[start:end][mismatch_inds]
            out_len_np = out_lens[start:end][mismatch_inds]
            curr_bsz = inp_np.shape[0]

            # chop input based on length of last instance (for encoder efficiency)
            max_in_len = int(in_len_np[-1])
            inp_np = inp_np[:, :max_in_len]

            # compute max output length and chop output (for decoder efficiency)
            max_out_len = int(np.amax(out_len_np))
            out_np = out_np[:, :max_out_len]

            # sentences are too short
            if max_in_len < args.min_sent_length:
                continue

            # downsample if input sentences are too short
            if args.short_batch_downsampling_freq > 0. and max_in_len < args.short_batch_threshold:
                if np.random.rand() < args.short_batch_downsampling_freq:
                    continue

            # randomly invert 50% of batches (to remove NMT bias)
            swap = random.random() > 0.5
            if swap:
                inp_np, out_np = out_np, inp_np
                in_trans_np_1, out_trans_np_1 = out_trans_np_1, in_trans_np_1
                in_trans_np_2, out_trans_np_2 = out_trans_np_2, in_trans_np_2
                max_in_len, max_out_len = max_out_len, max_in_len
                in_len_np, out_len_np = out_len_np, in_len_np
                in_trans_len_np_1, out_trans_len_np_1 = out_trans_len_np_1, in_trans_len_np_1
                in_trans_len_np_2, out_trans_len_np_2 = out_trans_len_np_2, in_trans_len_np_2

            # torchify input
            curr_inp = Variable(torch.from_numpy(inp_np.astype('int32')).long().cuda())
            curr_out = Variable(torch.from_numpy(out_np.astype('int32')).long().cuda())
            in_trans_1 = Variable(torch.from_numpy(in_trans_np_1).long().cuda())
            in_trans_2 = Variable(torch.from_numpy(in_trans_np_2).long().cuda())
            out_trans_1 = Variable(torch.from_numpy(out_trans_np_1).long().cuda())
            out_trans_2 = Variable(torch.from_numpy(out_trans_np_2).long().cuda())
            in_sent_lens = torch.from_numpy(in_len_np).long().cuda()
            out_sent_lens = torch.from_numpy(out_len_np).long().cuda()
            in_trans_lens_1 = torch.from_numpy(in_trans_len_np_1).long().cuda()
            in_trans_lens_2 = torch.from_numpy(in_trans_len_np_2).long().cuda()
            out_trans_lens_1 = torch.from_numpy(out_trans_len_np_1).long().cuda()
            out_trans_lens_2 = torch.from_numpy(out_trans_len_np_2).long().cuda()

            # forward prop
            preds = net(curr_inp, curr_out, in_trans_1, in_trans_2, out_trans_1,out_trans_2,
                in_sent_lens, out_sent_lens, in_trans_lens_1, in_trans_lens_2, out_trans_lens_1, out_trans_lens_2, max_out_len)

            num_batches += 1

            # if training, compute loss and backprop
            if not args.eval_mode:

                # compute masked loss
                loss = criterion(preds, curr_out.view(-1))
                ep_loss += loss.data.item()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(params, args.grad_clip)
                optimizer.step()

            # if training, save model and print some predictions every save_freq minibatches
            # if eval, just do beam search on a few instances per minibatch
            if args.eval_mode or (b_idx % args.save_freq == 0):
                preds = preds.view(curr_bsz, max_out_len, -1).cpu().data.numpy()
                preds = np.argmax(preds, -1)
                for i in range(min(3, curr_bsz)):

                    # hack around beam search bug
                    try:
                        np_i_1 = in_trans_1[i].cpu().data.numpy()
                        np_i_2 = in_trans_2[i].cpu().data.numpy()
                        np_o_1 = out_trans_1[i].cpu().data.numpy()
                        np_o_2 = out_trans_2[i].cpu().data.numpy()
                        eos = np.where(out_np[i]==vocab['EOS'])[0][0]
                        print ('swapped:', swap)
                        print ('input: %s' % ' '.join([rev_vocab[w] for (j,w) in enumerate(inp_np[i])\
                            if j < in_len_np[i]]))
                        print ('gt output: %s' % ' '.join([rev_vocab[w] for (j,w) in enumerate(out_np[i, :eos])\
                            if j < out_len_np[i]]))
                        print ('output top-level parse: %s' % ' '.join([rev_label_voc_1[z] for (j,z) in enumerate(np_o_1)\
                            if j < out_trans_len_np_1[i]]))
                        print ('output Length: %s' % ' '.join([rev_label_voc_2[z] for (j,z) in enumerate(np_o_2)\
                            if j < out_trans_len_np_2[i]]))
                        print ('greedy: %s' % ' '.join([rev_vocab[w] for w in preds[i, :]]))

                        std_f.write ('\n input: %s' % ' '.join([rev_vocab[w] for (j,w) in enumerate(inp_np[i])\
                            if j < in_len_np[i]]))
                        std_f.write  ('\n gt output: %s' % ' '.join([rev_vocab[w] for (j,w) in enumerate(out_np[i, :eos])\
                            if j < out_len_np[i]]))
                        std_f.write  ('\n output top-level parse: %s' % ' '.join([rev_label_voc_1[z] for (j,z) in enumerate(np_o_1)\
                            if j < out_trans_len_np_1[i]]))
                        std_f.write  ('\n output Length: %s' % ' '.join([rev_label_voc_2[z] for (j,z) in enumerate(np_o_2)\
                            if j < out_trans_len_np_2[i]]))
                        std_f.write  ('\n greedy: %s' % ' '.join([rev_vocab[w] for w in preds[i, :]]))

                        std_f.flush()

                        beams = net.beam_search(curr_inp[i].unsqueeze(0), in_trans_1[i].unsqueeze(0),
                            out_trans_1[i].unsqueeze(0), out_trans_2[i].unsqueeze(0), in_sent_lens[i:i+1],
                            in_trans_lens_1[i:i+1], out_trans_lens_1[i:i+1], out_trans_lens_2[i:i+1], vocab['EOS'])

                        for beam_idx, beam in enumerate(beams):
                            print ('gpu beam %d, score:%f: %s' % \
                                (beam_idx, beam[0], ' '.join([rev_vocab[w] for w in beam[-1]])))
                    except:
                        print ('beam search error')

                # print statistics about the batch
                if not args.eval_mode:
                    print ('done with batch %d / %d in epoch %d, loss: %f, time:%d\n\n' \
                        % (b_idx, len(train_minibatches), ep,
                           ep_loss / num_batches, time.time()-start_time)
)
                    std_f.write ('done with batch %d / %d in epoch %d, loss: %f, time:%d \n \n' \
                        % (b_idx, len(train_minibatches), ep,
                           ep_loss / num_batches, time.time()-start_time))

                    std_f.flush()
                    torch.save({'state_dict':net.state_dict(),
                        'ep_loss':ep_loss / num_batches,
                        'train_minibatches': train_minibatches,
                        'dev_minibatches': dev_minibatches,
                        'config_args': args}, out_file)

                    ep_loss = 0.
                    num_batches = 0
                    start_time = time.time()

        # adjust LR for next epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * args.lr_decay_factor
            print ('new LR:', param_group['lr'])

