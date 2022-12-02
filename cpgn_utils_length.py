import torch, time, argparse, os, codecs, h5py, random
import pickle as cPickle

import torch, sys, h5py
import numpy as np
from nltk import ParentedTree
import torch, time, argparse, os, codecs, h5py, random

# import h5py
# import pickle as pkl
import torch, sys, h5py
import numpy as np
from nltk import ParentedTree
import csv
from subwordnmt.apply_bpe import BPE, read_vocabulary


def is_paren(tok):
    return tok == ")" or tok == "("


# given list of parse strings, output numpy array containing the transformations
def length_indexify_transformations(in_p, out_p, label_voc, args):
    in_seqs = []
    out_seqs = []
    mismatch_inds = []

    max_trans_size = 1
    for idx in range(len(in_p)):
        in_l = [in_p[idx]]
        out_l = [out_p[idx]]

        try:
            x = [label_voc[z] for z in in_l]
            x = [label_voc[z] for z in out_l]
            in_seqs.append(in_l)
            out_seqs.append(out_l)
            mismatch_inds.append(idx)
        except:
            pass

    # no syntactic transformations in the batch!
    if len(in_seqs) == 0:
        return None

    # otherwise, indexify and return
    else:
        in_trans_np = np.zeros((len(in_seqs), max_trans_size), dtype='int32')
        out_trans_np = np.zeros((len(in_seqs), max_trans_size), dtype='int32')

        in_lengths = []
        out_lengths = []
        for idx in range(len(in_seqs)):
            curr_in = in_seqs[idx]
            in_trans_np[idx, :len(curr_in)] = [label_voc[z] for z in curr_in]
            in_lengths.append(len(curr_in))

            curr_out = out_seqs[idx]
            out_trans_np[idx, :len(curr_out)] = [label_voc[z] for z in curr_out]
            out_lengths.append(len(curr_out))
        return in_trans_np, out_trans_np, mismatch_inds,\
            np.array(in_lengths, dtype='int32'), np.array(out_lengths, dtype='int32')

def deleaf(tree):
    nonleaves = ''
    for w in str(tree).replace('\n', '').split():
        w = w.replace('(', '( ').replace(')', ' )')
        nonleaves += w + ' '

    arr = nonleaves.split()
    for n, i in enumerate(arr):
        if n + 1 < len(arr):
            tok1 = arr[n]
            tok2 = arr[n + 1]
            if not is_paren(tok1) and not is_paren(tok2):
                arr[n + 1] = ""

    nonleaves = " ".join(arr)
    return nonleaves.split() + ['EOP']

#removes levels of parse tree belowe specifice level or random levels
#if level is None
def parse_tree_level_dropout(tree, treerate, level=None):
    def parse_tree_level_dropout2(tree, level, mlevel):
        if level == mlevel:
            for idx, n in enumerate(tree):
                if isinstance(n, ParentedTree):
                    tree[idx] = "(" + n.label() + ")"
        else:
            for n in tree:
                parse_tree_level_dropout2(n, level + 1, mlevel)

    h = tree.height()

    if not level:
        level = 0
        for i in range(2, h):
            if np.random.rand() <= treerate:
                level = i
                break
        if level > 0:
            parse_tree_level_dropout2(tree, 1, level)

    else:
        parse_tree_level_dropout2(tree, 1, level)

#dropout constituents from tree
def tree_dropout(tree, treerate, level):
    if level == 0:
        for n in tree:
            tree_dropout(n, treerate, level + 1)
    else:
        for idx, n in enumerate(tree):
            if np.random.rand(1)[0] <= treerate and isinstance(n, ParentedTree):
                tree[idx] = "(" + n.label() + ")"
            elif not isinstance(n, ParentedTree):
                continue
            else:
                tree_dropout(n, treerate, level + 1)

# given list of parse strings, output numpy array containing the transformations
def parse_indexify_transformations(in_p, out_p, label_voc, args):

    in_trimmed_seqs = []
    in_seqs = []
    out_trimmed_seqs = []
    out_seqs = []

    max_trans_size = 0
    for idx in range(len(in_p)):

        # very rarely, a tree is invalid
        try:
            in_trimmed = ParentedTree.fromstring(in_p[idx])
            in_orig = ParentedTree.fromstring(in_p[idx])
            out_trimmed = ParentedTree.fromstring(out_p[idx])
            out_orig = ParentedTree.fromstring(out_p[idx])
        except:
            continue

        out_dh = parse_tree_level_dropout(out_trimmed, args.tree_level_dropout)
        parse_tree_level_dropout(in_trimmed, args.tree_level_dropout, level=out_dh)

        in_orig = deleaf(in_orig)
        in_trimmed = deleaf(in_trimmed)
        out_orig = deleaf(out_orig)
        out_trimmed = deleaf(out_trimmed)

        if max_trans_size < len(in_orig):
            max_trans_size = len(in_orig)
        if max_trans_size < len(out_orig):
            max_trans_size = len(out_orig)

        # only consider instances where top-level of input parse != top-level output
        if in_trimmed != out_trimmed:
            # make sure everything is in vocab
            try:             
                x = [label_voc[z] for z in in_orig]
                x = [label_voc[z] for z in out_orig]
                in_seqs.append(in_orig)
                out_seqs.append(out_orig)
                out_trimmed_seqs.append(out_trimmed)
                in_trimmed_seqs.append(in_trimmed)
            except:
                pass

    # no syntactic transformations in the batch!
    if len(in_seqs) == 0:
        return None

    # otherwise, indexify and return
    else:
        in_trans_np = np.zeros((len(in_seqs), max_trans_size), dtype='int32')
        out_trans_np = np.zeros((len(in_seqs), max_trans_size), dtype='int32')
        in_trimmed_np = np.zeros((len(in_seqs), max_trans_size), dtype='int32')
        out_trimmed_np = np.zeros((len(in_seqs), max_trans_size), dtype='int32')

        in_lengths = []
        out_lengths = []
        out_trimmed_lengths = []
        in_trimmed_lengths = []
        for idx in range(len(in_seqs)):
            curr_in = in_seqs[idx]
            in_trans_np[idx, :len(curr_in)] = [label_voc[z] for z in curr_in]
            in_lengths.append(len(curr_in))

            curr_out = out_seqs[idx]
            out_trans_np[idx, :len(curr_out)] = [label_voc[z] for z in curr_out]
            out_lengths.append(len(curr_out))

            curr_trimmed_in = in_trimmed_seqs[idx]
            in_trimmed_np[idx, :len(curr_trimmed_in)] = [label_voc[z] for z in curr_trimmed_in]
            in_trimmed_lengths.append(len(curr_trimmed_in))

            curr_trimmed_out = out_trimmed_seqs[idx]
            out_trimmed_np[idx, :len(curr_trimmed_out)] = [label_voc[z] for z in curr_trimmed_out]
            out_trimmed_lengths.append(len(curr_trimmed_out))

        # cut off extra padding
        in_trans_np = in_trans_np[:, :np.max(in_lengths)]
        out_trans_np = out_trans_np[:, :np.max(out_lengths)]
        in_trimmed_np = in_trimmed_np[:, :np.max(in_trimmed_lengths)]
        out_trimmed_np = out_trimmed_np[:, :np.max(out_trimmed_lengths)]

        return in_trans_np, out_trans_np, in_trimmed_np, out_trimmed_np,\
            np.array(in_lengths, dtype='int32'), np.array(out_lengths, dtype='int32'),\
            np.array(in_trimmed_lengths, dtype='int32'), np.array(out_trimmed_lengths, dtype='int32')


def create_trans_emb_weights(label_vocab, max_trans_size):
    weights_matrix = np.zeros((len(label_vocab), max_trans_size))
    print ('weights_matrix.shape: {}'.format(weights_matrix.shape))
    for word in label_vocab:
        try:
            weights_matrix[label_vocab[word], :int(word)] = [1]*int(word)

        except KeyError:
            print ('\n\nERRORRRRRRRRR \n\n')
            exit(1)
    return weights_matrix


en_folder = '../data-process/dataset/x-final/en/'
##### for en paws
def generate_vocab_pkl(threshold=50):
    vocab_list = []
    with open(en_folder + 'pawsen_vocab.txt', 'rb') as f:
        for line in f.readlines():
            vocab, freq = line.split()[0], line.split()[1]
            if freq >= threshold:
                vocab_list.append(vocab)

    vocab = {}
    vocab['PAD']=0
    vocab['START']=1
    vocab['EOS']=2

    for i in range(len(vocab_list)):
        vocab[vocab_list[i]] = i+3
    rev_vocab = dict((v,k) for (k,v) in vocab.iteritems())
    with open('data/pawsen_parse_vocab.pkl', 'wb') as f:
        cPickle.dump(vocab, f)
        cPickle.dump(rev_vocab, f)


# generate_vocab_pkl()
def parse_data():
    # instantiate BPE segmenter
    bpe_codes = codecs.open(en_folder+'paws_en.codes', encoding='utf-8')
    bpe_vocab = codecs.open(en_folder+'pawsen_vocab.txt', encoding='utf-8')
    bpe_vocab = read_vocabulary(bpe_vocab, 50)
    bpe = BPE(bpe_codes, '@@', bpe_vocab, None)

    # load vocab
    pp_vocab = cPickle.load(open('data/pawsen_parse_vocab.pkl', 'rb'))
    rev_pp_vocab = cPickle.load(open('data/pawsen_parse_vocab.pkl', 'rb'))

    # read parsed data
    infile = codecs.open(en_folder + 'train.tsv_sentences.tsv', 'r', 'utf-8')
    inrdr = csv.DictReader(infile, delimiter='\t')

    # loop over sentences and transform them
    sentenceMatrix_1 = []
    sentenceMatrix_2 = []
    lenMatrix_1 = []
    lenMatrix_2 = []

    max_seq_length = 0 #not used
    min_seq_length = 47
    total_cnt = 0
    non_equal_cnt = 0

    for d_idx, ex in enumerate(inrdr):
        try:
            ssent_1 = ' '.join(ex['sentence1'].split())
            seg_sent_1 = bpe.segment(ssent_1.lower()).split()

            ssent_2 = ' '.join(ex['sentence2'].split())
            seg_sent_2 = bpe.segment(ssent_2.lower()).split()
        except:
            continue
        # encode sentence using pp_vocab, leave one word for EOS
        seg_sent_1 = [pp_vocab[w] for w in seg_sent_1 if w in pp_vocab]
        # add EOS
        seg_sent_1.append(pp_vocab['EOS'])
        numpy_sent_len_1 = np.array(len(seg_sent_1), dtype='int32')

        if len(seg_sent_1)>max_seq_length:
            max_seq_length = len(seg_sent_1)

        if len(seg_sent_1)<min_seq_length:
            min_seq_length = len(seg_sent_1)

        sentenceMatrix_1.append(seg_sent_1)
        lenMatrix_1.append(numpy_sent_len_1)

        # encode sentence using pp_vocab, leave one word for EOS
        seg_sent_2 = [pp_vocab[w] for w in seg_sent_2 if w in pp_vocab]
        # add EOS
        seg_sent_2.append(pp_vocab['EOS'])
        # numpy_sent = np.array(seg_sent_1, dtype='int32')
        numpy_sent_len_2 = np.array(len(seg_sent_2), dtype='int32')

        if len(seg_sent_2)>max_seq_length:
            max_seq_length = len(seg_sent_2)

        if len(seg_sent_2)<min_seq_length:
            min_seq_length = len(seg_sent_2)

        total_cnt += 1
        if len(seg_sent_1) != len(seg_sent_2):
            non_equal_cnt += 1

        sentenceMatrix_2.append(seg_sent_2)
        lenMatrix_2.append(numpy_sent_len_2)

    sentenceMatrix_1_numpy = []
    sentenceMatrix_2_numpy = []

    max_seq_length = 40
    for i in range(len(sentenceMatrix_1)):
        a = sentenceMatrix_1[i]
        b = sentenceMatrix_2[i]

        a_new = np.zeros(max_seq_length)
        b_new = np.zeros(max_seq_length)
        a_new[:len(a)] = a
        b_new[:len(b)] = b

        sentenceMatrix_1_numpy.append(a_new)
        sentenceMatrix_2_numpy.append(b_new)

    sentenceMatrix_1 = np.array(sentenceMatrix_1_numpy, dtype='int32')
    sentenceMatrix_2 = np.array(sentenceMatrix_2_numpy, dtype='int32')
    lenMatrix_1 = np.array(lenMatrix_1, dtype='int32')
    lenMatrix_2 = np.array(lenMatrix_2, dtype='int32')

    result = {'inputs': sentenceMatrix_1,
              'outputs': sentenceMatrix_2,
              'in_lengths': lenMatrix_1,
              'out_lengths':lenMatrix_2
              }

    h = h5py.File('data/pawsen_data.hdf5')
    for k, v in result.items():
        h.create_dataset(k, data=v)

    h5f = h5py.File('data/parsed_data.h5', 'r')
    inp = h5f['inputs']
    out = h5f['outputs']
    # in_parses = h5f['input_parses']
    # out_parses = h5f['output_parses']
    in_lens = h5f['in_lengths']
    out_lens = h5f['out_lengths']

# python train_cpgn_length.py --gpu 1 --data data/pawsen_data.h5 --vocab data/pawsen_parse_vocab.pkl  --model scpn2_length_pawsen.pt

# parse_data()
# print '-- Done --'
