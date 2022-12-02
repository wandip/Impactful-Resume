import torch, time, argparse, os, codecs, h5py, cPickle, random

import torch, sys, h5py
import numpy as np
from nltk import ParentedTree
import torch, time, argparse, os, codecs, h5py, cPickle, random

# import h5py
# import pickle as pkl
import cPickle
import torch, sys, h5py
import numpy as np
from nltk import ParentedTree

def is_paren(tok):
    return tok == ")" or tok == "("

# given list of parse strings, output numpy array containing the transformations
def bert_indexify_transformations(in_p, out_p, label_voc, args):
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
        # print 'not none'

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

#returns tokenized parse tree and removes leaf nodes (i.e. words)
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


def create_trans_emb_weights(label_vocab, max_trans_size):
    weights_matrix = np.zeros((len(label_vocab), max_trans_size))
    print ('weights_matrix.shape: {}'.format(weights_matrix.shape))
    for word in label_vocab:
        try:
            weights_matrix[label_vocab[word], :int(word)] = [1]*int(word)

        except KeyError:
            exit(1)
    return weights_matrix

def reverse_bpe(sent):
    x = []
    cache = ''

    for w in sent:
        if w.endswith('@@'):
            cache += w.replace('@@', '')
        elif cache != '':
            x.append(cache + w)
            cache = ''
        else:
            x.append(w)

    return ' '.join(x)


def get_sentence (x, rev_pp_vocab):
    gen_sent = ' '.join([rev_pp_vocab[w] for w in x])
    return reverse_bpe(gen_sent.split())


def bertscore_list(data):
    # load vocab
    pp_vocab, rev_pp_vocab = cPickle.load(open('data/parse_vocab.pkl', 'rb'))

    # load data, word vocab, and parse vocab
    h5f = h5py.File(data, 'r')
    inps = h5f['inputs']
    outs = h5f['outputs']
    indices = []
    inps_list = []
    outs_list = []
    print('In get_bertscore_list')
    cnt = 0
    for i in range(len(inps)):
        cnt += 1
        if cnt % 100000 == 0:
            print (cnt)
        indices.append(str(i) + '_' + str(i))

        input_sentence = get_sentence(inps[i][:list(inps[i]).index(2)], rev_pp_vocab)
        inps_list.append(input_sentence)

        output_sentence = get_sentence(outs[i][:list(outs[i]).index(2)], rev_pp_vocab)
        outs_list.append(output_sentence)

    f_cmu = open('data/bert_scpn.pkl', 'wb')
    pkl.dump(inps_list, f_cmu)
    pkl.dump(outs_list, f_cmu)
    pkl.dump(indices, f_cmu)

    f_cmu.close()


def get_list_of_unique_berts():

    with open('data/unique_bert.txt', 'w') as f:
        for i in range(101):
            f.write(str(i) + '\n')

