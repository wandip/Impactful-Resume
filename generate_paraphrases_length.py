import torch, time, sys, argparse, os, codecs, h5py, csv
import numpy as np
from torch.autograd import Variable
from nltk import ParentedTree
from train_cpgn_length import CPGN
from subwordnmt.apply_bpe import BPE, read_vocabulary
import pickle as cPickle
import random
templates_length = []
for i in range(4, 41):
    templates_length.append([str(i)])

# encode sentences and parses for targeted paraphrasing
def encode_data(out_file, tp_templates, tp_template_lens):
    h5f = h5py.File(args.parsed_input_file, 'r')
    inp = h5f['inputs']
    out = h5f['outputs']
    in_lens = h5f['in_lengths']
    out_lens = h5f['out_lengths']

    test_indices = []
    with open('minibatches/test_minibatches_test_model.pt.txt', 'r') as f:
        for line in f.readlines():
            [start, end] = [int(x) for x in line.strip().split(',')]
            for i in range(start, end):
                test_indices.append(i)
    print ('len((test_indices)): {}'.format(len((test_indices))))
    print ('data is loaded')
    ref_file = open('evaluation4k/ref_paranmt_L_4k_2.txt', 'w')
    inp_file = open('evaluation4k/inp_paranmt_L_4k_2.txt', 'w')

    ref_dic = {}
    for i in test_indices:
        eos = np.where(out[i] == pp_vocab['EOS'])[0][0]
        ssent = ' '.join([rev_pp_vocab[w] for (j, w) in enumerate(out[i, :eos])\
                        if j < out_lens[i]-1])

        ref_dic[i]=ssent
        ref_file.write(ssent)
        ref_file.write('\n')

    fn = ['idx', 'template', 'generated_length', 'sentence']
    ofile = codecs.open(out_file, 'w', 'utf-8')

    out = csv.DictWriter(ofile, delimiter='\t', fieldnames=fn)
    out.writerow(dict((x, x) for x in fn))

    hyp_file = open('evaluation4k/hyp_length_paranmt_4k_2.txt', 'w')

    # loop over sentences and transform them
    cnt = 0
    for i in test_indices:
        # if cnt >5:
        #     break
        cnt += 1
        print (cnt)

        stime = time.time()
        input_sentence = ' '.join([rev_pp_vocab[w] for (j, w) in enumerate(inp[i])\
                        if j < in_lens[i]-1])

        print ('input_sentence: {}'.format(input_sentence))
        inp_file.write(input_sentence)
        inp_file.write('\n')

        # write gold sentence
        out.writerow({'idx': i,
                      'template': 'GOLD', 'generated_length': len(input_sentence.split()),
                      'sentence': input_sentence})

        torch_sent = Variable(torch.from_numpy(np.array(inp[i], dtype='int32')).long().cuda())
        torch_sent_len = torch.from_numpy(np.array([in_lens[i]], dtype='int32')).long().cuda()

        # generate paraphrases from parses
        try:
            beam_dict = net.batch_beam_search(torch_sent.unsqueeze(0), tp_templates,
                                              torch_sent_len[:], tp_template_lens, pp_vocab['EOS'], beam_size=3,
                                              max_steps=40)

            for b_idx in beam_dict:
                prob, _, _, seq = beam_dict[b_idx][0]
                tt = int(templates_length[b_idx][0])
                gen_sent = ' '.join([rev_pp_vocab[w] for w in seq[:-1]])
                gen_length = len(gen_sent.split())

                out.writerow({'idx': i,
                              'template': tt-1, 'generated_length': gen_length,
                              'sentence': gen_sent})

                if out_lens[i] == tt:
                    print ('===================== {} ==========================\n'.format(cnt))
                    print ('out_lens[i]: {}'.format(out_lens[i]))
                    print ('tt: {}'.format(tt))
                    print ('gen_sent: {}'.format(gen_sent))
                    hyp_file.write(gen_sent)
                    hyp_file.write('\n')

        except Exception as e:
            print ('beam search OOM')
            print (e)


        print (i, time.time() - stime)

    hyp_file.close()
    ref_file.close()
    inp_file.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Syntactically Controlled Paraphrase Transformer')

    ## paraphrase model args
    parser.add_argument('--gpu', type=str, default='1',
                        help='GPU id')
    parser.add_argument('--out_file', type=str, default='output4k/CPGN_length_paranmt4k_2.out',
                        help='paraphrase save path')
    parser.add_argument('--parsed_input_file', type=str, default='data/parsed_data.h5',
                        help='parse load path')
    parser.add_argument('--vocab', type=str, default='data/parse_vocab.pkl',
                        help='word vocabulary')
    parser.add_argument('--parse_vocab', type=str, default='data/unique_lenght.txt',
                        help='tag vocabulary')
    parser.add_argument('--pp_model', type=str, default='models/CPGN2_bert_length_4k.pt',
                        help='paraphrase model to load')

    ## BPE args
    parser.add_argument('--bpe_codes', type=str, default='data/bpe.codes')
    parser.add_argument('--bpe_vocab', type=str, default='data/vocab.txt')
    parser.add_argument('--bpe_vocab_thresh', type=int, default=50)

    args = parser.parse_args()
    # load saved models
    pp_model = torch.load(args.pp_model)

    # load vocab
    pp_vocab, rev_pp_vocab = cPickle.load(open(args.vocab, 'rb'))

    tag_file = codecs.open(args.parse_vocab, 'r', 'utf-8')
    parse_gen_voc = {}
    for idx, line in enumerate(tag_file):
        line = line.strip()
        parse_gen_voc[line] = idx
    rev_label_voc = dict((v, k) for (k, v) in parse_gen_voc.items())

    # load paraphrase network
    pp_args = pp_model['config_args']
    net = CPGN(pp_args.d_word, pp_args.d_hid, pp_args.d_nt, pp_args.d_trans,
               len(pp_vocab), len(parse_gen_voc), pp_args.use_input_parse, None)
    net.cuda()
    net.load_state_dict(pp_model['state_dict'])
    net.eval()

    # encode templates
    template_lens = [1 for x in templates_length]
    np_templates = np.zeros((len(templates_length), max(template_lens)), dtype='int32')
    for z, template in enumerate(templates_length):
        np_templates[z, :template_lens[z]] = [parse_gen_voc[w] for w in templates_length[z]]

    tp_templates = Variable(torch.from_numpy(np_templates).long().cuda())
    tp_template_lens = torch.from_numpy(np.array(template_lens, dtype='int32')).long().cuda()

    # instantiate BPE segmenter
    bpe_codes = codecs.open(args.bpe_codes, encoding='utf-8')
    bpe_vocab = codecs.open(args.bpe_vocab, encoding='utf-8')
    bpe_vocab = read_vocabulary(bpe_vocab, args.bpe_vocab_thresh)
    bpe = BPE(bpe_codes, '@@', bpe_vocab, None)

    # paraphrase the sst!
    encode_data(args.out_file, tp_templates, tp_template_lens)
