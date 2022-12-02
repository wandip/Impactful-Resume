import torch, time, sys, argparse, os, codecs, h5py, csv
import numpy as np
import pickle as cPickle
from torch.autograd import Variable
from train_cpgn_inpattern_length import CPGN 
from subwordnmt.apply_bpe import BPE, read_vocabulary

h5f = h5py.File('data/parsed_data.h5', 'r')
inp = h5f['inputs']
out = h5f['outputs']
in_lens = h5f['in_lengths']
out_lens = h5f['out_lengths']
print ('data is loaded')

test_indices = []
ref_file = open('evaluation4k/ref_paranmt_pattern_L_4k_23.txt', 'w')
inp_file = open('evaluation4k/inp_paranmt_pattern_L_4k_23.txt', 'w')
hyp_file = open('evaluation4k/hyp_pattern_paranmt_L_4k_23.txt', 'w')
# template_file = open('evaluation4k/template_pattern_paranmt_L_4k_23.txt', 'w')


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


# encode sentences and parses for targeted paraphrasing
def encode_data(out_f, tp_templates_1, tp_templates_2, tp_template_lens_1, tp_template_lens_2, fold):

    ref_dic = {}
    for i in fold:
        eos = np.where(out[i] == pp_vocab['EOS'])[0][0]
        ssent = ' '.join([rev_pp_vocab[w] for (j, w) in enumerate(out[i, :eos])\
                        if j < out_lens[i]-1])

        ref_dic[i]=ssent
        ref_file.write(ssent)
        ref_file.write('\n')

    # loop over sentences and transform them
    cnt = 0
    for i in fold:
        # if cnt > CNT:
        #     break
        cnt += 1
        
        resume_sentence = " the algorithm enables user generated content to be brought to the manufacturer of physical goods ."
        
        embedding= []
        for i in resume_sentence.split():
            embedding.append(pp_vocab[i] )
        embedding.append(2)
        elen = len(embedding)
        for i in range(40-elen):
            embedding.append(0)
        
        my_input = embedding
        
        my_input_length = elen

        stime = time.time()
        input_sentence = ' '.join([rev_pp_vocab[w] for (j, w) in enumerate(my_input)\
                        if j < elen-1])
        inp_file.write(input_sentence)
        inp_file.write('\n')

        print ('\n ============ START ============= ')
        # write gold sentence

        out_f.writerow({'idx': i,
                      'pattern': 'GOLD', 'length': len(input_sentence.split()),
                      'sentence': input_sentence})

        torch_sent = Variable(torch.from_numpy(np.array(my_input, dtype='int32')).long().cuda())
        torch_sent_len = torch.from_numpy(np.array([my_input_length], dtype='int32')).long().cuda()
        print(torch_sent.unsqueeze(0), tp_templates_1, tp_templates_2, torch_sent_len[:], tp_template_lens_1, tp_template_lens_2, pp_vocab['EOS'])
        # generate paraphrases from parses
        try:
            beam_dict = net.batch_beam_search(torch_sent.unsqueeze(0), tp_templates_1, tp_templates_2,
                torch_sent_len[:], tp_template_lens_1, tp_template_lens_2, pp_vocab['EOS'], beam_size=3, max_steps=40)

            # print(beam_dict, len(beam_dict))
            
            for b_idx in beam_dict:
                prob, _, _, seq = beam_dict[b_idx][0]

                # ##################
                # new gen EOF handling
                gen_sent_list=[]
                for w in seq[:-1]:
                    if rev_pp_vocab[w] == 'EOS':
                        break
                    gen_sent_list.append(rev_pp_vocab[w])
                gen_sent = ' '.join(gen_sent_list)
                # gen_sent = ' '.join([rev_pp_vocab[w] for w in seq[:-1]])
                gen_length = len(gen_sent_list)

                # ##################


                tt_p = templates_pattern_l[b_idx][0]
                tt_l = templates_pattern_l[b_idx][1][0]


                out_f.writerow({'idx': i,
                              'pattern': tt_p, 'length': tt_l,
                              'sentence': gen_sent})

                if b_idx == cnt - 1:
                    print ('===============================================\n')
                    print ('input: {}'.format(input_sentence))
                    print ('pattern: {}'.format(tt_p))
                    print ('length: {}'.format(tt_l))
                    print ('generated: {}'.format(gen_sent))
                # if len(ref_list[cnt - 1].split()) == tt:
                    hyp_file.write(gen_sent)
                    hyp_file.write('\n')
                    hyp_file.flush()

                    # template_file.write(tt_p)
                    # template_file.write('\n')
                    # template_file.flush()

                # gen_sent = ' '.join([rev_pp_vocab[w] for w in seq[:-1]])
                # out_f.writerow({'idx': ex['idx'],
                #     'template':tt, 'generated_length':len(seq[:-1]),
                #     'sentence':reverse_bpe(gen_sent.split())})

        except Exception as e:
            print (e)
            print ('beam search OOM')

        break

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Syntactically Controlled Paraphrase Transformer')

    ## paraphrase model args
    parser.add_argument('--gpu', type=str, default='1',
                        help='GPU id')
    parser.add_argument('--out_file', type=str, default='output4k/scpn_blank_2_paranmt_L_4k_2.out',
                        help='paraphrase save path')
    parser.add_argument('--parsed_input_file', type=str, default='data/parsed_data.h5',
                        help='parse load path')
    # parser.add_argument('--parsed_input_file', type=str, default='data/paws.csv',
    #         help='parse load path')
    parser.add_argument('--vocab', type=str, default='data/parse_vocab.pkl',
                        help='word vocabulary')
    parser.add_argument('--parse_vocab_2', type=str, default='data/unique_lenght.txt',
            help='tag vocabulary 2')

    parser.add_argument('--pp_model', type=str, default='models/scpn2_L_pattern_new_4k.pt',
                        help='paraphrase model to load')

    ## BPE args
    parser.add_argument('--bpe_codes', type=str, default='data/bpe.codes')
    parser.add_argument('--bpe_vocab', type=str, default='data/vocab.txt')
    parser.add_argument('--bpe_vocab_thresh', type=int, default=50)

    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    modell = args.pp_model
    slash = modell.find('/')+1
    modelll_name = modell[slash :]
    with open('minibatches/test_minibatches_' + 'test_model.pt'+ '.txt', 'r') as f:
        for line in f.readlines():
            [start, end] = [int(x) for x in line.strip().split(',')]
            for i in range(start, end):
                test_indices.append(i)


    # length = int(len(test_indices) / 400)  # length of each fold : 64
    folds = []
    # for i in range(399):
    #     folds += [test_indices[i * length:(i + 1) * length]]
    folds += [test_indices[0:len(test_indices)]]
    
    # print ('len(folds): {}'.format(len(folds)))

    # load saved models
    pp_model = torch.load(args.pp_model)

    # load vocab
    pp_vocab, rev_pp_vocab = cPickle.load(open(args.vocab, 'rb'))
    rev_pp_vocab[len(rev_pp_vocab)] = '___'
    pp_vocab['___'] = len(pp_vocab)

    parse_gen_voc_1 =pp_vocab
    rev_label_voc_1 = rev_pp_vocab

    tag_file_2 = codecs.open(args.parse_vocab_2, 'r', 'utf-8')
    parse_gen_voc_2 = {}
    for idx, line in enumerate(tag_file_2):
        line = line.strip()
        parse_gen_voc_2[line] = idx
    rev_label_voc_2 = dict((v,k) for (k,v) in parse_gen_voc_2.items())

    # load paraphrase network
    pp_args = pp_model['config_args']
    # print 'pp_args.d_nt: {}'.format(pp_args.d_nt)
    # print 'len(parse_gen_voc) - 1: {}'.format(len(parse_gen_voc) - 1)
    net = CPGN(pp_args.d_word, pp_args.d_hid, pp_args.d_nt, 40, pp_args.d_trans, 64,
        len(pp_vocab), len(parse_gen_voc_1), len(parse_gen_voc_2), pp_args.use_input_parse)

    net.cuda()
    net.load_state_dict(pp_model['state_dict'])
    net.eval()

    fn = ['idx', 'pattern', 'length', 'sentence']
    ofile = codecs.open(args.out_file, 'w', 'utf-8')

    out_f = csv.DictWriter(ofile, delimiter='\t', fieldnames=fn)
    out_f.writerow(dict((x, x) for x in fn))
    # print(folds)
    for fold in folds:
        # encode templates
        # template_lens = ['___ ___ ___ ___ ___' for x in range(CNT)]
        templates_pattern_l = []

        for i in fold:
            input_i = inp[i]
            output_i = out[i]

            flag_j = False
            flag_k = False
            for j in range(out_lens[i]):
                if output_i[j] not in input_i:
                    flag_j = True
                    for k in range(j + 1, out_lens[i]):
                        if output_i[k] not in input_i:
                            flag_k = True

                            if j == 0:
                                if k == j + 1:
                                    res = rev_pp_vocab[output_i[j]] + ' ' + rev_pp_vocab[output_i[k]] + ' ___'
                                else:
                                    if k == out_lens[i] - 1:
                                        res = rev_pp_vocab[output_i[j]] + ' ___ ' + rev_pp_vocab[output_i[k]]
                                    else:
                                        res = rev_pp_vocab[output_i[j]] + ' ___ ' + rev_pp_vocab[output_i[k]] + ' ___'
                            else:
                                if k == j + 1 and k == out_lens[i] - 1:
                                    res = '___ ' + rev_pp_vocab[output_i[j]] + ' ' + rev_pp_vocab[output_i[k]]
                                elif k == j + 1:
                                    res = '___ ' + rev_pp_vocab[output_i[j]] + ' ' + rev_pp_vocab[output_i[k]] + ' ___'
                                else:
                                    if k == out_lens[i] - 1:
                                        res = '___ ' + rev_pp_vocab[output_i[j]] + ' ___ ' + rev_pp_vocab[output_i[k]]
                                    else:
                                        res = '___ ' + rev_pp_vocab[output_i[j]] + ' ___ ' + rev_pp_vocab[
                                            output_i[k]] + ' ___'
                            break

                    if not flag_k:
                        if j == 0:
                            res = rev_pp_vocab[output_i[j]] + ' ___'
                        elif j == out_lens[i] - 1:
                            res = '___ ' + rev_pp_vocab[output_i[j]]
                        else:
                            res = '___ ' + rev_pp_vocab[output_i[j]] + ' ___'
                    break

            if not flag_j:
                res = '___'

            templates_pattern_l.append((str(res), [str(out_lens[i])]))
            # print("*"*100)
            # print(input_i, len(input_i), res, str(out_lens[i]))
        # encode templates
        # #1: pattern
        # template_1_lens = [6 for x in templates_pattern_l]
        
        # print(templates_pattern_l)
        templates_pattern_l = []
        templates_pattern_l.append(("___ allows ___ created ___", ["17"]))        
        
        template_1_lens = [len(x[0].split(' ')) for x in templates_pattern_l]
        
        # print(template_1_lens)
        
        np_templates_1 = np.zeros((len(templates_pattern_l), max(template_1_lens)), dtype='int32')
        for z, template in enumerate(templates_pattern_l):
            np_templates_1[z, :template_1_lens[z]] = [parse_gen_voc_1[w] for w in templates_pattern_l[z][0].split(' ')]

        tp_templates_1 = Variable(torch.from_numpy(np_templates_1).long().cuda())
        tp_template_lens_1 = torch.from_numpy(np.array(template_1_lens, dtype='int32')).long().cuda()

        # #2: Length
        template_2_lens = [1 for x in templates_pattern_l]
        # print 'template_2_lens: {}'.format(template_2_lens)
        np_templates_2 = np.zeros((len(templates_pattern_l), max(template_2_lens)), dtype='int32')
        for z, template in enumerate(templates_pattern_l):
            np_templates_2[z, :template_2_lens[z]] = [parse_gen_voc_2[w] for w in templates_pattern_l[z][1]]

        tp_templates_2 = Variable(torch.from_numpy(np_templates_2).long().cuda())
        tp_template_lens_2 = torch.from_numpy(np.array(template_2_lens, dtype='int32')).long().cuda()

        # instantiate BPE segmenter
        bpe_codes = codecs.open(args.bpe_codes, encoding='utf-8')
        bpe_vocab = codecs.open(args.bpe_vocab, encoding='utf-8')
        bpe_vocab = read_vocabulary(bpe_vocab, args.bpe_vocab_thresh)
        bpe = BPE(bpe_codes, '@@', bpe_vocab, None)

        # paraphrase the sst!
        encode_data(out_f, tp_templates_1, tp_templates_2, tp_template_lens_1, tp_template_lens_2, fold)

        break

    hyp_file.close()
    # template_file.close()
    ref_file.close()
    inp_file.close()
