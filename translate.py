"""
Usage: translate.py -model exp\transformer_accu_43.572.chkpt -vocab D:\Work\[Archive]\transformer\data_small\vocab.pt -src exp\input.txt -output exp\prediction.txt
"""
import codecs

import torch
import torch.utils.data
import argparse
from tqdm import tqdm

from dataset import collate_fn, TranslationDataset
from transformer.Translator import Translator
from preprocess import read_instances_from_file, convert_instance_to_idx_seq


def main():
    parser = _build_parser()
    opt = parser.parse_args()

    # Prepare DataLoader
    vocab = torch.load(opt.vocab)

    max_len = 50
    test_src_word_insts = read_instances_from_file(opt.src, max_len, True)

    # load test word sequence
    test_src_insts = convert_instance_to_idx_seq(test_src_word_insts, vocab['src'])
    test_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=vocab['src'],
            tgt_word2idx=vocab['tgt'],
            src_insts=test_src_insts),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=collate_fn)

    # do translation
    translator = Translator(opt)
    with codecs.open(opt.output, 'w', 'utf-8') as f:
        for batch in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
            all_hyp, all_scores = translator.translate_batch(*batch)
            for idx_seqs in all_hyp:
                for idx_seq in idx_seqs:
                    eos = test_loader.dataset.tgt_word2idx['</s>']
                    if eos in idx_seq:
                        real_len = min(len(idx_seq), idx_seq.index(eos))
                    else:
                        real_len = len(idx_seq)
                    eos_line = idx_seq[:real_len]
                    prediction_line = ' '.join([test_loader.dataset.tgt_idx2word[idx] for idx in eos_line])

                    f.write(prediction_line + '\n')
    print('[Info] Finished.')


def _build_parser():
    parser = argparse.ArgumentParser(description='translate.py')
    parser.add_argument('-model', required=True, help='transformer.chkpt')
    parser.add_argument('-src', required=True, help='input.txt')
    parser.add_argument('-vocab', required=True, help='vocab.pt')
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")

    parser.add_argument('-beam_size', type=int, default=5, help='Beam size')
    parser.add_argument('-batch_size', type=int, default=30, help='Batch size')
    parser.add_argument('-n_best', type=int, default=3)
    return parser


if __name__ == "__main__":
    main()
