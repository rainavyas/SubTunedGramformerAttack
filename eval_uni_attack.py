'''
Evaluate universal adversarial substitution attack.

Evaluate by counting average number of edits between original input
(with attack phrase) and  GEC model prediction

Save the original, reference and prediction files as: .inc, .ref, .pred
'''

import sys
import os
import argparse
import torch
from gec_tools import get_sentences, correct, count_edits
from Seq2seq import Seq2seq
import json
from datetime import date
from uni_attack import substitute
import string

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_sentences_dict(data_path, remove_punct=False):
    with open(data_path, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip('\n') for l in lines]
    exclude = set(string.punctuation)
    id2text = {}
    for l in lines:
        parts = l.split()
        id = parts[0]
        text = ' '.join(parts[1:])
        # remove space before full stop
        if text[-2:] == ' .':
            text = text[:-2]+'.'
        if remove_punct:
            # remove punctuation
            text = ''.join(ch for ch in text if ch not in exclude)
        id2text[id] = text
    return id2text

def align_data(inc_dict, pred_dict, corr_dict):
    inc_sens = []
    pred_sens = []
    corr_sens = []
    for i, (id, text) in enumerate(corr_dict.items()):
        try:
            pred_sens.append(pred_dict[id]+'\n')
            inc_sens.append(inc_dict[id]+'\n')
            corr_sens.append(text+'\n')
        except:
            # print(f'{i}) {id} in corrected but not in predicted')
            pass
    assert len(pred_sens) == len(inc_sens), "Mismatch in num items"
    return inc_sens, pred_sens, corr_sens

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('IN', type=str, help='Path to input data')
    commandLineParser.add_argument('CORR', type=str, help='Path to correct output test data')
    commandLineParser.add_argument('MODEL', type=str, help='Path to Gramformer model')
    commandLineParser.add_argument('SUB_DICT', type=str, help='path to json dict with substitutions')
    commandLineParser.add_argument('BASE', type=str, help='Path base for output files')
    commandLineParser.add_argument('--seed', type=int, default=1, help='reproducibility')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval_uni_attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    set_seeds(args.seed)

    # Load Model
    model = Seq2seq()
    model.load_state_dict(torch.load(args.MODEL, map_location=torch.device('cpu')))
    model.eval()

    # Load input sentences
    inc_id2text = get_sentences_dict(args.IN)

    # Load correct sentences
    corr_id2text = get_sentences_dict(args.CORR)

    # Load sub dict
    with open(args.SUB_DICT) as json_file:
        sub_dict = json.load(json_file)
    
    # Perform substitution attack
    pred_id2text = {}
    for i, (id, sent) in enumerate(inc_id2text.items()):
        set_seeds(args.seed)
        print(f'On {i}/{len(inc_id2text)}')
        sent = substitute(sent, sub_dict)
        correction = correct(model, sent)
        pred_id2text[id] = correction

    # Align files and save
    inc_sens, pred_sens, corr_sens = align_data(inc_id2text, pred_id2text, corr_id2text)

    # Save to output files
    filename = f'{args.BASE}.inc'
    with open(filename, 'w') as f:
        f.writelines(inc_sens)
    filename = f'{args.BASE}.pred'
    with open(filename, 'w') as f:
        f.writelines(pred_sens)
    filename = f'{args.BASE}.corr'
    with open(filename, 'w') as f:
        f.writelines(corr_sens)