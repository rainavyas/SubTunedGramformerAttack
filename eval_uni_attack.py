'''
Evaluate universal adversarial attack.

Evaluate by counting average number of edits between original input
(with attack phrase) and  GEC model prediction
'''

import sys
import os
import argparse
import torch
from gec_tools import get_sentences, correct, count_edits
from Seq2seq import Seq2seq
import json
from datetime import date
from statistics import mean, stdev
from uni_attack import substitute

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('IN', type=str, help='Path to input data')
    commandLineParser.add_argument('MODEL', type=str, help='Path to Gramformer model')
    commandLineParser.add_argument('SUB_DICT', type=str, help='path to json dict with substitutions')
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
    _, sentences = get_sentences(args.IN)

    # Load sub dict
    with open(args.SUB_DICT) as json_file:
        sub_dict = json.load(json_file)
    
    # Perform substitution attack
    edit_counts = []
    for i, sent in enumerate(sentences):
        print(f'On {i}/{len(sentences)}')
        sent = substitute(sent, sub_dict)
        correction = correct(model, sent)
        edit_counts.append(count_edits(sent, correction))
    
    # Report stats
    print('-----------')
    print(f'Average number of edits {mean(edit_counts)} +- {stdev(edit_counts)}')