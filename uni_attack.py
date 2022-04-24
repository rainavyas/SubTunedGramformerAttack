'''
Perform substitution adversarial attack on 
GEC system, with aim of finding universal adversarial phrase
that minimises average number of edits between original and 
predicted gec sentence.
'''
import sys
import os
import argparse
import torch
from gec_tools import get_sentences, correct, count_edits
from Seq2seq import Seq2seq
import json
from datetime import date
from statistics import mean

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_avg(model, sentences, sub_dict):
    edit_counts = []
    for sent in sentences:
        import pdb; pdb.set_trace()
        sent = substitute(sent, sub_dict)
        correction = correct(model, sent)
        edit_counts.append(count_edits(sent, correction))
    return mean(edit_counts)

def substitute(sent, sub_dict):
    if len(sent) > 0:
        for original_word, sub_word in sub_dict.items():
            sent = sent.replace(original_word, sub_word)
    return sent

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('IN', type=str, help='Path to input data')
    commandLineParser.add_argument('MODEL', type=str, help='Path to Gramformer model')
    commandLineParser.add_argument('VOCAB', type=str, help='ASR vocab file')
    commandLineParser.add_argument('WORD', type=str, help='Current word to substitute')
    commandLineParser.add_argument('LOG', type=str, help='Specify txt file to log iteratively better words')
    commandLineParser.add_argument('--prev_attack', type=str, default='', help='Json file containing dict (original word:substitution)')
    commandLineParser.add_argument('--num_points', type=int, default=1000, help='Number of training data points to consider')
    commandLineParser.add_argument('--search_size', type=int, default=400, help='Number of words to check')
    commandLineParser.add_argument('--start', type=int, default=0, help='Vocab batch number')
    commandLineParser.add_argument('--seed', type=int, default=1, help='reproducibility')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/uni_attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    set_seeds(args.seed)

    # Load Model
    model = Seq2seq()
    model.load_state_dict(torch.load(args.MODEL, map_location=torch.device('cpu')))
    model.eval()

    # Load input sentences
    _, sentences = get_sentences(args.IN, num=args.num_points)

    # Get list of words to try
    with open(args.VOCAB, 'r') as f:
        test_words = json.loads(f.read())
    test_words = [str(word).lower() for word in test_words]

    # Keep only selected batch of words
    start_index = args.start*args.search_size
    test_words = test_words[start_index:start_index+args.search_size]

    # Add target word itself to batch of words
    test_words = [args.WORD] + test_words

    # Load dict containing existing substitutions
    if args.prev_attack == '':
        sub_dict = {}
    else:
        with open(args.prev_attack) as json_file:
            sub_dict = json.load(json_file)

    # Initialise empty log file
    with open(args.LOG, 'w') as f:
        f.write("Logged on "+ str(date.today()))

    best = ('none', 1000)
    for word in test_words:
        sub_dict[args.WORD] = word
        edits_avg = get_avg(model, sentences, sub_dict)

        if edits_avg < best[1]:
            best = (word, edits_avg)
            # Write to log
            with open(args.LOG, 'a') as f:
                out = '\n'+best[0]+" "+str(best[1])
                f.write(out)
