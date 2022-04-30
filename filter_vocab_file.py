'''
Filter vocab list by POS
'''

import argparse
import sys
import os
import nltk
from collections import defaultdict
from nltk.tokenize import word_tokenize
import json

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('VOCAB', type=str, help='ASR vocab file')
    commandLineParser.add_argument('OUT', type=str, help='Path to dir to save data')
    commandLineParser.add_argument('--num_points', type=int, default=1000, help='Number of training data points to consider')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/filter_vocab_file.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load entire vocabulary
    with open(args.VOCAB, 'r') as f:
        words = json.loads(f.read())
    words = [str(word).lower() for word in words]

    # Dictionary to store words for each POS
    pos_dict = defaultdict(list)

    for word in words:
        tokens = word_tokenize(word)
        tagged = nltk.pos_tag(tokens)
        for (w, tag) in tagged:
            pos_dict[tag].append(w)

    # Save each group of words
    for pos_tag, words_list in pos_dict.items():
        print(f'{pos_tag}: {len(words_list)}')
        filename = f'{args.OUT}/{pos_tag}.json'
        with open(filename, 'w') as f:
            json.dump(words_list, f)

