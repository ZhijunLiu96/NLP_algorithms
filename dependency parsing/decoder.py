from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])
    def parse_sentence(self, words, pos):
        state = State(range(1, len(words)))
        state.stack.append(0)    

        while state.buffer: 
            #pass
            # TODO: Write the body of this loop for part 4
            if len(state.stack)==0 and len(state.buffer) >=1 :
                state.shift()
            elif len(state.stack)==1 and len(state.buffer) >1 :
                state.shift()
            else:
                feature = self.extractor.get_input_representation(words, pos, state)
                feature = np.array(feature).reshape((-1, 6))
                score = self.model.predict_proba(feature)
                #r = score.index(max(score))
                for i in range(91):
                    if score[0,i] == np.max(score):
                        r = i
                        break
                action = self.output_labels[r]

                if action[0] == 'right_arc':
                    p = state.stack[-1]
                    c = state.buffer[-1]
                    state.right_arc(action[1])
                    state.deps.add((p, c, r))
                elif action[0] == 'left_arc':
                    c = state.stack[-1]
                    p = state.buffer[-1]
                    state.left_arc(action[1])
                    state.deps.add((p, c, r))
                else:
                    state.shift()
        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c, words[c], pos[c], p, r))
        return result 
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
