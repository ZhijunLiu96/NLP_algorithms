import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg
import math

### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        # TODO, part 2
        table = {}
        cache = {}
        rhs_rules = self.grammar.rhs_to_rules
        length = len(tokens)

        for j in range(1, length+1):
            rules = rhs_rules[(tokens[j-1],)]
            for rule in rules:
                left = rule[0]
                right = rule[1]
                cache[left] = right[0]
            table[(j-1,j)] = cache
            cache = {}

        for i in range(2, length+1): #length
            for j in range(i, length+1): # column
                # table[(j,row)]
                row = j-i
                for k in range(row+1, j):
                    component1 = table.get((row, k))
                    component2 = table.get((k, j))
                    if component1 and component2:
                        element1 = component1.keys()
                        element2 = component2.keys()
                        possible_grammar = []
                        for ele1 in element1:
                            for ele2 in element2:
                                possible_grammar.append((ele1, ele2))
                        for element in possible_grammar:
                            if rhs_rules[element]:
                                r = rhs_rules[element]
                                for rule in r:
                                    cache[rule[0]] = rule[1]
                table[(row, j)] = cache
                cache = {}
        #print(table)
        if table.get((0, length)):
            sentence = table.get((0, length))
            if sentence['TOP']:
                return True
            elif sentence['S']:
                return True
        else:
            return False

       
    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        # TODO, part 3
        table = {}
        cache = {}
        rhs_rules = self.grammar.rhs_to_rules
        length = len(tokens)
        probs = {}
        prob_cache = {}

        for i in range(1, length+1):
            rules = rhs_rules[(tokens[i-1],)]
            for rule in rules:
                left = rule[0]
                right = rule[1]
                cache[left] = right[0]
                prob_cache[left] = math.log(rule[2])
            table[(i-1,i)] = cache
            probs[(i-1,i)] = prob_cache
            cache = {}
            prob_cache = {}
        #print(probs)
        #print(prob_cache)

        for i in range(2, length+1): #length
            for j in range(i, length+1): # column
                # table[(j,row)]
                row = j-i
                for k in range(row + 1, j):
                    component1 = table.get((row, k))
                    component2 = table.get((k, j))
                    if component1 and component2:
                        element1 = component1.keys()
                        element2 = component2.keys()
                        possible_grammar = []
                        for ele1 in element1:
                            for ele2 in element2:
                                possible_grammar.append((ele1, ele2))
                        for element in possible_grammar:
                            if rhs_rules[element]:
                                r = rhs_rules[element]
                                for rule in r:
                                    cache[rule[0]] = ((rule[1][0], row, k),(rule[1][1], k, j))
                                    #print(rule[2])
                                    prob_cache[rule[0]] = math.log(rule[2]) + probs[(row, k)][rule[1][0]] + probs[(k, j)][rule[1][1]]
                table[(row, j)] = cache
                cache = {}
                probs[(row, j)] = prob_cache
                prob_cache = {}
        #print(probs)
        if check_probs_format(probs) and check_table_format(table):
            return table, probs
        else:
            print("wrong format")


def get_tree(chart, i,j,nt): 
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4

    if isinstance(chart[(i, j)][nt], str):
        return(nt, chart[(i,j)][nt])
    elif isinstance(chart[(i, j)][nt], tuple):
        left_i = chart[(i, j)][nt][0][1]
        left_j = chart[(i, j)][nt][0][2]
        left_nt = chart[(i, j)][nt][0][0]
        right_i = chart[(i, j)][nt][1][1]
        right_j = chart[(i, j)][nt][1][2]
        right_np = chart[(i, j)][nt][1][0]
        return (nt, get_tree(chart,left_i, left_j, left_nt), get_tree(chart, right_i, right_j, right_np))
        #return chart

if __name__ == "__main__":
    
    with open('atis3.pcfg', 'r') as grammar_file:
        grammar = Pcfg(grammar_file) 
        parser = CkyParser(grammar)
        toks =['flights', 'from', 'miami', 'to', 'cleveland', '.']
        print(parser.is_in_language(toks))
        table, probs = parser.parse_with_backpointers(toks)
        print(get_tree(table, 0, 6, 'TOP'))
        chart = get_tree(table, 0, 6, 'TOP')
        assert check_probs_format(probs)
        assert check_table_format(table)

        
