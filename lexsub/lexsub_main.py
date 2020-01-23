#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
import numpy as np
from collections import defaultdict
import string

def tokenize(s):
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos):
    # TODO Part 1
    possible_synonyms = []
    names = []
    lemma_list = wn.synsets(lemma,pos)
    for i in range(len(lemma_list)):
        l = lemma_list[i].lemmas()
        possible_synonyms.append(l)
    for i in range(len(possible_synonyms)):
        for j in range(len(possible_synonyms[i])):
            name = possible_synonyms[i][j].name()
            names.append(name)
    possible_synonyms = set()
    for i in range(len(names)):
        names[i] = names[i].replace("_", " ")
        if names[i] not in possible_synonyms and names[i] != lemma:
            possible_synonyms.add(names[i])
    return possible_synonyms

def smurf_predictor(context):
    """
    Just suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context):
    # TODO replace for part 2
    frequency = defaultdict(list)
    possible_synonyms = []
    lemma = context.lemma
    lemma = lemma.replace("_", " ")
    lemma_list = wn.synsets(context.lemma, context.pos)
    for i in range(len(lemma_list)):
        l = lemma_list[i].lemmas()
        possible_synonyms.append(l)
    for i in range(len(possible_synonyms)):
        for j in range(len(possible_synonyms[i])):
            name = possible_synonyms[i][j].name()
            freq = possible_synonyms[i][j].count()
            name = name.replace("_", " ")
            if name != lemma:
                frequency[name].append(freq)
    frequency2 = {k: sum(v) for (k, v) in frequency.items()}
    predictor = max(frequency2, key=frequency2.get)
    return predictor

def wn_simple_lesk_predictor(context):
    # TODO replace for part 3
    stop_words = set(stopwords.words('english'))
# Simplified Leak Algorithm
    lemma = context.lemma
    pos = context.pos
    wordform = context.word_form
# tokenize context
    left = context.left_context
    right = context.right_context
    if left == ['None']:
        left = []
    if right == ['None']:
        right = []
    sentence0 = left + [wordform] + right  ## word form
    sentence = []
    for i in range(len(sentence0)):
        a = tokenize(sentence0[i])
        for j in range(len(a)):
            sentence.append(a[j])
    sentence = set(sentence)
    sentence = sentence - stop_words
# gloss
    collection_1 = defaultdict(list)
    collection_2 = defaultdict(int)
    s1 = wn.synsets(lemma, pos)
    for i in range(len(s1)):
        syn = s1[i]
        definition = syn.definition()
        example = syn.examples()
        hyp = syn.hypernyms()
        collection_1[syn].append(definition)
        for q in range(len(example)):
            collection_1[syn].append(example[q])
        if len(hyp) > 0:
            for j in range(len(hyp)):
                example2 = hyp[j].examples()
                definition2 = hyp[j].definition()
                if len(example2) > 0:
                    for p in range(len(example2)):
                        collection_1[syn].append(example2[p])
                if len(definition2) > 0:
                    collection_1[syn].append(definition2)
# overlap
        gloss = collection_1[syn]
        token = []
        for k in range(len(gloss)):
            tok = tokenize(gloss[k])
            for length in range(len(tok)):
                token.append(tok[length])
        token = set(token)
        token = token - stop_words
        overlap = token.intersection(sentence)
        collection_2[syn] = len(overlap)
    key = sorted(collection_2, key=collection_2.get, reverse=True)
    lemmas = key[0].lemmas()
    candidate = defaultdict(int)
    for j in range(len(lemmas)):
        candidate[lemmas[j].name()] = lemmas[j].count()
    candidates = sorted(candidate, key = candidate.get, reverse=True)
    for l in range(len(candidates)):
        if candidates[l] != lemma:
            candidates[l] = candidates[l].replace("_", " ")
            return candidates[l]
    return wn_frequency_predictor(context) # using the method of part II instead of return None


   
class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

    def predict_nearest(self,context):
        # TODO replace for part 4
        lemma = context.lemma
        pos = context.pos
        possible = get_candidates(lemma, pos)
        result = {}
        for element in possible:
            try:
                result[element] = self.model.similarity(lemma, element)
            except KeyError:
                result[element] = 0
        key = sorted(result, key=result.get, reverse=True)
        return key[0]


    def predict_nearest_with_context(self, context):
        # TODO replace for part 5
        stop_words = set(stopwords.words('english'))
        lemma = context.lemma
        pos = context.pos
        possible = get_candidates(lemma, pos)

        left = context.left_context
        right = context.right_context
        left0 = []
        right0 = []
        if left != ['None'] and len(left) > 0:
            for i in range(len(left)):
                l = tokenize(left[i])
                for j in range(len(l)):
                    left0.append(l[j])
        left0 = set(left0)
        left0 = left0 - stop_words
        left0 = list(left0)
        if right != ['None'] and len(right) > 0:
            for i in range(len(right)):
                r = tokenize(right[i])
                for j in range(len(r)):
                    right0.append(r[j])
        right0 = set(right0)
        right0 = right0 - stop_words
        right0 = list(right0)
        sentence = left0 + [lemma] + right0

        k = len(left0)
        candidate_index = k - 5
        sentence_list = []
        for i in range(11):
            if 0 <= candidate_index <= len(sentence) - 1:
                sentence_list.append(sentence[candidate_index])
            candidate_index = candidate_index + 1
        sentence_sum = np.zeros(300)
        for i in range(len(sentence_list)):
            try:
                word = self.model.wv[sentence_list[i]]
                sentence_sum = sentence_sum + word
            except KeyError:
                sentence_sum = sentence_sum + np.zeros(300)
        result = {}
        for element in possible:
            try:
                element_v = self.model.wv[element]
                result[element] = np.dot(sentence_sum, element_v) / (np.linalg.norm(sentence_sum) * np.linalg.norm(element_v))
            except KeyError:
                result[element] = 0
        key = sorted(result, key=result.get, reverse=True)
        return key[0]


if __name__=="__main__":

    # TODO At submission time, this program should run your best predictor (part 6).
    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)

    def improved_predict(context):
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        lemma = context.lemma
        pos = context.pos
        possible = get_candidates(lemma, pos)

        left = context.left_context
        right = context.right_context
        left0 = []
        right0 = []
        if left != ['None'] and len(left) > 0:
            for i in range(len(left)):
                l = tokenize(left[i])
                for j in range(len(l)):
                    l[j] = lemmatizer.lemmatize(l[j]) ### improvement: turn words in context into standard form
                    left0.append(l[j])
        left0 = set(left0)
        left0 = left0 - stop_words
        left0 = list(left0)
        if right != ['None'] and len(right) > 0:
            for i in range(len(right)):
                r = tokenize(right[i])
                for j in range(len(r)):
                    r[j] = lemmatizer.lemmatize(r[j]) ### improvement: turn words in context into standard form
                    right0.append(r[j])
        right0 = set(right0)
        right0 = right0 - stop_words
        right0 = list(right0)
        sentence = left0 + [lemma] + right0

        k = len(left0)
        candidate_index = k - 5
        sentence_list = []
        for i in range(11):
            if 0 <= candidate_index <= len(sentence) - 1:
                sentence_list.append(sentence[candidate_index])
            candidate_index = candidate_index + 1
        sentence_sum = np.zeros(300)
        for i in range(len(sentence_list)):
            try:
                word = predictor.model.wv[sentence_list[i]]
                sentence_sum = sentence_sum + word
            except KeyError:
                sentence_sum = sentence_sum + np.zeros(300)
        result = {}
        for element in possible:
            try:
                element_v = predictor.model.wv[element]
                result[element] = np.dot(sentence_sum, element_v) / (np.linalg.norm(sentence_sum) * np.linalg.norm(element_v))
            except KeyError:
                result[element] = 0
        key = sorted(result, key=result.get, reverse=True)
        return key[0]



    for context in read_lexsub_xml(sys.argv[1]):

        prediction = improved_predict(context)
        #prediction = predictor.predict_nearest_with_context(context)
        #prediction = smurf_predictor(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
