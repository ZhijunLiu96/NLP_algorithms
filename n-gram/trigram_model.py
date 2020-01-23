import sys
from collections import defaultdict, Counter
import math
import random
import os
import os.path



def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile, 'r') as corpus:
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """
    grams = []
    gram = ()
    if n >= len(sequence):
        pass
        #print("n cannot be larger than the length of sequence")
    else:
        if n == 1:
            sequence.insert(0, "START")
            sequence.append("STOP")
            for i in range(len(sequence)):
                gram = tuple(sequence[i:(i+n)])
                grams.append(gram)
        else:
            start = ["START"]*(n-1)
#             for i in range(len(start)):
#                 sequence.insert(0, start[i])
#             sequence.append("STOP")
            sequence = start + sequence +["STOP"]
            for i in range(0, len(sequence)-n+1):
                gram = tuple(sequence[i:(i+n)])
                grams.append(gram)
    return grams


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)



    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = defaultdict(int) # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)

        ##Your code here
        for sentence in corpus:
            unigram0 = get_ngrams(sentence, 1)
            bigram0 = get_ngrams(sentence, 2)
            trigram0 = get_ngrams(sentence, 3)
#            unigram = unigram.append(unigram0)
#            bigram = bigram.append(bigram0)
#            trigram = trigram.append(trigram0)
            for unigram in unigram0:
                self.unigramcounts[unigram] = self.unigramcounts[unigram] + 1
            for bigram in bigram0:
                self.bigramcounts[bigram] = self.bigramcounts[bigram] + 1
            for trigram in trigram0:
                self.trigramcounts[trigram] = self.trigramcounts[trigram] + 1
#        self.unigramcounts = Counter(unigram)
#        self.bigramcounts = Counter(bigram)
#        self.trigramcounts = Counter(trigram)

        return self.unigramcounts, self.bigramcounts, self.trigramcounts

    def raw_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        if self.bigramcounts[trigram] != 0:
            count_tri = self.trigramcounts[trigram]
            count_bi = self.bigramcounts[trigram[:-1]]
            p = count_tri/count_bi
        else:
            p = 0
        return p

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        if self.unigramcounts[bigram[:-1]] != 0:
            count_bi = self.bigramcounts[bigram]
            count_uni = self.unigramcounts[bigram[:-1]]
            p = count_bi/count_uni
        else:
            p = 0
        return p
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        top = self.unigramcounts[unigram]
        under = sum(self.unigramcounts.values()) - self.unigramcounts[("START",)] # what about start and stop??
        p = top/under
        # hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        return p


    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        p1 = self.raw_trigram_probability(trigram)
        p2 = self.raw_bigram_probability(trigram[:-1])
        p3 = self.raw_unigram_probability(trigram[:-2])
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        p = lambda1*p1 + lambda2*p2 + lambda3*p3
        return p
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        sentence = get_ngrams(sentence, 3)
        p = 0
        for word in sentence:
            p_word = self.smoothed_trigram_probability(word)
            log_p = math.log2(p_word)
            p = p+log_p

        return p

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        log_sum = 0
        m = 0
        for sentence in corpus:
            log_sum = log_sum + self.sentence_logprob(sentence)
            m = m+len(sentence)
        l = log_sum/m
        perplexity = 2**(-l)
        return perplexity


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            total = total + 1
            if pp1 < pp2:
                correct = correct + 1
    
        for f in os.listdir(testdir2):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            total = total + 1
            if pp1 > pp2:
                correct = correct + 1

        accuracy = correct/total
        return accuracy

if __name__ == "__main__":


    # put test code here...
    # >>> python -3 trigram_model.py brown_train.txt
    # >>> python -3 trigram_model.py brown_test.txt
    # you can then call methods on the model instance in the interactive 
    # Python prompt.
    brown_train = "txt"
    brown_test = "txt"


    # Testing perplexity: 
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)
    model = TrigramModel(brown_train)
    dev_corpus = corpus_reader(brown_test, model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(pp)
    # pp = 115.17844420736678


    # Essay scoring experiment:
    train_high = "txt"
    train_low = "txt"
    test_high = "path"
    test_low = "path"
    acc = essay_scoring_experiment(train_high, train_low, test_high, test_low)
    print(acc)
    # acc = essay_scoring_experiment('train_high.txt', 'train_low.txt", "test_high", "test_low")
    # acc = 0.8067729083665338


