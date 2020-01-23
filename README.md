# NLP_algorithms
NLP algorithms, including n-gram models, CKY algorithm, dependency parsing, and an algorithm to find Lexical Substitution

## [n-gram algorithm](https://github.com/ZhijunLiu96/NLP_algorithms/blob/master/n-gram/trigram_model.py)
1. Read corpus, get a list of n-gram tuples;
2. Create a n-gram object: use dictionaries to store unigram, bigram, and trigram counts, calculated raw (unsmoothed) unigram, bigram, and trigram probability, sentence logprobability and perplexity;
3. Grade essays and calculate accuracy.

(Perplexity(per word) measures how well the ngram models predicts the sample, which is defined as <img src="https://latex.codecogs.com/svg.latex?\Large&space;2^{-l}" title="\Large 2^{-l}" />)
<img src="https://github.com/ZhijunLiu96/NLP_algorithms/blob/master/n-gram/perplexity.png" hight = "20%" width = "20%">

## [CKY algorithm](https://github.com/ZhijunLiu96/NLP_algorithms/tree/master/CKY)
1. [grammar.py](https://github.com/ZhijunLiu96/NLP_algorithms/blob/master/CKY/grammar.py)
Create an object Pcfg to read the rules of grammar and probabilities
