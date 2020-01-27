# NLP_algorithms
NLP algorithms, including n-gram models, CKY algorithm, dependency parsing, and an algorithm to find Lexical Substitution

## [N-gram algorithm](https://github.com/ZhijunLiu96/NLP_algorithms/blob/master/n-gram/trigram_model.py)
1. Read corpus, get a list of n-gram tuples;
2. Create a n-gram object: use dictionaries to store unigram, bigram, and trigram counts, calculate raw (unsmoothed) unigram, bigram, and trigram probability, sentence logprobability and perplexity;
3. Grade essays and calculate accuracy.

(Perplexity(per word) measures how well the ngram models predict the sample, which is defined as <img src="https://latex.codecogs.com/svg.latex?\Large&space;2^{-l}" title="\Large 2^{-l}" />)
<img src="https://github.com/ZhijunLiu96/NLP_algorithms/blob/master/n-gram/perplexity.png" hight = "20%" width = "20%">

## [Cocke-Kasami-Younger (CKY) algorithm](https://github.com/ZhijunLiu96/NLP_algorithms/tree/master/CKY)
1. [grammar.py](https://github.com/ZhijunLiu96/NLP_algorithms/blob/master/CKY/grammar.py)
: Create an object Pcfg to read the rules of grammar and probabilities, store them in a dictionary whose values are lists of tuples, and verify if the grammar is a valid PCFG in CNF;
2. [cky.py](https://github.com/ZhijunLiu96/NLP_algorithms/blob/master/CKY/cky.py)
: A bottom-up algorithm uses dictionary (key is a tuple to denote the location in the matrix) to parse sentences in Chomsky Normal Form (CNF);
3. [evaluate_parser.py](https://github.com/ZhijunLiu96/NLP_algorithms/blob/master/CKY/evaluate_parser.py)
: It then compares the predicted tree against the target tree by precision, recall, and F-score.


## [Dependency parsing](https://github.com/ZhijunLiu96/NLP_algorithms/tree/master/dependency%20parsing)
1. [conll_reader.py](https://github.com/ZhijunLiu96/NLP_algorithms/blob/master/dependency%20parsing/conll_reader.py)
: Read and store the dependency relations into a tree structure (DependencyStructure); 
2. [get_vocab.py](https://github.com/ZhijunLiu96/NLP_algorithms/blob/master/dependency%20parsing/get_vocab.py)
: Generate an index of words and POS indices;
3. [extract_training_data.py](https://github.com/ZhijunLiu96/NLP_algorithms/blob/master/dependency%20parsing/extract_training_data.py)
: Create an object State, initialize a stack and a buffer, and define functions (shift, left_arc, right_arc) for the dependency parsing; Create a class FeatureExtractor to generate the input and output for the neural network algorithm;
4. [decoder.py](https://github.com/ZhijunLiu96/NLP_algorithms/blob/master/dependency%20parsing/decoder.py)
: Create an object Parser which returns the result of dependency parsing in the format of DependencyStructure;
5. [train_model.py](https://github.com/ZhijunLiu96/NLP_algorithms/blob/master/dependency%20parsing/train_model.py)
: Use package keras to train a neural network model;
6. [evaluate.py](https://github.com/ZhijunLiu96/NLP_algorithms/blob/master/dependency%20parsing/evaluate.py)
: Calculate the accuracy of both unlabeled and labeled pairs.

## [Lexical substitution](https://github.com/ZhijunLiu96/NLP_algorithms/tree/master/lexsub)
[lexsub_main.py](https://github.com/ZhijunLiu96/NLP_algorithms/blob/master/lexsub/lexsub_main.py): 
1. Define functions to tokenize words, get synonyms from wordnet, and predict the possible synonym with the highest total occurence frequency;
2. Simplified Leak Algorithm: Use dictionaries to create bags of words for synonyms by using definitions and examples, and compute the overlap between the definition of the synset and the context of the target word;
3. Most Similar Synonym: Use word2vec model to calculate the similarity of each cadidates, and return the most similar one;
4. Predict the nearest with context: Create a single vector for the target word and its context by summing together the vectors for all words in the sentence, obtaining a single sentence vector; Measure the similarity of the potential synonyms to this sentence vector;
5. Improved algorithm: Improve Simplified Leak Algorithm by turning all words in the context into standard formats.
