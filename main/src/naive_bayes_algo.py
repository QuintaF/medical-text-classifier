'''
methods for the classification
based on the Naive Bayes algorithm
'''
import math

def a_priori(class_docs, total):
    '''
    probability of the class being the correct class

    :returns: a priori probability for a specific class
    '''

    return math.log(class_docs/total)


def likelihood(word_count, total_words_in_class_text, words_in_vocabulary):
    '''
    likelihood, given a class, that a word will belong to it.
    The computation uses laplace(add-1) smoothing 
    to account for unknown words.
    
    :returns: likelihood for a specific word and class
    '''
    return math.log((word_count + 1)/(total_words_in_class_text + words_in_vocabulary))


def create_vocabulary(documents):
    '''
    creates a set containing words present in every document

    :returns: set of words
    '''
    vocab = set()
    for doc in documents:
        for word in doc[0]:
            vocab.add(word)

    return vocab


def word_count(word, big_doc):
    '''
    counts how many times a word occurs in 
    all the documents of a specific class

    :returns: occurences count
    '''
    count = 0
    for bow in big_doc:
        for w, occ in bow.items():
            if w == word:
                count += occ
                break;

    return count


def training(training_set, med_length, oth_length, classes):
    '''
    Training phase for the classifier.
    Using the pre-processed words in the training set it
    computes prior probability for each class P(c), and likelihoods
    for each class and word P(w|c).
    To avoid underflow it uses log values.

    :returns: prior probabilities, likelihoods
    '''
    total_docs = len(training_set) 
    total_words_in_big_doc = {classes[0] : med_length, classes[1] : oth_length} #total words in big_doc for classes

    priors = {}
    likelihoods = {}
    n_doc_c = {} #number of documents for classes
    big_doc = {} #list of FreqDists for classes
    for c in classes:
        n_doc_c[c] = 0 
        big_doc[c] = []
        likelihoods[c] = {}

    for doc in training_set:
        n_doc_c[doc[1]] += 1
        big_doc[doc[1]].append(doc[0])

    for c in classes:
        priors[c] = a_priori(n_doc_c[c], total_docs)

    vocabulary = create_vocabulary(training_set)

    for word in vocabulary:
        for c in classes:
            count = word_count(word, big_doc[c])
            likelihoods[c][word] = likelihood(count, total_words_in_big_doc[c], len(vocabulary))

    return priors, likelihoods, vocabulary


def test_classification(test, log_prior, log_likelihoods, vocabulary, classes):
    '''
    computes the likelihood with which every class 
    could've generated the test document

    :returns: the class that most likely "generated" the test document 
    '''
    sums = {}
    for c in classes:
        sums[c] = log_prior[c]
        
    for word in test:
        if word in vocabulary:
            for c in classes:
                sums[c] += log_likelihoods[c][word]

    max_likelihood = sums[classes[0]]
    max_class = classes[0]
    for i in range(1, len(classes)):
        if max_likelihood < sums[classes[i]]:
            max_likelihood = sums[classes[i]]
            max_class = classes[i]
    
    return max_class, max_likelihood


def naive_bayes_training(training_set, med_length, oth_length, classes):
    '''
    Naive Bayes algorithm: a generative probabilistic classifier.
    Training Phase:
    Works on BoW intuition and Conditional independence.
    Computes: log_prior P(c), log_likelihoods P(w|c), vocabulary V
    
    :returns: log_prior, log_likelihoods, vocabulary
    '''

    return training(training_set, med_length, oth_length, classes)

def naive_bayes_testing(log_prior, log_likelihoods, vocabulary, test_set, classes):
    '''
    Naive Bayes algorithm: a generative probabilistic classifier.
    Test Phase:
    The goal is to classify whether a text is 'medical' or 'non medical'.
    Most fitting class as c = argmax P(x) * P(f|c)

    :returns: list of classifications
    '''
    print('Best Class{:>20}'.format('likelihood\n'))
    classifications = []
    for doc in test_set:
        max_class, max_likelihood = test_classification(doc, log_prior, log_likelihoods, vocabulary, classes)
        classifications.append((max_class,max_likelihood))       
        print('{0:<20}{1}'.format(max_class, max_likelihood))

    return classifications