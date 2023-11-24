#modules
import nltk
from nltk.corpus import stopwords

#local methods
from retrieval import training_text_retrieval, test_text_retrieval, random_texts
from naive_bayes_algo import naive_bayes_training, naive_bayes_testing
from arg_parser import parse_args
from evaluation import evaluate


def tokenization(texts):
    '''
    transforms a text into a list of tokens(punctuation is removed)

    :returns: list of token lists
    '''
    token_list = []
    
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    for txt in texts:
        tokens = tokenizer.tokenize(txt)
        tokens = [tkn.lower() for tkn in tokens]
        token_list.append(tokens)

    return token_list


def remove_stop_words(tokens_list):
    '''
    removes stopwords

    :returns: token lists without stopwords
    '''
    
    nltk.download("stopwords", quiet="True")
    stop_words = stopwords.words("english")

    filtered_list = []
    for tokens in range(0, len(tokens_list)):
        for word in tokens_list[tokens]:
            if word.casefold() not in stop_words:
                filtered_list.append(word)
            
        tokens_list[tokens] = filtered_list
        filtered_list = []

    return tokens_list


def stemming(texts):
    '''
    reduces every word to its stem using Porter's stemmer.
    
    :return: stemmed list
    '''

    stemmer = nltk.stem.PorterStemmer()
    for txt in texts:
        for word in range(0, len(txt)):
            txt[word] = stemmer.stem(txt[word])
    
    return texts


def lemmatizing(tokens_list):
    '''
    reduces every word to its lemma using WordNet lemmatizer.
    
    :return: lemmatized list
    '''

    lemmatizer = nltk.stem.WordNetLemmatizer()
    for txt in tokens_list:
        for word in range(0, len(txt)):
            txt[word] = lemmatizer.lemmatize(txt[word])

    return tokens_list


def bag_of_words(tokens_list):
    '''
    transforms a word list following the concept of Bag of Words(BoW).
    the list becomes a set of tuples (word, count)

    :returns: BoW list
    '''

    for txt in range(0, len(tokens_list)):
        tokens_list[txt] = nltk.FreqDist(tokens_list[txt]) 
    
    return tokens_list


def simple_pipeline(texts):
    '''
    performes a normalization of the english texts in input following the pipeline:
        - text tokenization
        - remove stopwords
        - lemmatization or stemming
        - remove non keywords
        - bag of words creation

    :return: BoW for the set of texts
    '''

    #pipeline
    tokens = tokenization(texts)
    tokens = remove_stop_words(tokens)

    if args.preprocessing == 'stems':
        tokens = stemming(tokens)
    if args.preprocessing == 'lemmas':
        tokens = lemmatizing(tokens)

    return tokens


def golden_labels(unlabeled_set, label):
    '''
    attaches the correct label to each bag of words

    :returns: set of tuples (bow, label)
    '''

    labeled_set = [(bow, label) for bow in unlabeled_set]
    
    return labeled_set


def main():
    classes = ['medical', 'non-medical']

    #text retrieval
    '''
    text length are saved since in pre-processing 
    we keep only the words we're interested in
    thus changing the ratio between keywords and the total document words 
    needed for the Naive Bayes classification
    '''
    medical_texts, med_lengths = training_text_retrieval('medical')
    other_texts, oth_lengths = training_text_retrieval('non-medical')

    #texts pre-processing
    medical_tokens = simple_pipeline(medical_texts)
    other_tokens = simple_pipeline(other_texts)
    
    medical_bow = bag_of_words(medical_tokens)
    other_bow = bag_of_words(other_tokens)

    #attach labels to frequency distrubutions
    medical = golden_labels(medical_bow, 'medical')
    non_medical = golden_labels(other_bow, 'non-medical')

    #training set creation
    training_set = medical + non_medical

    # Naive Bayes: TRAINING
    print('\n-- Training --', end='')
    log_prior, log_likelihoods, vocabulary = naive_bayes_training(training_set, sum(med_lengths), sum(oth_lengths), classes)
    print(end='\x1b[2K')
    print('\n-- Training Ended --')

    if args.random:
        # Classify a random page
        test_text,_ = random_texts()
        print('\n-- Processing Random Text --', end='')
        test_tokens = simple_pipeline(test_text)
        test_set = bag_of_words(test_tokens)
        print(end='\x1b[2K')
        print('\n-- Classifying Random Text --')
        naive_bayes_testing(log_prior, log_likelihoods, vocabulary, test_set, classes)  
    elif args.pages:
        # Classify given pages
        test_text,_ = random_texts(args.pages)
        print('\n-- Processing Given Texts --', end='')
        test_tokens = simple_pipeline(test_text)
        test_set = bag_of_words(test_tokens)
        print(end='\x1b[2K')
        print('\n-- Classifying Given Texts --')
        naive_bayes_testing(log_prior, log_likelihoods, vocabulary, test_set, classes)
    else:
        #TEST: text retrieval
        med_test_texts,_ = test_text_retrieval('medical')
        oth_test_texts,_ = test_text_retrieval('non-medical')

        #texts pre-processing
        print('\n-- Processing Test Set --', end='')
        medical_tokens = simple_pipeline(med_test_texts)
        other_tokens = simple_pipeline(oth_test_texts)
        
        medical_bow = bag_of_words(medical_tokens)
        other_bow = bag_of_words(other_tokens)

        #wikipedia test pages
        test_set2 = medical_bow + other_bow
        correct_classes = 10*['medical'] + 10*['non-medical']

        #Nayve Bayes
        print(end='\x1b[2K')
        print('\n-- Classifying Test Set --')
        classifications = naive_bayes_testing(log_prior, log_likelihoods, vocabulary, test_set2, classes)

        #evaluation test
        print('\n-- Evaluating Test Set Classifications --')
        rates = [0,0,0,0]   #TP, TN, FP, FN
        for index, cl in enumerate(classifications):
            if cl[0] == correct_classes[index]:
                if cl[0] == 'medical':
                    rates[0] += 1
                else:
                    rates[1] += 1
            else:
                if cl[0] == 'medical':
                    rates[2] += 1
                else:
                    rates[3] += 1

        print(rates)
        evaluate(rates[0], rates[1], rates[2], rates[3])

    return 0

if __name__ == '__main__':
    args = parse_args()
    main()