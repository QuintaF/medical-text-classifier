# medical-text-classifier
 First assignment of the NLP course. Attribute to a given english text in input a class among the following: medical, non-medical.

## Usage
Start the algorithm:
```
usage: main.py [-h] [--preprocessing {lemmas,stems,none}] [--random] [--pages [PAGES ...]]
```
Usage options:
```
options:
  -h, --help            show this help message and exit
  --preprocessing {lemmas,stems,none}, -pp {lemmas,stems,none}
                        choose to use lemmatization/stemming/none of the two, during pre-processing
  --random, -r          classify a random Wikipedia page(it's on the user to check if the page is medical visiting the link)
  --pages [PAGES ...], -p [PAGES ...]
                        classify 0 or more Wikipedia page(it's on the user to check if the page is medical visiting the links)
```

### Classification
The default execution, without using --random or --pages, classifies a set of 10 medical and 10 non-medical texts and gives back some evaluation parameters(precision, recall, accuracy and f-score).<br>
Using --random or --pages will not provide an evaluation for the classifier.

## Repository Structure

main/\
└── src/\
&emsp;&emsp;&emsp;├── argparser.py&emsp;...python parser for command line arguments.\
&emsp;&emsp;&emsp;├── evaluation.py&emsp;...functions to compute precision,recall and accuracy.\
&emsp;&emsp;&emsp;├── main.py&emsp;...the execution pipeline.\
&emsp;&emsp;&emsp;├── naive_bayes_algo.py&emsp;...functions for Naive Bayes training and testing.\
&emsp;&emsp;&emsp;└── retrieval.py&emsp;...functions for Wikipedia texts retrieval.           

## Assignment
The classification of texts using wikipedia.
The problem to solve is attributing to a text given in input (only English language) a class among two: medical/non-medical. The implementation could be performed by using, at your choice, OpenNLP (Java library), NLTK (Python library), SpaCy technology https://spacy.io/, a pre-trained technology for pipelines in Python, or the GATE Java technology that is online at https://gate.ac.uk/. The usage of pre-annotated texts (in Wikipedia there is a set of annotations you will find within the text itself, that you can use as means to identify medical documents and separate them from non-medical) can be the base of any implementation admissible. You will be permitted to implement solutions based on Naive Bayes methods, both on Bag Of Words without any pre-processing or pre-processed by the SnowBall stop word list, stemming methods based on Porter's algorithm and finally lemmatization based on the WordNet Lemmatizer. You will also be permitted to implement a solution based on Logistic Regression approach, again with a feature extraction based on pre-processing with a Naive classifier, or extracted directly from Wikipedia based on the annotated keywords. Also in this case, texts could be pre-processed or not. Notice that to access wikipedia files you could use API that are well documented, including in the documentation itself the rules and the netiquette constraints to take into consideration. You can find a direct link to the rules, access to the API and methods to implement API invocation at https://www.mediawiki.org/wiki/API:Main_page.

---