import argparse

def parse_args():
    '''
    builds a parser for 
    command line arguments

    :returns: args values
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument("--preprocessing", "-pp", default='lemmas', choices=['lemmas','stems','none'], help="choose to use lemmatization/stemming/none of the two, during pre-processing")

    #If none of the following 2 is given then the default classification is performed(including evaluation)
    #Note that -r subscribes -p, only one is chosen at a time
    parser.add_argument("--random", "-r", action="store_true", help="classify a random Wikipedia page(it's on the user to check if the page is medical visiting the link)")
    parser.add_argument("--pages", "-p", action="store", nargs='*', help="classify 0 or more Wikipedia page(it's on the user to check if the page is medical visiting the links)")

    return parser.parse_args()