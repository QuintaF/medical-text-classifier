import wikipedia 
import wikipediaapi


#wikipedia object to extract informations
wiki_wiki = wikipediaapi.Wikipedia(
        user_agent =  'medical-text-classifier',
        language = 'en', 
        extract_format= wikipediaapi.ExtractFormat.WIKI)


def random_texts(wiki_pages=None):
    '''
    retrieves a random text or a list of given texts
    using the wikipedia-api.

    :returns: random text
    '''
    if wiki_pages:
        print('\nPages to classify:')
    else:
        wiki_pages = [wikipedia.random(pages=1)]
        print('\nRandom wikipedia page to classify:')

    return text_retrieval(wiki_pages)


def training_text_retrieval(_class):
    '''
    retrieves medical texts using the wikipedia-api.
    
    :returns: list of medical related texts
    '''
    if _class == 'medical':
        wiki_pages = ["Anatomy", "Medicine", "Electrocardiography", "Medical specialty", "Genetics", "Neuroscience", "Pharmacology", "Toxicology", "Immunology", "Pathology"]
        print("\nMedical training pages retrieved:")
    else:
        wiki_pages =["Codex Seraphinianus", "Jesus of Nazareth", "Rick and Morty", "Text corpus", "The Legend of Zelda", "Computer science", "Cthulhu", "Iron Maiden", "Ferrari", "Biology"]
        print("\nNon-Medical training pages retrieved:")
    
    return text_retrieval(wiki_pages)


def test_text_retrieval(_class):
    '''
    retrieves medical or non-medical texts using the wikipedia-api.

    :returns: list of medical or non-medical related texts
    '''
    if _class == 'medical':
        wiki_pages = ["Radiology", "Biochemistry", "Myocardial_infarction", "Embryology", "Cytopathology", "Gastroenterology", "Surgery", "Infectious_diseases_(medical_specialty)", "Ophthalmology", "Transfusion_medicine"]
        print("\nMedical test pages retrieved:")
    else:
        wiki_pages = ["University", "The_Sopranos", "John_Fitzgerald_Kennedy", "Telecommunications", "Neuromancer", "Scrubs_(TV_series)", "René_Magritte", "Post-World_War_II_economic_expansion", "Soul_Nomad_%26_the_World_Eaters", "Khao_Sok_National_Park"]
        print("\n Non-Medical test pages retrieved:")
    
    return text_retrieval(wiki_pages)


def text_retrieval(wiki_pages):
    '''
    retrieves given pages using the wikipedia-api.

    :returns: list of texts, texts lengths
    '''

    page_texts = []
    texts_length =[]
    for page in wiki_pages:
        text = wiki_wiki.page(page).text
        page_texts.append(text)
        texts_length.append(len(text))
        print("\t- https://en.wikipedia.org/wiki/" + str(page).replace(' ','_'))
    
    return page_texts, texts_length