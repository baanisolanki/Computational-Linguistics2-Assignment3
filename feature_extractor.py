import os
from scipy.sparse import hstack, csr_matrix  
#hstack for hrizontal stacking (horizontal combining)
#using sparse csr_matrix s it doesn't keep null rows; saves space
from sklearn.feature_extraction.text import CountVectorizer #gives us the document-term matrix

#list of English function words 
FUNCTION_WORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

def load_lexicon(filepath):
    #helper func to load opinion words from Hu and Liu lexicon, ignoring comments
    words = set() #set for membership checking; usews hashing faster than list
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if not line.startswith(';') and line.strip(): #comment lines begin with ; so skipping those
                    words.add(line.strip())
    return words

#for the 7 feature configurations in task 1 (plus config 8 for extra credit)
def extract_features(train_docs, test_docs, config_type):
    binary_flag = False #if binary then just presence or absence, not frequency
    stop_words_list = None
    fit_on_all = False #dont exclude test words
    
    #bag of words excluding unseen words in the test set
    if config_type == 1:
        pass 
        
    #bag of words including unseen words in the test set
    elif config_type == 2:
        fit_on_all = True 
        
    #bag of words using word frequency as 1 ie binarization
    elif config_type == 3:
        binary_flag = True
        
    #cntent word frequencies. ignore function words and unseen words
    elif config_type == 4:
        stop_words_list = FUNCTION_WORDS
        
    #content word frequencies of 1 per word; ignoring function words
    elif config_type == 5:
        binary_flag = True
        stop_words_list = FUNCTION_WORDS
        
    #bag of words with negation feature
    #fconfigs 6 and 7, train_docs and test_docs must be passed in AFTER running the apply_negation() function from main.py
    elif config_type == 6:
        pass
        
    #bag of words binarized with negation feature
    elif config_type == 7:
        binary_flag = True
        
    #extra credit, BOW + polarity lexicon features
    elif config_type == 8:
        pass

    #vectorizer
    #already tokenized so doesn't tokenize again; same with preprocessing; flags for binarization and stop words included or not; not token pattern casue I am coding my own tokenizesrs
    vectorizer = CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x,binary=binary_flag,stop_words=stop_words_list,token_pattern=None)
    
    if fit_on_all:
        #fit on both train and test sets so test-only words are added to the vocabulary
        #fit learns vocab, transform uses this vocb to give us the document vectors
        vectorizer.fit(train_docs + test_docs)
        X_train = vectorizer.transform(train_docs)
        X_test = vectorizer.transform(test_docs)
    else:
        #standard approach ie fit only on train, automatically ignoring unseen words in test
        X_train = vectorizer.fit_transform(train_docs)
        X_test = vectorizer.transform(test_docs)
        
    # for extra credit
    if config_type == 8:
        pos_words = load_lexicon('opinion_lexicon/positive-words.txt') 
        neg_words = load_lexicon('opinion_lexicon/negative-words.txt')
        
        def get_lexicon_features(docs):
            features = []
            for doc in docs:
                pos_count = 0
                neg_count = 0

                for token in doc:
                    if token in pos_words: #count of number of tokens in positive docs
                        pos_count += 1

                    if token in neg_words: #count for negative
                        neg_count += 1

            features.append([pos_count, neg_count])#appending these features
            return csr_matrix(features) #sparse matrix of features
        
        train_lex_feat = get_lexicon_features(train_docs)
        test_lex_feat = get_lexicon_features(test_docs)
        
        # append the new positive/negative counts as new columns to the sparse matrix
        X_train = hstack([X_train, train_lex_feat]).tocsr()
        X_test = hstack([X_test, test_lex_feat]).tocsr()
        
    return X_train, X_test, vectorizer