from sklearn.feature_extraction.text import CountVectorizer

#list of English function words 
FUNCTION_WORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

#for the 7 feature configurations in task 1
def extract_features(train_docs, test_docs, config_type):
    binary_flag = False
    stop_words_list = None
    fit_on_all = False
    
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
        
    # vii) Bag of words binarized with negation feature
    elif config_type == 7:
        binary_flag = True

    #vectorizer to accept pre-tokenized lists
    vectorizer = CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x,binary=binary_flag,stop_words=stop_words_list,token_pattern=None)
    
    if fit_on_all:
        #fit on both train and test sets so test-only words are added to the vocabulary
        vectorizer.fit(train_docs + test_docs)
        X_train = vectorizer.transform(train_docs)
        X_test = vectorizer.transform(test_docs)
    else:
        #standard approach ie fit only on train, automatically ignoring unseen words in test
        X_train = vectorizer.fit_transform(train_docs)
        X_test = vectorizer.transform(test_docs)
        
    return X_train, X_test, vectorizer