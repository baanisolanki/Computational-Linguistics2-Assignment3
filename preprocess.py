import os
import nltk
from nltk.tokenize import word_tokenize #using punkt tokenizer 

def load_data(data_dir): #data_dir shall have path
    #loading the text files from pos and neg and returning a list of labels, 1 is pos, 0 is neg
    documents = []
    labels = []
    
    for label_type, label_val in [('pos', 1), ('neg', 0)]: #creting path based on +ve or -ve
        folder_path = os.path.join(data_dir, label_type)
        for filename in os.listdir(folder_path): #gives names of the files
            if filename.endswith(".txt"):
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f: #complete file path, opening in read mode,encoding
                    # The dataset is already down-cased with one sentence per line
                    text = f.read()
                    documents.append(text)
                    labels.append(label_val)
                    
    return documents, labels

def apply_negation(tokens):
    #if negaton is applied, the prepends 'NOT_' to every word after a logical negation until the next punctuation mark.

    negation_words = {"n't", "not", "no", "never"}
    punctuation = {".", ",", "?", "!", ";", ":", "-", "--", "(", ")"}
    
    processed_tokens = []
    negation_active = False
    
    for token in tokens:
        #on hitting punctuation, turn off the negation flag
        if token in punctuation:
            negation_active = False
            processed_tokens.append(token)
            continue
            
        #applying if the flag is active
        if negation_active:
            processed_tokens.append(f"NOT_{token}")
        else:
            processed_tokens.append(token)
            
        #if the current token is a negation word, activate the flag
        if token in negation_words:
            negation_active = True
            
    return processed_tokens

def preprocess_text(text, use_negation=False):
    #tokenizes the text and aplies negation if needed
    #tokenize to properly separate words and punctuation
    tokens = word_tokenize(text) #from nltk library
    
    if use_negation:
        tokens = apply_negation(tokens)
        
    return tokens
