import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from preprocess import load_data, preprocess_text
from feature_extractor import extract_features
from classifiers import CustomNaiveBayes, CustomLogisticRegression

def save_misclassifications(y_true, y_pred, raw_docs, model_name, filename="fumbles.txt", num_samples=3): #fumbled sentences added to fumbles.txt for task4

    false_positives = []
    false_negatives = []
    
    #ensure y_true is a list for easy indexing
    y_true = list(y_true)
    
    for i in range(len(y_true)):
        if y_true[i] == 0 and y_pred[i] == 1:
            false_positives.append(raw_docs[i])
        elif y_true[i] == 1 and y_pred[i] == 0:
            false_negatives.append(raw_docs[i])
            
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"---  :)  {model_name}   [; ---\n")
        f.write(f"{'='*50}\n")
        
        f.write("\n[ False Positives (True: Neg, Predicted: Pos) ]\n")
        for doc in false_positives[:num_samples]:
            #truncating to 500 characters so the file doesn't become overwhelmingly huge
            f.write(f"- {doc[:500]}...\n\n")
            
        f.write("\n[ False Negatives (True: Pos, Predicted: Neg) ]\n")
        for doc in false_negatives[:num_samples]:
            f.write(f"- {doc[:500]}...\n\n")

def main():
    #clearing the fumbles file at the start of the run
    open("fumbles.txt", "w", encoding="utf-8").close()

    print("Loading data...")
    docs, labels = load_data('reviews_data/data') #data stored here; relative path
    
    #splitiing the 2000 files into 80% training and 20% testing sets
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(docs, labels, test_size=0.2, random_state=42)
    
    for config_type in range(1, 8):
        print(f"\n{'='*50}")
        print(f"RUNNING CONFIGURATION {config_type}")
        print(f"{'='*50}")
        
        # Task 2 configs 6 and 7 require the negation feature
        use_neg = True if config_type in [6, 7] else False
        
        print("Preprocessing and tokenizing text...")
        X_train_tok = [preprocess_text(doc, use_negation=use_neg) for doc in X_train_raw]
        X_test_tok = [preprocess_text(doc, use_negation=use_neg) for doc in X_test_raw]
        
        print("Extracting features...")
        X_train_vec, X_test_vec, _ = extract_features(X_train_tok, X_test_tok, config_type)
        
        #evaluating Naive Bayes 
        print("\n[ Custom Naive Bayes ]")
        nb = CustomNaiveBayes()
        nb.fit(X_train_vec, y_train)
        nb_preds = nb.predict(X_test_vec)
        
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, nb_preds))
        print("\nMetrics (Precision, Recall, F1, Accuracy):")
        # classification_report automatically computes precision, recall, f1, and accuracy
        print(classification_report(y_test, nb_preds, digits=4))
        
        #saving naive bais fumbles
        save_misclassifications(y_test, nb_preds, X_test_raw, f"Naive Bayes (Config {config_type})")
        
        #evaluate logistic regression 
        print("\n[ Custom Logistic Regression (SGD) ]")
        #gotta tune learning_rate and epochs for optimal convergence
        lr = CustomLogisticRegression(learning_rate=0.01, epochs=20)
        lr.fit(X_train_vec, y_train)
        lr_preds = lr.predict(X_test_vec)
        
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, lr_preds))
        print("\nMetrics (Precision, Recall, F1, Accuracy):")
        print(classification_report(y_test, lr_preds, digits=4))

        #saving logistic fumbles
        save_misclassifications(y_test, lr_preds, X_test_raw, f"Logistic Regression (Config {config_type})")

if __name__ == '__main__':
    main()