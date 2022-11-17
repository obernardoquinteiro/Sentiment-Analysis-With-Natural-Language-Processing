import re
import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

word_types = {
    "Positive" : ["fantastic", "awesome", "great", "good", "nice", "wonderful", "love", "loved", "like", "better", "favourite", "favorite", "amazing"],
    "Negative" : ["awful", "bad", "terrible", "smelly", "disgusting", "disappoint", "disappointed", "worse", "no", "garbage", "sucks", "suck"],
    "DetNeg": ["didn't", "not", "can't", "didnt", "cant", "doesn't", "didn't", "dont", "don't", "won't", "wont", "wouldn't", "wouldnt"],
    "Exponential": ["really", "very", "so", "pretty"]
}
   
def lemmatize (review: str):
    word_lemmatizer = WordNetLemmatizer()
    for elem in review:
        word_lemmatizer.lemmatize(elem)
    return ''.join([str(t) + " " for t in review]) 

def review_processing(review: str):

    punctuation = '''!()-[]{};:'"\,<>./?@#+-$%^&*_~'''
    bar = '\n'
    re_html = re.compile(r'<[^>]+>')
    review = re_html.sub('', review)
    for ele in review:
        if ele in punctuation or ele in bar:
            review = review.replace(ele, "")

            
    tokenized_review = word_tokenize(review)
    
    
    stop_words = set(stopwords.words('english'))
    for word in word_types["DetNeg"]:
        if word in stop_words:
            stop_words.remove(word)

    #review_tokens = []
    small_change = []
    for elem in tokenized_review:

        '''elem = re.sub(r"won’t", "will not", elem)
        elem = re.sub(r"would't", "would not",elem)
        elem = re.sub(r"could't", "could not",elem)
        elem = re.sub(r"\'d", " would",elem)
        elem = re.sub(r"can\'t", "can not",elem)
        elem = re.sub(r"n\'t", " not", elem)
        elem = re.sub(r"\'re", " are", elem)
        elem = re.sub(r"\'s", " is", elem)
        elem = re.sub(r"\'ll", " will", elem)
        elem = re.sub(r"\'t", " not", elem)
        elem = re.sub(r"\'ve", " have", elem)
        elem = re.sub(r"\'m", " am", elem)'''
        l = re.compile(elem.lower())
        r = re.compile(elem)
        lowerelem = elem.lower()
        small_change.append(lowerelem)
        #Keyword tagging from that failed to improve efficiency
        '''if elem not in stop_words:
            if any(l.match(dne) for dne in word_types["DetNeg"]):
                review_tokens.append("DETNEG")

            if any(r.match(po) for po in word_types["Positive"]):
                elem = "POSITIVE"

            elif any(l.match(vpo) for vpo in word_types["Positive"]):
                elem = "VERYPOSITIVE"
            
            elif any(r.match(ne) for ne in word_types["Negative"]):
                elem = "NEGATIVE"

            elif any(l.match(vne) for vne in word_types["Negative"]):
                elem = "VERYNEGATIVE"

            elif any(l.match(exp) for exp in word_types["Exponential"]):
                elem = "EXPONENTIAL"
            
            review_tokens.append(elem)
'''
    
    lemmatized_tokens = lemmatize(small_change)
    return lemmatized_tokens
    

"""
   Separate every review from each file into [type, review] and put
   in a vector containing all reviews
"""
def load_train_info (filename: str):
    datasource = open(filename,"r", encoding="UTF8")
    data = []
    lines = datasource.readlines()
    for line in lines:
        toPut = [line.split("\t")[0], line.split("\t")[1]]
        data.append(toPut)
    return data

def load_test_info (filename: str):
    datasource = open(filename,"r", encoding="UTF8")
    data = []
    lines = datasource.readlines()
    testlines = lines
    if(len(testlines[0].split("\t")) != 1):
        for line in lines:
            toPut = [line.split("\t")[0], line.split("\t")[1]]
            data.append(toPut)
    else:
        for line in lines:
            toPut = line
            data.append(toPut)

    return data

"""
    Parse the input documents to generate the train and test sets in the correct fashion
"""
def parse_docs (args: [str]):
    curr_flag = ""
    train_info, test_info = [], []
    for token in args:
        if token == "-train" or token == "-test":
            curr_flag = token
        else:
            if curr_flag == "-train":
                train_info = load_train_info (filename=token)
            elif curr_flag == "-test":
                test_info = load_test_info (filename=token)
   
    train = {}
    train["reviews"] = [line[1] for line in train_info]
    train["types"] = [line[0] for line in train_info]
    
    test = {}
    if isinstance(test_info[0], str):
        test["reviews"] = [line for line in test_info]

    else:
        test["reviews"] = [line[1] for line in test_info]
        test["types"] = [line[0] for line in test_info] 

       
    return train, test

"""
This method was only used for testing purposes, during development
"""
def accuracy_by_type (test_types, final_predict, test_reviews):
    word_type_count = {"=Excellent=" : 0, "=VeryGood=" : 0, "=Good=" : 0, "=Unsatisfactory=" : 0, "=Poor=" : 0}
    correct_word_type_count = {"=Excellent=" : 0, "=VeryGood=" : 0, "=Good=" : 0, "=Unsatisfactory=" : 0, "=Poor=" : 0}
    total_expected = 0
    total_correct = 0
    for i in range(len(final_predict)):
        pred = final_predict[i]
        expected = test_types[i]
        word_type_count[expected] += 1
        total_expected += 1
        if pred == expected:
            correct_word_type_count[expected] += 1
            total_correct += 1
        '''else:
            print("ERROR")
            print(test_reviews[i])
            print(pred)
            print(expected)
            print("------------")'''
            
    for typ in word_type_count.keys():
        acc=  round(correct_word_type_count[typ] / word_type_count[typ], 3)
        print(f"Accuracy for {typ}: {acc}")

    
    total_accuracy = round(total_correct / total_expected, 3)
    print(f"Accuracy Geral = {total_accuracy}")

def predict_review_types(train, test):
    model = svm.SVC()
    vec = TfidfVectorizer(preprocessor=review_processing, tokenizer=None, ngram_range=(1,2))
    training_features = vec.fit_transform(train["reviews"])
    testing_features = vec.transform(test["reviews"])
    
    model.fit(training_features, train["types"])
    final_predict = model.predict(testing_features)
    #print(accuracy_by_type(test["types"], final_predict, test["reviews"]))

    return final_predict

def run (args):
    if len(args) != 5:
        raise Exception ("Invalid number of arguments.")
    
    train, test = parse_docs(args[1:])
    predictions = predict_review_types(train, test)
    '''print(confusion_matrix(test["types"], predictions))
    print(classification_report(test["types"], predictions))
    print(accuracy_score(test["types"], predictions))'''
    for pred in predictions:
        print(pred)


if __name__ == "__main__":
    run(sys.argv)