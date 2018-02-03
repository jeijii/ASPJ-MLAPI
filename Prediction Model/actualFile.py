from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import logging
import sys
from sklearn import datasets
import pickle
import string
import spacy
import nltk
class topics():

    def __init__(self, topicNumber , wordList):
        self.topicName = topicNumber
        self.wordList = wordList


class classifier():
    def __init__(self , clf , countVector , tfidfVector):
        self.clf = clf
        self.countVector = countVector
        self.tfidfVector = tfidfVector

class tags():
    def __init__(self , tags):
        self.tags = tags
    def addTag(self , tag):
        list = []
        for w in self.tags:
            list.append(w)

        list.append(tag)
        self.tags = list
#Preprocessing methods
def tokenizedDocuments(documents):
    list = []
    documents = documents.lower()
    token = word_tokenize(documents)
    return token

#Preprocessing methods
def removeStopWords(documents):
    list = []
    sw = set(stopwords.words('English'))
    for w in documents:
        if(w not in sw ):
            list.append(w)
    return list
#Preprocessing methods
def removePunctuation(document):
    exclude = set(string.punctuation)
    list = []
    for w in document:
        if(w not in exclude):
            list.append(w)

    return list

def removeNumbers(document):
    myList = ([x for x in document if not isinstance(x, int)])
    myList = ([x for x in document if not isinstance(x, float)])
    return myList

# def removeShortWords(document):
#     list =[]
#     for w in document:
#         if(len(w) > 2):
#             list.append(w)
#     return list

def performPOSLemmatization(document):
    lemma = WordNetLemmatizer()
    posTagDescription = []
    bagOfWordsList = []
    finalWordsList = []

    word = nltk.pos_tag(document)
    # chunkGram = r"""Chunk: {<.*>}
    #                            }<VB.?|IN.?|DT|RB.?>{"""
    chunkGram = r"""Chunk: {<NN.?>}"""
    chunkParse = nltk.RegexpParser(chunkGram)
    chunked = chunkParse.parse(word)
    words = []
    pos = []
    for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
        words.append(" ".join([a for (a, b) in subtree.leaves()]))
        pos.append(" ".join([b for (a, b) in subtree.leaves()]))

    bagOfWordsList.append(words)
    posTagDescription.append(pos)
    for (set, posSet) in zip(bagOfWordsList, posTagDescription):
        fakeList = []
        for (w, pos) in zip(set, posSet):
            value = get_wordnet_pos(pos)

            if (value != ''):
                try:
                    lemmatizedWords = lemma.lemmatize(w, value)
                    fakeList.append(lemmatizedWords)
                except:
                    print(pos)
        finalWordsList= fakeList

    return finalWordsList

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

def preprocessData(question):
    print("Tokenizing the question")
    question  = tokenizedDocuments(question)
    print("Performing POS and Lemmatization")
    question = performPOSLemmatization(question)
    print("Remove Stop Words")
    question = removeStopWords(question)
    print("Removing Punctuation")
    # question = removePunctuation(question)
    # print("Remove Short Words")
    # question = removeShortWords(question)
    print("Remove Numbers")
    question = removeNumbers(question)
    print(question)
    return question



def saveClassifier(trained_classifer):
    with open("classifier", "wb") as fp:  # Pickling
        pickle.dump(trained_classifer, fp)

def loadClassifier():
    try:
        with open("classifier", "rb") as fp:  # Unpickling
            b = pickle.load(fp)
            return b
    except:
    #Use sklearn dataset
        training_set = fetch_20newsgroups(data_home="data", subset='train', remove=('headers', 'footers', 'quotes'),
                                      shuffle='true')
        #Use my own dataset on food
        bunch = load_files('Dictionary')
        trained_classifer = trainingClassifer(training_set , bunch)
        saveClassifier(trained_classifer)
        return trained_classifer



def trainingClassifer(training_set , bunch):
    dataList = []
    newList = []
    list = []
    for data in bunch.data:
        string = data
        sentence = None
        string = str(string)
        list = []
        for i in string.split():
            if i.isalnum():
                sentence = ''.join(i)
                list.append(sentence)

        newList.append(' '.join(list))

    full_set = []
    for w in training_set.data:
        full_set.append(w)
    for d in newList:
        full_set.append(d)

    counter = 0
    for w in training_set.target:
        if w > counter:
            counter = w

    #
    foodLength = len(bunch.target)
    newList = []
    for value in training_set.target:
        newList.append(value)
    for i in range(foodLength):
        value = counter + 1
        newList.append(value)

    # creating vector
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(full_set)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    clf = MultinomialNB()
    clf.fit(X_train_tfidf, newList)

    classfierObject = classifier(clf , count_vect , tfidf_transformer)
    return classfierObject

def predictQuestion(question1, clf):
    training_set = fetch_20newsgroups()
    X_new_counts = clf.countVector.transform(question1)
    tfidf_transformer = TfidfTransformer()
    X_new_tfidf = clf.tfidfVector.transform(X_new_counts)

    predicted = clf.clf.predict(X_new_tfidf)
    return predicted
    # for doc, category in zip(question1, predicted):
    #     return category
        # print('%r => %s' % (doc, training_set.target_names[category]))

def loadTags(category , question):
    for w in category:
        tagSet = []
        if int(w) is 0:
            tagSet.append(tags(['Alternate', 'Religion']))
        elif int(w) is 1:
            tagSet.append(tags(['Computer', 'Graphics']))
        elif int(w) is 2:
            tagSet.append(tags(['Computer', 'Utilities']))
        elif int(w) is 3:
            tagSet.append(tags(['Computer', 'IBM', 'Personal Computer', 'System', 'Hardware']))
        elif int(w) is 4:
            tagSet.append(tags(['Computer', 'System', 'macOs', 'Hardware']))
        elif int(w) is 5:
            tagSet.append(tags(['Computer', 'System', 'Windows']))
        elif int(w) is 6:
            tagSet.append(tags(['Sales']))
        elif int(w) is 7:
            tagSet.append(tags(['Automobile', 'Vehicle']))
        elif int(w) is 8:
            tagSet.append(tags(['Vehicles', 'Motorcycle', 'Automobile']))
        elif int(w) is 9:
            tagSet.append(tags(['Recreation', 'Sports', 'Baseball']))
        elif int(w) is 10:
            tagSet.append(tags(['Recreation', 'Sports', 'Hockey']))
        elif int(w) is 11:
            tagSet.append(tags(['Cryptography', "Science"]))
        elif int(w) is 12:
            tagSet.append(tags(['Electronics']))
        elif int(w) is 13:
            tagSet.append(tags(['Medicine', 'Health']))
        elif int(w) is 14:
            tagSet.append(tags(['Space']))
        elif int(w) is 15:
            tagSet.append(tags(['Religion', 'Christianity']))
        elif int(w) is 16:
            tagSet.append(tags(['Politics', 'Guns', 'Firearms']))
        elif int(w) is 17:
            tagSet.append(tags(['Politics', 'Middle East']))
        elif int(w) is 18:
            tagSet.append(tags(['Politics']))
        elif int(w) is 19:
            tagSet.append(tags(['Religion']))
        elif int(w) is 20:
            tagSet.append(tags(['Food']))



    tokens = []
    for w in question1:
        tokens = preprocessData(w)

    list = []
    for w in tagSet:
        for tag in w.tags:
            list.append(tag)

    for w in tokens:
        w = w.capitalize()
        if w not in list:
            list.append(w)


    return list


logging.basicConfig(filename='dataLog.log',level=logging.DEBUG,format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logging.StreamHandler(sys.stdout)
logging.info("log started")
try:
    question = sys.argv[1]
    clf = loadClassifier()
    question1 = [question]

    category = predictQuestion(question1 , clf)
    tag = loadTags(category , question1)
    print(tag)

except:
    logging.info('Error parsing data')
    print('Error predicting question')

