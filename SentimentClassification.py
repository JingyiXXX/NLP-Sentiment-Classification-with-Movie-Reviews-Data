# -*- coding: utf-8 -*-
"""
Author: Jingyi Xiao

"""

# import packages
import numpy as np

import nltk
from nltk.text import Text
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import wordnet
nltk.download('wordnet') 
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw')

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer, TfidfTransformer 
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV

# parameters
seed = 2021
test_size = 0.3

#file path
# pos_file_path =  "/drive/My Drive/a1_dataset/rt-polarity.pos"
# neg_file_path =  "/drive/My Drive/a1_dataset/rt-polarity.neg"
# if you are reading from your local directory, use:
pos_file_path =  "./rt-polaritydata/rt-polarity.pos"
neg_file_path =  "./rt-polaritydata/rt-polarity.neg"

# mount to google drive
# from google.colab import drive
# drive.mount('/drive')

def load_data(file_path, enc = 'latin-1'):
    """
    Load data from the destination (google drive).
    """
    text = list(open(file_path, "r", encoding=enc).readlines())
    text = [s.strip() for s in text]
    return text

positive_reviews = load_data(file_path=pos_file_path, enc='latin-1')
negative_reviews = load_data(file_path=neg_file_path, enc='latin-1')

# Class is balanced
print(f'There are {len(positive_reviews)} in positive reviews. The frist review is: "{positive_reviews[0]}"')
print(f'There are {len(negative_reviews)} in negative reviews. The frist review is: "{negative_reviews[0]}"')

# label data
def label_data(reviews_data, assign_label):
    """
    Assign labels to reviews.
    positive reviews are assigned with label 1.
    negative reviews are assigned with label 0.

    reviews_data: list of sentences.
    assign_label: 0 or 1.
    """

    labels = [assign_label for review in reviews_data]
    return [reviews_data, labels]

positive_reviews, pos_y = label_data(reviews_data=positive_reviews, assign_label=1)
negative_reviews, neg_y = label_data(reviews_data=negative_reviews, assign_label=0)

class SentimentClassification:
    def __init__(self, 
                 positive_reviews=positive_reviews, 
                 negative_reviews=negative_reviews, 
                 pos_y=pos_y, 
                 neg_y=neg_y, 
                 seed=seed, 
                 test_size=test_size):
      

        self.positive_reviews = positive_reviews
        self.negative_reviews = negative_reviews
        self.pos_y = pos_y
        self.neg_y = neg_y
        self.seed = seed
        self.test_size = test_size
        self.stop_words = set(stopwords.words("english"))


#
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# ------------------------------This part is for preprocessing decisions---------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
#

    def tokenWithPunctuationRemoval(self):
        '''
        preprocessing decision a)   
        '''
        tokenizer = RegexpTokenizer(r'\w+')
        positive_reviews_words = [tokenizer.tokenize(review) for review in self.positive_reviews]
        negative_reviews_words = [tokenizer.tokenize(review) for review in self.negative_reviews]
        return positive_reviews_words, negative_reviews_words
    
    def getPosTag(self, positive_reviews_words, negative_reviews_words):
        positive_pos = [pos_tag(review) for review in positive_reviews_words]
        negative_pos = [pos_tag(review) for review in negative_reviews_words]
        return positive_pos, negative_pos
    
    
    def getWordnetPos(self, treebank_tag):
        '''
        return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v) 

        This code refers:
        https://newbedev.com/wordnet-lemmatization-and-pos-tagging-in-python

        '''
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
          # else, use the default value, pos='n'
            return wordnet.NOUN

    def lemmaWithPos(self, positive_reviews_words, negative_reviews_words):
        '''
        preprocessing decision b)   

        This method is to lemmatize tokens in order to remove
        the differences that comes from suffixes and obtain the 
        stem of a word (lexical categories).

        However, lemmatizatin sometimes gets confused about the 
        word itself without specifying the POS of the word. 
        For example, for the following sentence:
        ['he', 'leaves', 'the', 'appartment', 'with', 'great', 'happiness']
        Lemmatization may give the result as:
        "leaves"  -----lemmatization----->    "leaf"
        Because lemmatization considers "leaves" as a noun.
        
        To avoid this problem to some extent, I decide to lemmatize 
        with POS tagging, and it will reduce the confusion that comes 
        from upspecified lexical categories.

        '''

        # add POS tagging
        positive_pos, negative_pos = self.getPosTag(positive_reviews_words, negative_reviews_words)
        # get lemma
        wordnet_lemmatizer = WordNetLemmatizer()
        positive_reviews_words = [[wordnet_lemmatizer.lemmatize(word[0], pos=self.getWordnetPos(word[1])) for word in s]
                      for s in positive_pos]
        negative_reviews_words = [[wordnet_lemmatizer.lemmatize(word[0], pos=self.getWordnetPos(word[1])) for word in s]
                      for s in negative_pos]
        return positive_reviews_words, negative_reviews_words
    
    
    def removeStopWordsAndShortWords(self, positive_reviews_words, negative_reviews_words):
        '''
        preprocessing decision c)

        This method is to remove the stop words and words whose length is less than 2.
        
        Stop words and short words removal should come after the lemmatization with 
        POS tagging because we might need stop words for POS tagging to preserve 
        the contextual information.

        '''
        positive_reviews_words = [[w for w in sentence if not w in self.stop_words if len(w) > 2] for sentence in positive_reviews_words]
        negative_reviews_words = [[w for w in sentence if not w in self.stop_words if len(w) > 2] for sentence in negative_reviews_words]
        return positive_reviews_words, negative_reviews_words


    def selectWordType(self, reviews_data):
        all_words = []
        for sent in reviews_data:
            for w in sent:
              if w[1][0] in self.selected_word_type:
                    all_words.append(w[0])
        return all_words

    def buildVocabulary(self, pos_words, neg_words, selected_word_type=['J', 'R', 'V'], intersection=False):
        '''
        preprocessing decision d)

        This method is to obtain a vocabulary which will be passed to vectorizer.
        
        This method is only used when vocabulary needs to be built manually.
        e.g., prescreening adjective. 

        If the max_features in countvectorizer or tfidfvectorizer is used. 
        this preprocessing is skipped
        '''
        self.selected_word_type = selected_word_type
        # build vocabulary only based on train data set
        # firstly need to split train 
        y = self.pos_y + self.neg_y
        X_train, X_test, y_train, y_test = train_test_split(
            (pos_words + neg_words), y, test_size=self.test_size, random_state=self.seed)
        
        # restore positive_reviews_words for training set
        positive_reviews_words = []
        for i, sent in enumerate(X_train):
          if y_train[i] == 1:
            positive_reviews_words.append(sent)

        # restore negative_reviews_words for training set
        negative_reviews_words = []
        for i, sent in enumerate(X_train):
          if y_train[i] == 0:
            negative_reviews_words.append(sent)
        
        # add POS tagging
        positive_pos, negative_pos = self.getPosTag(positive_reviews_words, negative_reviews_words)
        # get selected word type
        all_words_positive = self.selectWordType(positive_pos) 
        all_words_negative = self.selectWordType(negative_pos) 

        if intersection:
            review_vocabulary = list(set(all_words_positive) ^ set(all_words_negative))
        else:
            review_vocabulary = set(all_words_positive + all_words_negative)
        return review_vocabulary
        
#
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# --------------------This part stores different models/experiment in this assignment--------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# to get final model: use method getFinalModel()
#



    def getBasicModel(self):
        '''
        This method returns the basic model and its result.
        Only simple tokenization is used

        Basic model uses 0/1 matrix to represent the presence of each word
        in vocabulary using Countvectorizer.
        '''      
        # tokenize without removing punctuation
        positive_reviews_words = [word_tokenize(review) for review in self.positive_reviews]
        negative_reviews_words = [word_tokenize(review) for review in self.negative_reviews]
        y = self.pos_y + self.neg_y

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(
          (positive_reviews_words+negative_reviews_words), y, test_size=test_size, random_state=seed)
        
        def dummyProcesser(text):
            return text

        # build training pipeline
        train_pipe = make_pipeline(
        CountVectorizer(tokenizer=dummyProcesser, 
                          preprocessor=dummyProcesser,
                          binary=True),
          LogisticRegression(max_iter=200)
          )
        
        
        # set up params
        params = {
                  'countvectorizer__max_features':[500], #2000, 3000, 5000, 7000, None
                  }

        # set up gs
        gs = GridSearchCV(train_pipe, params, cv=5, n_jobs=-1)

        gs.fit(X_train, y_train)
        print(f'in sample accuracy is: {gs.score(X_train, y_train)}')
        print(f'out of sample accuracy is: {gs.score(X_test, y_test)}')
        print(f'The best hyperparameters: {gs.best_params_}')
        #print(f'Cross-validation result: {gs.cv_results_}')
        return gs
        # in sample accuracy is: 0.7371030416722497
        # out of sample accuracy is: 0.6914660831509847

    def getFinalModel(self):
        '''
        This method returns the FINAL MODEL described in the report

        Use preprocessing dession a) b) e) f) g)
        
        use all words as vocabulary
        use TfidfVectorizer
        use bigrams

        Fifth experiment includes tokenization, lemmatization with POS, 
        use all tokens, the input is TF-IDF freuquencies.
        '''
        positive_reviews_words, negative_reviews_words = self.tokenWithPunctuationRemoval()

        positive_reviews_words, negative_reviews_words = self.lemmaWithPos(
                                                  positive_reviews_words, 
                                                  negative_reviews_words)
      
        y = self.pos_y + self.neg_y        
        # train test split
        X_train, X_test, y_train, y_test = train_test_split(
                    (positive_reviews_words+negative_reviews_words), y, 
                    test_size=test_size, random_state=seed)

        def dummyProcesser(text):
            return text
        
        train_pipe = make_pipeline(
          TfidfVectorizer(tokenizer=dummyProcesser, 
                          preprocessor=dummyProcesser,
                          binary=False,
                          encoding='latin-1'),
          LogisticRegression(max_iter=200)
          )
        
        params = {
          'tfidfvectorizer__ngram_range': [(1, 2)],   #[(1, 1), (1, 2), (1, 3)], 
          'tfidfvectorizer__use_idf': [True],   #[True, False],
          'tfidfvectorizer__sublinear_tf': [True],  #[True, False],
          'tfidfvectorizer__norm': ["l2"],
          'tfidfvectorizer__max_df': [1.0], # [1.0, 0.7 0.8, 0.9],
          'tfidfvectorizer__min_df': [1], #[1, 0.01, 0.05 0.1, 0.15],
          'tfidfvectorizer__max_features': [1000, 3000, 5000, 7000, 10000],
          'logisticregression__C': [1],
          'logisticregression__penalty': ['l2'],
        }

        # set up gs
        gs = GridSearchCV(train_pipe, params, cv=5, n_jobs=-1)

        gs.fit(X_train, y_train)
        print(f'in sample accuracy is: {gs.score(X_train, y_train)}')
        print(f'out of sample accuracy is: {gs.score(X_test, y_test)}')
        print(f'The best hyperparameters: {gs.best_params_}')
        # print(f'Cross-validation result: {gs.cv_results_}')
        # in sample accuracy is: 0.8980302827281255
        # out of sample accuracy is: 0.7674273210378243
        return gs



    def getFirstExperiment(self):
        '''
        This method returns the model and its result of the first experiment.
        Use preprocessing dession a) b) c) e) f) g)
        
        First experiment includes tokenization, lemmatization with POS, Remove
        stop words and short words, the input is the number of occurences of 
        words using Countvectorizer.
        '''
        positive_reviews_words, negative_reviews_words = self.tokenWithPunctuationRemoval()

        positive_reviews_words, negative_reviews_words = self.lemmaWithPos(
                                                  positive_reviews_words, 
                                                  negative_reviews_words)
        
        positive_reviews_words, negative_reviews_words = self.removeStopWordsAndShortWords(
                                                  positive_reviews_words, 
                                                  negative_reviews_words)
        
        y = self.pos_y + self.neg_y
        # train test split
        X_train, X_test, y_train, y_test = train_test_split(
                    (positive_reviews_words+negative_reviews_words), y, 
                    test_size=test_size, random_state=seed)

        def dummyProcesser(text):
            return text
        
        train_pipe = make_pipeline(
        CountVectorizer(tokenizer=dummyProcesser, 
                          preprocessor=dummyProcesser,
                          binary=False),
          LogisticRegression(max_iter=200)
          )

        params = {
            'countvectorizer__max_df': [1.0, 0.9],
            'countvectorizer__min_df': [1, 0.1],
            'countvectorizer__max_features': [3000], #500, 1000, 3000, 5000, 7000
            'logisticregression__C': [0.8, 1],
            'logisticregression__penalty': ['l1', 'l2'],
        }

        # set up gs
        gs = GridSearchCV(train_pipe, params, cv=5, n_jobs=-1)

        gs.fit(X_train, y_train)
        print(f'in sample accuracy is: {gs.score(X_train, y_train)}')
        print(f'out of sample accuracy is: {gs.score(X_test, y_test)}')
        print(f'The best hyperparameters: {gs.best_params_}')
        # print(f'Cross-validation result: {gs.cv_results_}')
        # in sample accuracy is: 0.9402385099825807
        # out of sample accuracy is: 0.7421069084088777
        return gs



    def getSecondExperiment(self):
        '''
        This method returns the model and its result of the second experiment.
        Use preprocessing dession a) b) c) d) e) f) g)
        
        vocabulary includes only adjective, adverb, verb

        second experiment includes tokenization, lemmatization with POS, Remove
        stop words and short words,use only adjective, adverb, verb, the input is the 
        number of occurences of words using Countvectorizer.
        '''
        positive_reviews_words, negative_reviews_words = self.tokenWithPunctuationRemoval()

        positive_reviews_words, negative_reviews_words = self.lemmaWithPos(
                                                  positive_reviews_words, 
                                                  negative_reviews_words)
        
        positive_reviews_words, negative_reviews_words = self.removeStopWordsAndShortWords(
                                                  positive_reviews_words, 
                                                  negative_reviews_words)
        
        review_vocabulary = self.buildVocabulary(positive_reviews_words, 
                                                 negative_reviews_words)

        y = self.pos_y + self.neg_y        
        # train test split
        X_train, X_test, y_train, y_test = train_test_split(
                    (positive_reviews_words+negative_reviews_words), y, 
                    test_size=test_size, random_state=seed)

        def dummyProcesser(text):
            return text
        
        train_pipe = make_pipeline(
        CountVectorizer(tokenizer=dummyProcesser, 
                          preprocessor=dummyProcesser,
                          vocabulary=review_vocabulary,
                          binary=False),
          LogisticRegression(max_iter=200)
          )
        
        params = {
            'countvectorizer__max_df': [1.0, 0.9],
            'countvectorizer__min_df': [1, 0.1],
            'logisticregression__C': [0.8, 1],
            'logisticregression__penalty': ['l1', 'l2'],
        }

        # set up gs
        gs = GridSearchCV(train_pipe, params, cv=5, n_jobs=-1)

        gs.fit(X_train, y_train)
        print(f'in sample accuracy is: {gs.score(X_train, y_train)}')
        print(f'out of sample accuracy is: {gs.score(X_test, y_test)}')
        print(f'The best hyperparameters: {gs.best_params_}')
        # print(f'Cross-validation result: {gs.cv_results_}')
        # in sample accuracy is: 0.8834248961543615
        # out of sample accuracy is: 0.7311659893716786
        return gs



    def getThirdExperiment(self):
        '''
        This method returns the model and its result of the third experiment.
        Use preprocessing dession a) b) c) d) e) f) g)
        
        vocabulary includes only adjective, adverb, verb
        use Countvectorizer TfidfTransformer

        Third experiment includes tokenization, lemmatization with POS, Remove
        stop words and short words,use only adjective, adverb, verb, the input is the 
        frequencies of words using Countvectorizer and TfidfTransformer.
        '''
        positive_reviews_words, negative_reviews_words = self.tokenWithPunctuationRemoval()

        positive_reviews_words, negative_reviews_words = self.lemmaWithPos(
                                                  positive_reviews_words, 
                                                  negative_reviews_words)
        
        positive_reviews_words, negative_reviews_words = self.removeStopWordsAndShortWords(
                                                  positive_reviews_words, 
                                                  negative_reviews_words)
        
        review_vocabulary = self.buildVocabulary(positive_reviews_words, 
                                                 negative_reviews_words)

        y = self.pos_y + self.neg_y        
        # train test split
        X_train, X_test, y_train, y_test = train_test_split(
                    (positive_reviews_words+negative_reviews_words), y, 
                    test_size=test_size, random_state=seed)

        def dummyProcesser(text):
            return text
        
        train_pipe = make_pipeline(
          CountVectorizer(tokenizer=dummyProcesser, 
                          preprocessor=dummyProcesser,
                          vocabulary=review_vocabulary,
                          binary=False),
          TfidfTransformer(),
          LogisticRegression(max_iter=200)
          )
        
        params = {
          'countvectorizer__ngram_range': [(1, 1), (1, 2)], 
          'tfidftransformer__use_idf': [True],
          'tfidftransformer__sublinear_tf': [True, False], 
          'tfidftransformer__norm': ["l2"],
          'logisticregression__C': [0.8, 1],
          'logisticregression__penalty': ['l2'],
        }

        # set up gs
        gs = GridSearchCV(train_pipe, params, cv=5, n_jobs=-1)

        gs.fit(X_train, y_train)
        print(f'in sample accuracy is: {gs.score(X_train, y_train)}')
        print(f'out of sample accuracy is: {gs.score(X_test, y_test)}')
        print(f'The best hyperparameters: {gs.best_params_}')
        # print(f'Cross-validation result: {gs.cv_results_}')
        # in sample accuracy is: 0.8700254589307249
        # out of sample accuracy is: 0.745232885276649
        return gs



    def getFourthExperiment(self):
        '''
        This method returns the model and its result of the Fourth experiment.
        Use preprocessing dession a) b) c) d) e) f) g)
        
        vocabulary includes only adjective, adverb, verb
        use TfidfVectorizer
        use bigrams

        Fourth experiment includes tokenization, lemmatization with POS, Remove
        stop words and short words,use only adjective, adverb, verb, the input is 
        TF-IDF freuquencies.
        '''
        positive_reviews_words, negative_reviews_words = self.tokenWithPunctuationRemoval()

        positive_reviews_words, negative_reviews_words = self.lemmaWithPos(
                                                  positive_reviews_words, 
                                                  negative_reviews_words)
        
        positive_reviews_words, negative_reviews_words = self.removeStopWordsAndShortWords(
                                                  positive_reviews_words, 
                                                  negative_reviews_words)
        
        review_vocabulary = self.buildVocabulary(positive_reviews_words, 
                                                 negative_reviews_words)

        y = self.pos_y + self.neg_y        
        # train test split
        X_train, X_test, y_train, y_test = train_test_split(
                    (positive_reviews_words+negative_reviews_words), y, 
                    test_size=test_size, random_state=seed)

        def dummyProcesser(text):
            return text
        
        train_pipe = make_pipeline(
          TfidfVectorizer(tokenizer=dummyProcesser, 
                          preprocessor=dummyProcesser,
                          vocabulary=review_vocabulary,
                          binary=False,
                          encoding='latin-1'),
          LogisticRegression(max_iter=200)
          )
        
        params = {
          'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)], 
          'tfidfvectorizer__use_idf': [True, False],
          'tfidfvectorizer__sublinear_tf': [True, False], 
          'tfidfvectorizer__norm': ["l2"],
          'tfidfvectorizer__max_df': [1.0], # [1.0, 0.7 0.8, 0.9],
          'tfidfvectorizer__min_df': [1], #[1, 0.01, 0.05 0.1, 0.15],
          'logisticregression__C': [0.8, 1],
          'logisticregression__penalty': ['l2'],
        }

        # set up gs
        gs = GridSearchCV(train_pipe, params, cv=5, n_jobs=-1)

        gs.fit(X_train, y_train)
        print(f'in sample accuracy is: {gs.score(X_train, y_train)}')
        print(f'out of sample accuracy is: {gs.score(X_test, y_test)}')
        print(f'The best hyperparameters: {gs.best_params_}')
        # print(f'Cross-validation result: {gs.cv_results_}')
        # in sample accuracy is: 0.8700254589307249
        # out of sample accuracy is: 0.745232885276649
        return gs



    def getFifthExperiment(self):
        '''
        This method returns the model and its result of the fifth experiment.
        Use preprocessing dession a) b) c) e) f) g)
        
        use all words as vocabulary
        use TfidfVectorizer
        use bigrams

        Fifth experiment includes tokenization, lemmatization with POS, Remove
        stop words and short words,use all tokens, the input is 
        TF-IDF freuquencies.
        '''
        positive_reviews_words, negative_reviews_words = self.tokenWithPunctuationRemoval()

        positive_reviews_words, negative_reviews_words = self.lemmaWithPos(
                                                  positive_reviews_words, 
                                                  negative_reviews_words)
        
        positive_reviews_words, negative_reviews_words = self.removeStopWordsAndShortWords(
                                                  positive_reviews_words, 
                                                  negative_reviews_words)

        y = self.pos_y + self.neg_y        
        # train test split
        X_train, X_test, y_train, y_test = train_test_split(
                    (positive_reviews_words+negative_reviews_words), y, 
                    test_size=test_size, random_state=seed)

        def dummyProcesser(text):
            return text
        
        train_pipe = make_pipeline(
          TfidfVectorizer(tokenizer=dummyProcesser, 
                          preprocessor=dummyProcesser,
                          binary=False,
                          encoding='latin-1'),
          LogisticRegression(max_iter=200)
          )
        
        params = {
          'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)], 
          'tfidfvectorizer__use_idf': [True, False],
          'tfidfvectorizer__sublinear_tf': [True, False], 
          'tfidfvectorizer__norm': ["l2"],
          'tfidfvectorizer__max_df': [1.0], # [1.0, 0.7 0.8, 0.9],
          'tfidfvectorizer__min_df': [1], #[1, 0.01, 0.05 0.1, 0.15],
          'tfidfvectorizer__max_features': [1000, 3000, 5000, 7000, 10000],
          'logisticregression__C': [0.8, 1],
          'logisticregression__penalty': ['l2'],
        }

        # set up gs
        gs = GridSearchCV(train_pipe, params, cv=5, n_jobs=-1)

        gs.fit(X_train, y_train)
        print(f'in sample accuracy is: {gs.score(X_train, y_train)}')
        print(f'out of sample accuracy is: {gs.score(X_test, y_test)}')
        print(f'The best hyperparameters: {gs.best_params_}')
        # print(f'Cross-validation result: {gs.cv_results_}')
        # in sample accuracy is: 0.8832909017821251
        # out of sample accuracy is: 0.7583619881212879
        return gs