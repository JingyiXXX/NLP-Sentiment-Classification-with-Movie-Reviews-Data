# NLP-Sentiment-Classification-with-Movie-Reviews-Data
This repository tests different sentence-level preprocessing methods using v1.0 sentence polarity movie review dataset in Bo Pang and Lillian Lee 2005

Thanks for reading.

# Introduction

This README is for explaining the Class SentimentClassification
in sentimentClassification.py.


# How to use Class sentimentClassification 

I used Google Colab to load the raw data. So firstly,
please change the directory path at the beginning of the code:

And PLEASE make sure functions **load_data** and **label_data** run
properly, as they will generate necessary input for the Class.

if you are reading from your local directory, use:
pos_file_path =  "./rt-polaritydata/rt-polarity.pos"
neg_file_path =  "./rt-polaritydata/rt-polarity.neg"

run the following command in terminal: 
*python a1.py*
This shall give the result of final model.


## 1. What does Class SentimentClassification include?
Class SentimentClassification includes ALL of the preprocessing decisions,
which are defined as different methods
This class also includes ALL trials of experiment (final model, baseline model, other
attempts of experiments)


## 2. How to obtain Final model instance and its result?

instantiate the class
sclf = SentimentClassification()

Call function - getFinalModel
gs =sclf.getFinalModel()

I did not include all the hyperparameters being tried
in this method, in order to reduce the running time. Otherwise,
it could take hours.


## 3. How to obtain basic model (baseline) instance and its result?

sclf = SentimentClassification()
gs = sclf.getBasicModel()


## 4. Other experimental results:

1) first experiment:
	
	I use preprocessing dession a) b) c) e) f) g) described in the report
        It includes tokenization, lemmatization with POS, Remove
        stop words and short words, the input is the number of occurences of 
        words using Countvectorizer.
        # in sample accuracy is: 0.9402385099825807
        # out of sample accuracy is: 0.7421069084088777

2) second experiment:

        I use preprocessing dession a) b) c) d) e) f) g) described in the report
        
        vocabulary includes only adjective, adverb, verb

        second experiment includes tokenization, lemmatization with POS, Remove
        stop words and short words,use only adjective, adverb, verb, the input is the 
        number of occurences of words using Countvectorizer.

	selecting a subset of vocabulary does improve the result than the basic models,
	but it does not help too much.
	when ranking the most frequent words in both review classes, I notice they are 
	very similar (good, bad…). I tried to remove the intersection of these most frequent
	words, but the accuracy decreases. I speculate that it is because even if they have
	common frequent words, but the number of occurences of these words still deliver 
 	information to allow models to distingush the class

3) third experiment:
	
	This returns the model of third experiment.
        I use preprocessing dession a) b) c) d) e) f) g)
        
        vocabulary includes only adjective, adverb, verb
        I also use Countvectorizer TfidfTransformer

        Third experiment includes tokenization, lemmatization with POS, Remove
        stop words and short words,use only adjective, adverb, verb, the input is the 
        frequencies of words using Countvectorizer and TfidfTransformer.
        # in sample accuracy is: 0.8700254589307249
        # out of sample accuracy is: 0.745232885276649


4) fourth experiment:

        This returns the model of Fourth experiment.
        I use preprocessing dession a) b) c) d) e) f) g)
        
        vocabulary includes only adjective, adverb, verb
        use TfidfVectorizer
        use bigrams

        Fourth experiment includes tokenization, lemmatization with POS, Remove
        stop words and short words,use only adjective, adverb, verb, the input is 
        TF-IDF freuquencies.
        # in sample accuracy is: 0.8700254589307249
        # out of sample accuracy is: 0.745232885276649


5) fifth experiment:
        
	I contains the model the fifth experiment.
        I use preprocessing dession a) b) c) e) f) g)
        
        I also use all words as vocabulary
        use TfidfVectorizer
        use bigrams

        Fifth experiment includes tokenization, lemmatization with POS, Remove
        stop words and short words,use all tokens, the input is 
        TF-IDF freuquencies. I will let the model itself to select the best 
	input features, instead of only using a subset of vocabulary.


# Citation

There are some codes I refer to the following links:

*refer the idea of using only adjective, adverb, and verb vocabulary*
https://towardsdatascience.com/basic-binary-sentiment-analysis-using-nltk-c94ba17ae386

*Use sklearn TfidfVectorizer with already tokenized inputs*
https://stackoverflow.com/questions/48671270/use-sklearn-tfidfvectorizer-with-already-tokenized-inputs

*lemmatization with pos tagging*
https://towardsdatascience.com/building-a-text-normalizer-using-nltk-ft-pos-tagger-e713e611db8

*pos tagging for tokenized sentences*
https://towardsdatascience.com/testing-the-waters-with-nltk-3600574f891c

*convert treebank tag to wordnet tag:*
https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python



Thanks for reading.
