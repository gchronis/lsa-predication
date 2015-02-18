import nltk, re, pprint, string
from gensim import corpora, models, similarities
import codecs
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, gutenberg, PlaintextCorpusReader


###
### CREATE PLAINTEXT READER
###
gutenberg_dir = nltk.data.find('corpora/gutenberg.zip').join('gutenberg/')
root = '/usr/local/share/nltk_data/corpora/gutenberg/'
reader = PlaintextCorpusReader(gutenberg_dir, '.*\emma.txt') # doctest: +SKIP  #actual regexp should read '.*\.txt'



###
### MAKE DICTIONARY
###

# collect statistics about all tokens


dictionary = corpora.Dictionary([[word.lower() for word in document]for f in reader.fileids() for document in reader.sents(f)])
#print("num tokens: " + str(sum(dictionary.itervalues())))
print("preprocessing dictionary size(num types): " + str(len(dictionary.token2id)))

# remove stop words and words that appear only once
stoplist = stopwords.words('english')
print("STOPLIST:",stoplist)

#punctuation = string.punctuation+"''"+r"\W.*"
#punctuation = list(string.punctuation)+["''"]+[r"\W.*"]
punct_re= re.compile("\W.*", re.IGNORECASE) 

stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
            if stopword in dictionary.token2id]
punct_ids = [dictionary.token2id[punctmark] for punctmark in dictionary.token2id
             if punct_re.search(punctmark)]
print("PUNCT_IDS: ", punct_ids)

once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]

# remove stopwords and words that appear only once
dictionary.filter_tokens(stop_ids + once_ids + punct_ids)

# remove gaps in id sequence after words that were removed
dictionary.compactify()

print("Postprocessing dictionary size: " + str(len(dictionary.token2id)))
dictionary.save('emma/emma.dict')
#print(dictionary.token2id)



###
### MAKE CORPUS
###

class MyCorpus(object):
    def __iter__(self):
        for f in reader.fileids():
           for sent in reader.sents(f):
               document = [word.lower() for word in sent if word in dictionary.token2id]
               doc_bow = dictionary.doc2bow(document)
               if doc_bow:
                   yield doc_bow

corpus = MyCorpus() # doesn't load the corpus into memory

print("Post-postprocessing dictionary size: " + str(len(dictionary.token2id)))
corpora.MmCorpus.serialize('emma/emma_corpus.mm', corpus)
