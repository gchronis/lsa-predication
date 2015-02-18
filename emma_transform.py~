import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora, models, similarities


#LOAD IN OLD DICT AND CORPUS
dictionary = corpora.Dictionary.load('emma/emma.dict')
corpus = corpora.MmCorpus('emma/emma_corpus.mm')
print(corpus)

#TRAIN TFIDF MODEL
tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
tfidf.save('emma/model/emma.tfidf') # save TFIDF model trained on emma

#CREATE TFIDF TRANSFORM
corpus_tfidf = tfidf[corpus] # create a wrapper over the original corpus: bow->tfidf
corpora.MmCorpus.serialize('emma/emma_tfidf.mm', corpus_tfidf)

# TRAIN LSI MODEL
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=10) # initialize an LSI transformation
lsi.save('emma/model/emma.lsi') # save LSI model trained on emma 

# CREATE LSI INSTANCE
corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
corpora.MmCorpus.serialize('emma/emma_lsi.mm', corpus_lsi)
lsi.print_topics(10)

# CREATE LSI SIMILARITY INDEX
index = similarities.Similarity('emma/index', corpus_lsi, num_features = 10)
index.save('emma/emma_lsi.index')
