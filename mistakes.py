import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora, models, similarities

# LOAD EXSTING EMMA CORPUS AND DICTIONARY
dictionary = corpora.Dictionary.load('emma/emma.dict')
corpus = corpora.MmCorpus('emma/emma_corpus.mm')
print(corpus)


# CREATE TFIDF TRANFORM FOR EMMA
tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
tfidf.save('emma/model/emma.tfidf') # save TFIDF model trained on emma
corpus_tfidf = tfidf[corpus] # create a wrapper over the original corpus: bow->tfidf
corpora.MmCorpus.serialize('emma/emma_tfidf.mm', corpus)


# CREATE LSA SPACE FOR EMMA
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=10) # initialize an LSI transformation
lsi.save('emma/model/emma.lsi') # save LSI model trained on emma 
corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
lsi.print_topics(10)


# CREATE INDEX FOR EMMA
index = similarities.Similarity('emma/index', corpus_lsi, num_features = 10)
index.save('emma/emma_lsi.index')


# CREATE QUERY VEC FOR EMMA
doc = "very well"
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow] # convert the query to LSI space
print(vec_lsi)


# QUERY EMMA FOR MOST SIMILAR SENTENCES
sims = index[vec_lsi] # perform a similarity query against the corpus>

sims = sorted(enumerate(sims), key=lambda item: -item[1])
print("The ten most similar sentences to " + doc + " in Emma by Jane Austen are: ")
print(sims[1:10]) # print sorted (document number, similarity score) 2-tuples


