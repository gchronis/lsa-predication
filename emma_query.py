

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


from gensim import corpora, models, similarities
dictionary = corpora.Dictionary.load('emma/emma.dict')
#corpus = corpora.MmCorpus('emma/emma_corpus.mm')
lsi = models.LsiModel.load('emma/model/emma.lsi')
corpus_lsi = corpora.MmCorpus('emma/emma_corpus.mm')
index = 



lsi.save('emma/model/emma.lsi') # save LSI model trained on emma                                                                                                                     
corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
lsi.print_topics(10)
index = similarities.Similarity(lsi[corpus])
index.save('emma/emma_lsi.index')

# QUERY EMMA FOR MOST SIMILAR SENTENCES
sims = index[vec_lsi] # perform a similarity query against the corpus>
sims = sorted(enumerate(sims), key=lambda item: -item[1])
print("The ten most similar sentences to " + doc + " in Emma by Jane Austen are: ")
print(sims[1:10]) # print sorted (document number, similarity score) 2-tuples 
