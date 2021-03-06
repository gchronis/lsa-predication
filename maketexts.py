import nltk, re, pprint
import gensim
import codecs
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords, gutenberg, reader

#sent_tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')

#fileids = []
#for id in nltk.corpus.gutenberg.fileids():
#   fileids.append(id)

#texts = codecs.open('gutenberg_lines.txt','w',encoding='utf-8')
#sentences = []
#for f in fileids:
#   for sentence in nltk.corpus.gutenberg.sents(f):
#       sentences.append(sentence)

#for sentence in sentences:
#   texts.write(sentence)


root = '/usr/local/share/nltk_data/corpora/gutenberg'
reader = PlaintextCorpusReader(root, '.*\.txt',sent_tokenizer) # doctest: +SKIP


documents = []

for f in reader.fileids():
   for sent in reader.sents(f):
      documents.append(sent)

stoplist = stopwords.words('english')
punct = 
texts = [[word.lower() for word in document if word not in stoplist and word not in punct]
         for document in documents]
   
all_tokens = sum(texts, [])
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
texts = [[word for word in text if word not in tokens_once] 
         for text in texts]

print(texts[1:100])
