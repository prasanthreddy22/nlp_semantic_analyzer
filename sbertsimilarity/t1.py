"""
This is a simple application for sentence embeddings: semantic search

We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.

This script outputs for various queries the top 5 most similar sentences in the corpus.
"""
from sentence_transformers import SentenceTransformer
import scipy.spatial
import sys

def nlpeval(cid,fid,model):
  embedder = SentenceTransformer(model)

  #  Corpus with example sentences
  list=[];
  
  corpus=[line.rstrip('\n') for line in open(cid)]
  corpus_embeddings = embedder.encode(corpus)

  # Query sentences:
  
  
  queries=[line.rstrip('\n') for line in  open(fid)]

  query_embeddings = embedder.encode(queries)

  # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
  closest_n = 1
  for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    

    for idx, distance in results[0:closest_n]:
        list.append( "(Score: %.4f)" % (1-distance))

  return(list)



