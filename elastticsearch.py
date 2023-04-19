from elasticsearch import Elasticsearch
from transformers import AutoTokenizer, AutoModel

# Load the sentence dataset and language model
sentences = load_sentences('sentences.txt')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Generate a sentence embedding for each sentence using the language model
sentence_embeddings = []
for sentence in sentences:
    inputs = tokenizer(sentence, return_tensors='pt')
    outputs = model(**inputs)
    embedding = outputs.pooler_output.detach().numpy()[0]
    sentence_embeddings.append(embedding)

# Connect to Elasticsearch and create an index
es = Elasticsearch()
es.indices.create(index='sentences', body={
    'mappings': {
        'properties': {
            'embedding': {
                'type': 'dense_vector',
                'dims': sentence_embeddings[0].shape[0]
            }
        }
    }
})

# Add the sentence embeddings to Elasticsearch
for i, embedding in enumerate(sentence_embeddings):
    es.index(index='sentences', id=i, body={
        'sentence': sentences[i],
        'embedding': embedding.tolist()
    })

# Generate a query embedding and retrieve the most similar sentences
query_sentence = 'I love this movie'
inputs = tokenizer(query_sentence, return_tensors='pt')
outputs = model(**inputs)
query_embedding = outputs.pooler_output.detach().numpy()[0]
res = es.search(index='sentences', body={
    'query': {
        'script_score': {
            'query': {'match_all': {}},
            'script': {
                'source': 'cosineSimilarity(params.query_embedding, doc["embedding"]) + 1.0',
                'params': {'query_embedding': query_embedding.tolist()}
            }
        }
    }
})

# Display the most similar sentences to the query sentence
similar_sentences = [hit['_source']['sentence'] for hit in res['hits']['hits'][:10]]
print(similar_sentences)
