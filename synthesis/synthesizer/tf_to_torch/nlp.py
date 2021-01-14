from transformers import *
from synthesis import *

err_msg = "sparse_coo_tensor(): argument 'size' must be tuple of ints\", ' not NoneTyp"

sentences = err_msg
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
sentence_embeddings = tokenizer.encode(err_msg)
word_to_embedding = zip(sentences.split(' '), sentence_embeddings)
for line in word_to_embedding:
    print (line)

for sentence, embedding in zip(sentences, sentence_embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")