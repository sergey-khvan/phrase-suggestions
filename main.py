from sentence_transformers import SentenceTransformer
import numpy as np
import re 
import csv

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

with open('text.txt', 'r') as file:
    text = file.read()
with open('terms.csv', 'r') as file:
    # Create a CSV reader
    csv_reader = csv.reader(file)

    # Extract phrases into a list
    phrase_list = []
    for row in csv_reader:
        # Assuming the phrase is in the first column (index 0)
        phrase = row[0]
        phrase_list.append(phrase)

delimiter = '.,'  # This string contains both dots and commas as delimiters

# Split the text using the custom delimiter
result = re.split('[' + delimiter + ']', text)

# Remove any empty strings from the result
sentences = [item.strip().lower() for item in result if item.strip()]

sentence_emb = model.encode(sentences, convert_to_tensor=True)
phrase_emb = model.encode(phrase_list, convert_to_tensor=True)

similarities= []
for i, sentence in enumerate(sentence_emb):
    max_similarity = -1  
    most_similar_phrase = None
    for j, phrase in enumerate(phrase_emb):
        similarity = cosine_similarity(sentence, phrase)
        if similarity > 0.35 and similarity > max_similarity:
            max_similarity = similarity
            most_similar_phrase = phrase_list[j]
    if most_similar_phrase is not None:
        similarities.append((sentences[i], most_similar_phrase, max_similarity))

print(similarities)