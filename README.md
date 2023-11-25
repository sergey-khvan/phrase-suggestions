# phrase-suggestions
Text suggestions from the list of phases

Uses hugging face models for calculation of the cosine similarity of the sentences/phrases.

In particular sentence-transformers/all-MiniLM-L6-v2

Other models which were tested:
* Bert, the results were not appropriate
* Spicy, compared to Bert results were better, but still not good enough

Used approaches:
1. Text preprocessing:
   1. lower() function was used to make all words lowercase.
   2. text was split into subsentences (During testing, it has shown to be the most accurate way)
   3. subsentences were put to a list
2. Similarity was calculated between each phrase and each subsentence.
3. Only one suggestion with max similarity was kept for each sentence.


## How to use:
* Replace the text in text.txt 
* Replace the phrases in .csv file
* Run either .py file or .ipynb
