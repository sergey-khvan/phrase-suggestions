# phrase-suggestions
Text Improvement Engine for a task

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
2. Phrases were lowered and put to a list


## How to use:
* Replace the text in text.txt 
* Replace the phrases in .csv file
* Run either .py file or .ipynb

**Note: If you want to use google colab, you will need to mount your drive and upload the txt and csv files there.**
