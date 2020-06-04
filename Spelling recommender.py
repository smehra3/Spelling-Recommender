
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 2 - Introduction to NLTK
# 
# In part 1 of this assignment you will use nltk to explore the Herman Melville novel Moby Dick. Then in part 2 you will create a spelling recommender function that uses nltk to find words similar to the misspelling. 

# ## Part 1 - Analyzing Moby Dick

# In[1]:


import nltk
import pandas as pd
import numpy as np

nltk.download('punkt')
nltk.download('all-nltk')

# If you would like to work with the raw text you can use 'moby_raw'
with open('moby.txt', 'r') as f:
    moby_raw = f.read()
    
# If you would like to work with the novel in nltk.Text format you can use 'text1'
moby_tokens = nltk.word_tokenize(moby_raw)
text1 = nltk.Text(moby_tokens)


# ### Example 1
# 
# How many tokens (words and punctuation symbols) are in text1?
# 
# *This function should return an integer.*

# In[ ]:


def example_one():
    
    return len(nltk.word_tokenize(moby_raw)) # or alternatively len(text1)

example_one()


# ### Example 2
# 
# How many unique tokens (unique words and punctuation) does text1 have?
# 
# *This function should return an integer.*

# In[ ]:


def example_two():
    
    return len(set(nltk.word_tokenize(moby_raw))) # or alternatively len(set(text1))

example_two()


# ### Example 3
# 
# After lemmatizing the verbs, how many unique tokens does text1 have?
# 
# *This function should return an integer.*

# In[ ]:


from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
def example_three():

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w,'v') for w in text1]

    return len(set(lemmatized))

example_three()


# ### Question 1
# 
# What is the lexical diversity of the given text input? (i.e. ratio of unique tokens to the total number of tokens)
# 
# *This function should return a float.*

# In[9]:


def answer_one():
    return len(set(nltk.word_tokenize(moby_raw)))/len(nltk.word_tokenize(moby_raw))
answer_one()


# ### Question 2
# 
# What percentage of tokens is 'whale'or 'Whale'?
# 
# *This function should return a float.*

# In[3]:


def answer_two():
    wWhales_count = 0
    for w in nltk.word_tokenize(moby_raw):
        if (w == 'whale' or w == 'Whale'):
            wWhales_count += 1
    return wWhales_count*100/len(nltk.word_tokenize(moby_raw))
answer_two()


# ### Question 3
# 
# What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?
# 
# *This function should return a list of 20 tuples where each tuple is of the form `(token, frequency)`. The list should be sorted in descending order of frequency.*

# In[8]:


from nltk.probability import FreqDist
def answer_three():
    
    wordfreq = FreqDist(text1)
    wordfreq_desc = sorted(wordfreq.items(), key=lambda x: x[1],reverse=True)
    
    return wordfreq_desc[:20]# Your answer here
answer_three()


# ### Question 4
# 
# What tokens have a length of greater than 5 and frequency of more than 150?
# 
# *This function should return an alphabetically sorted list of the tokens that match the above constraints. To sort your list, use `sorted()`*

# In[10]:


def answer_four():
    wordfreq = FreqDist(text1)
    freqAnswerFour = [w for w in wordfreq.keys() if len(w)>5 and wordfreq[w]>150]
    
    return sorted(freqAnswerFour)# Your answer here

answer_four()


# ### Question 5
# 
# Find the longest word in text1 and that word's length.
# 
# *This function should return a tuple `(longest_word, length)`.*

# In[22]:


def answer_five():
    global longest_word
    length=0
    for w in moby_tokens:
        if len(w)>length:
            longest_word=w
            length=len(w)
    
    return (longest_word, length)# Your answer here

answer_five()


# ### Question 6
# 
# What unique words have a frequency of more than 2000? What is their frequency?
# 
# "Hint:  you may want to use `isalpha()` to check if the token is a word and not punctuation."
# 
# *This function should return a list of tuples of the form `(frequency, word)` sorted in descending order of frequency.*

# In[15]:


def answer_six():
    
    wordfreq = FreqDist(text1)
    freqAnswerFour = sorted([(f,w) for w, f in wordfreq.items() if w.isalpha() and f>2000], key=lambda x: x[0], reverse=True)
    
    return freqAnswerFour# Your answer here

answer_six()


# ### Question 7
# 
# What is the average number of tokens per sentence?
# 
# *This function should return a float.*

# In[33]:


def answer_seven():
    sentences = nltk.sent_tokenize(moby_raw)
    return len(moby_tokens)/len(sentences)# Your answer here
answer_seven()


# ### Question 8
# 
# What are the 5 most frequent parts of speech in this text? What is their frequency?
# 
# *This function should return a list of tuples of the form `(part_of_speech, frequency)` sorted in descending order of frequency.*

# In[4]:


def answer_eight():
    posfreq = {}
    pos = [tag for word, tag in nltk.pos_tag(text1)]
    for p in pos:
        if p not in posfreq:
            posfreq[p] = 0 
        posfreq[p]=posfreq[p]+1
    posfreqsorted = sorted(posfreq.items(), key=lambda x: x[1], reverse=True)
    return posfreqsorted[:5]# Your answer here

answer_eight()


# ## Part 2 - Spelling Recommender
# 
# For this part of the assignment you will create three different spelling recommenders, that each take a list of misspelled words and recommends a correctly spelled word for every word in the list.
# 
# For every misspelled word, the recommender should find find the word in `correct_spellings` that has the shortest distance*, and starts with the same letter as the misspelled word, and return that word as a recommendation.
# 
# *Each of the three different recommenders will use a different distance measure (outlined below).
# 
# Each of the recommenders should provide recommendations for the three default words provided: `['cormulent', 'incendenece', 'validrate']`.

# In[6]:


from nltk.corpus import words

correct_spellings = words.words()


# ### Question 9
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the trigrams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[7]:


def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):
    recommendations=[]
    for e in entries:
        distances = [(nltk.jaccard_distance(set(nltk.ngrams(e, n=3)), 
                                        set(nltk.ngrams(a, n=3))), a) for a in correct_spellings if a[0]==e[0] and len(a)>2]
        recommendations.append(sorted(distances)[0][1])
    return recommendations# Your answer here
    
answer_nine()


# ### Question 10
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the 4-grams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[ ]:


def answer_ten(entries=['cormulent', 'incendenece', 'validrate']):
    recommendations=[]
    for e in entries:
        distances = [(nltk.jaccard_distance(set(nltk.ngrams(e, n=4)), 
                                        set(nltk.ngrams(a, n=4))), a) for a in correct_spellings if a[0]==e[0] and len(a)>2]
        recommendations.append(sorted(distances)[0][1])
    return recommendations# Your answer here
    
answer_ten()


# ### Question 11
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Edit distance on the two words with transpositions.](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance)**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[9]:


def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):
    recommendations=[]
    for e in entries:
        distances = [(nltk.edit_distance(e, a), a) for a in correct_spellings if a[0]==e[0] and len(a)>2]
        recommendations.append(sorted(distances)[0][1])
    return recommendations# Your answer here
    
answer_eleven()


# In[ ]:




