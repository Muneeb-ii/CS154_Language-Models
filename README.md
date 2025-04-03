# Project 4A

### Name

Muneeb Azfar Nafees

### Introspection

1. **_Stemming and Lemmatization_**: 
Performing stemming and lemmatization was one of the project's more challenging parts. Initially, I attempted to use stemming, but it produced unexpected results by stripping away essential parts of words, leading to distorted meanings. For instance, it removed suffixes and prefixes in an overly aggressive way, impacting the overall context of the words. This prompted me to switch to lemmatization, which, after experimentation, gave much better results as it preserved the base meaning of the words. Precisely, lemmatization could convert words to their root forms without drastically altering their meaning. It took some time to figure out which method to use and how to apply it correctly, especially since installing nltk and selecting the correct part-of-speech (POS) tags was initially unfamiliar. I found multiple resources online that helped guide me, such as tutorials and documentation from nltk, which greatly assisted me in learning how to use the WordNetLemmatizer. After running the preprocessed reviews through lemmatization, I was confident I had chosen the right approach for this task.

2. **_TF-IDF Function_**: 
The most challenging and time-consuming function to implement was the one responsible for calculating the TF-IDF (Term Frequency-Inverse Document Frequency) scores. The complexity lay in the data structure, as I had to process dictionaries within lists. This required iterating over every term in each review and calculating its frequency and inverse document frequency across all reviews. The challenge was developing the correct logic to ensure that the word counts and document frequencies were correctly calculated without missing anything. This involved debugging the logic multiple times and refining the loops and conditions to ensure accurate results. After several failed attempts and adjustments, I devised a method that correctly computed the TF-IDF scores for each word, ensuring the results were consistent with theoretical expectations. Testing the final implementation on my sample set of reviews helped me validate that the function worked as intended, which was a massive relief after the time invested.

### Resources

1. https://www.datacamp.com/tutorial/stemming-lemmatization-python
2. https://www.nltk.org
3. https://stackoverflow.com/questions/50236117/scraping-ssl-certificate-verify-failed-error-for-http-en-wikipedia-org
4. https://www.w3schools.com/python/ref_dictionary_get.asp

### *DO NOT EDIT BELOW THIS LINE*
---

## Goal

The goals of this project are:

* creating custom functions
* using I/O to read files
* creating language models! 

## Description

In this project, you will extend the tasks of processing text from the previous project to develop models for language. Language models help create a concise representation of text which helps in more complicated tasks such as classification! We will develop two language models - Bag of Words and TF-IDF.

Following are the main tasks for the project:

### `language_model.py`

Create a new Python file that contains the necessary functions.

### `preprocess_text`

Create a function called `preprocess_text`. This function will take in a single argument called `text` which is of data type `string`. It will return a list of pre-processed string. After you take in a string, it should do the following steps:

* convert all text into lowercase letters
* remove punctuation
* tokenize the words
* remove stop words
* (Extra Credit) - Use stemming and lemmatization to convert words into their root form.

There is a file called `list_of_stop_words.txt` that you should use to remove them. (Hint: You can use the function from `helper.py` from the previous project to read the contents of `list_of_stop_words.txt`).


```python
>>> movie_review: str = "This was a fantastic movie. I loved it so much. It was a great time watching this movie"
>>> preprocessed_text: list[str] = preprocess_text(movie_review)
>>> print(preprocessed_text) # Without lemmatization/stemming
['fantastic', 'movie', 'loved', 'great', 'time', 'watching', 'movie']
# Assuming that the function preprocess_text also stems and lemmatizes each word
>>> print(preprocessed_text) # With stemming/lemmatization
['fantastic', 'movie', 'love', 'great', 'time', 'watch', 'movie']
```

### `get_unique_words`

Create a function called `get_unique_words` that takes in a single argument called `preprocessed_review`, which is a `list` of `string` and returns a `set` of unique words in that review.

```python
>>> movie_review: str = "This was a fantastic movie. I loved it so much. It was a great time watching this movie"
>>> preprocessed_text: list[str] = preprocess_text(movie_review) # Without lemmatization/stemming
>>> unique_words: list[str] = get_unique_words(preprocessed_text)
>>> print(unique_words)
{'fantastic', 'movie', 'loved', 'great', 'time', 'watching'}
```


### `create_vocabulary`

Create a function called `create_vocabulary` that takes in a single argument called `reviews` which is a `list` of `string`. The function returns a `set` of unique words in all the `reviews`. This function should call `preprocess_text` on each item in the `reviews` list and then call `get_unique_words` on its output. Each time it should update the set of words it gets from `get_unique_words`.

```python
>>> reviews: list[str] = [
    "This was such an amazing movie",
    "I am such a big fan of this movie, it is amazing",
    "This was a horrible movie. Not recommended at all."
]
>>> vocabulary = create_vocabulary(reviews)
>>> print(vocabulary)
{'amazing', 'movie', 'big', 'fan', 'horrible', 'recommended'.}
```

### `calculate_term_frequency_for_each_review`

Create a function called `calculate_term_frequency_for_each_review` which takes in a single argument - `preprocessed_review` which is a `list` of `string` and it returns a `dictionary`, where the `key` is the word in the `preprocessed_review` and the `value` is the word count. 

```python
>>> movie_review: str = "This was a horrible movie. Not recommended at all."
>>> preprocessed_review: list[str] = preprocess_text(movie_review)
>>> term_frequency: dict[str, int]  = calculate_term_frequency_for_each_review(preprocessed_review)
>>> print(term_frequency)
{'horrible': 1, 'movie': 1, 'recommended': 1}
```

### `calculate_term_frequency_for_corpus`

Next, create a function called `calculate_term_frequency_for_corpus`. This function will take in a list of `preprocessed_reviews`, where each review is a `list` of `list` of `string`. It will return a `list` of `dictionaries`. Each `dictionary` represents the term frequency for one review, where the `key` is the word and the `value` is its frequency in that particular review.

```python
>>> reviews: list[str] = [
    "This was a fantastic movie",
    "I loved this movie so much",
    "Fantastic time watching this movie!"
]
>>> preprocessed_reviews: list[list[str]] = [preprocess_text(review) for review in reviews]
>>> tf_corpus = calculate_term_frequency_for_corpus(preprocessed_reviews)
>>> print(tf_corpus)
[
  {'fantastic': 1, 'movie': 1},
  {'loved': 1, 'movie': 1},
  {'fantastic': 1, 'time': 1, 'watching': 1, 'movie': 1}
]
```

### `calculate_inverse_document_frequency`

Now, create the final function `calculate_tf_idf` that will compute the Term Frequency-Inverse Document Frequency (TF-IDF) score for each word in each review. This function will take in two arguments:

* `term_frequencies`: A list of dictionaries containing a dictionary of word frequency from each review.
* `vocabulary`: A set of unique words from the entire corpus.

It will return a list of dictionaries, where each dictionary represents the TF-IDF scores for each word in a review.

To compute the TF-IDF score, follow these steps:

* For each word, calculate Term Frequency (TF) in each review.
* Calculate the Inverse Document Frequency (IDF) for the word across all reviews:

$$ IDF(𝑡) = log (𝑁 / df (𝑡)) $$

where 𝑁 is the total number of reviews, and df(𝑡) is the number of reviews containing the word 𝑡

* Multiply TF by IDF to get the TF-IDF score.

```python
>>> reviews = [
    "This was a fantastic movie",
    "I loved this movie so much",
    "Fantastic time watching this movie!"
]

# Preprocess each review
>>> preprocessed_reviews = [preprocess_text(review) for review in reviews]
>>> print(preprocessed_reviews)
[['fantastic', 'movie'], ['loved', 'movie'], ['fantastic', 'time', 'watching', 'movie']]

# Calculate term frequencies for each review
>>> term_frequencies = calculate_term_frequency_for_corpus(preprocessed_reviews)
>>> print(term_frequencies)
[
  {'fantastic': 1, 'movie': 1},
  {'loved': 1, 'movie': 1},
  {'fantastic': 1, 'time': 1, 'watching': 1, 'movie': 1}
]

# Create the vocabulary from all reviews
>>> vocabulary = create_vocabulary(reviews)
>>> print(vocabulary)
{'fantastic', 'movie', 'loved', 'time', 'watching'}

# Calculate TF-IDF scores
>>> tf_idf_corpus = calculate_tf_idf(term_frequencies, vocabulary)
>>> for doc_tf_idf in tf_idf_corpus:
...     print(doc_tf_idf)
{'fantastic': 0.4055, 'movie': 0.0}
{'loved': 1.0986, 'movie': 0.0}
{'fantastic': 0.4055, 'time': 1.0986, 'watching': 1.0986, 'movie': 0.0}
```
