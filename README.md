# Bag of Words & TF-IDF Language Models

This project focuses on building foundational NLP language models — **Bag-of-Words** and **TF-IDF** — using raw Python. It includes complete text preprocessing, vocabulary extraction, term frequency, and inverse document frequency computation. Developed as part of **CS154**, an introductory Python course with NLP components.

---

## 📌 Features

- Text preprocessing pipeline with:
  - Lowercasing
  - Punctuation removal
  - Tokenization
  - Stop word removal
  - Optional stemming/lemmatization (Extra Credit)
- Functions to:
  - Extract unique words
  - Create corpus vocabulary
  - Calculate term frequencies for reviews and corpora
  - Compute TF-IDF scores for each review
- Custom logic built from scratch without external NLP libraries

---

## 🧠 Reflections

- **Lemmatization**: Switching from stemming to lemmatization significantly improved word representation and model accuracy. Learning to use NLTK's `WordNetLemmatizer` and POS tagging was crucial.
- **TF-IDF Calculation**: This part was the most logic-intensive. I iteratively refined loops and conditionals to handle nested dictionaries and frequency normalization. Debugging and testing thoroughly ensured accurate output.

---

## 🛠️ How to Use

1. Install NLTK if using lemmatization:
   ```bash
   pip install nltk
   ```

2. Download necessary NLTK resources:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   nltk.download('averaged_perceptron_tagger')
   ```

3. Run your Python file:
   ```bash
   python language_model.py
   ```

4. Place `list_of_stop_words.txt` in the project directory.

---

## 📚 Resources

- [Stemming & Lemmatization – DataCamp](https://www.datacamp.com/tutorial/stemming-lemmatization-python)
- [NLTK Documentation](https://www.nltk.org)
- [W3Schools Python `.get()` Reference](https://www.w3schools.com/python/ref_dictionary_get.asp)
- [SSL Error Fix – Stack Overflow](https://stackoverflow.com/questions/50236117/scraping-ssl-certificate-verify-failed-error-for-http-en-wikipedia-org)

---

## 🪪 License

This project is licensed under the MIT License.
