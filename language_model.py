from helper import get_file_contents
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

list_of_stop_words: str = get_file_contents("list_of_stop_words.txt")
characters_to_remove = ["!", ",", ".", "?", ":", ";", "-", "_"]
translation_table = str.maketrans("", "", "".join(characters_to_remove))


def preprocess_text(text: str) -> set[str]:
    text_lower: str = text.lower()
    text_no_punctuation: str = text_lower.translate(translation_table)
    text_tokenized: list[str] = text_no_punctuation.split(" ")
    text_no_stop_words: list[str] = [
        f for f in text_tokenized if f not in list_of_stop_words
    ]
    text_stemmed: list[str] = [wnl.lemmatize(f, pos="v") for f in text_no_stop_words]
    return set(text_stemmed)


movie_review: str = "This was a fantastic movie. I loved it so much. It was a great time watching this movie"
preprocessed_text: list[str] = preprocess_text(movie_review)
print(preprocessed_text)


def get_unique_words(text: list[str]) -> set[str]:
    text_unique: set[str] = set(text)
    return text_unique


def create_vocabulary(reviews: list[str]) -> set[str]:
    vocabulary: list[str] = []
    for each_item in reviews:
        vocabulary.extend(get_unique_words(preprocess_text(each_item)))
    return set(vocabulary)


def calculate_term_frequency_for_each_review(
    preprocessed_review: list[str],
) -> dict[str, int]:
    term_frequency: dict[str, int] = {}
    for each_item in preprocessed_review:
        term_frequency.update({each_item: preprocessed_review.count(each_item)})
    return term_frequency


def calculate_term_frequency_for_corpus(
    preprocessed_reviews: list[list[str]],
) -> list[dict[str, int]]:
    tf_corpus: list[dict[str, int]] = []
    for each_review in preprocessed_reviews:
        term_frequency: dict[str, int] = {}
        for each_item in each_review:
            term_frequency.update({each_item: each_review.count(each_item)})
        tf_corpus.append(term_frequency)
    return tf_corpus
