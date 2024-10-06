from helper import get_file_contents

list_of_stop_words: str = get_file_contents("list_of_stop_words.txt")
characters_to_remove = ['!', ',', '.', '?', ':', ';', '-', '_']
translation_table = str.maketrans('', '', ''.join(characters_to_remove))

def preprocess_text(text: str) -> str:
    text_lower: str  = text.lower()
    text_no_punctuation: str = text_lower.translate(translation_table)
    text_tokenized: list[str] = text_no_punctuation.split(" ")
    text_no_stop_words: list[str]=[f for f in text_tokenized if f not in list_of_stop_words]
    return text_no_stop_words

def get_unique_words(text: list[str]) -> set[str]:
    text_unique: set[str] = set(text)
    return text_unique

def create_vocabulary(reviews: list[str]) -> set[str]:
    vocabulary: list[str] = []
    for each_item in reviews:
        vocabulary.extend(get_unique_words(preprocess_text(each_item)))
    return set(vocabulary)

def calculate_term_frequency_for_each_review(preprocessed_review: list[str]) -> dict[str,int]:
    term_frequency: dict[str,int] = {}
    for each_item in preprocessed_review:
        term_frequency.update({each_item:preprocessed_review.count(each_item)})
    return term_frequency

def calculate_term_frequency_for_corpus(preprocessed_reviews: list[list[str]]) -> list[dict[str,int]]:
    tf_corpus: list[dict[str,int]] = []
    for each_review in preprocessed_reviews:
        term_frequency: dict[str,int] = {}
        for each_item in each_review:
            term_frequency.update({each_item : each_review.count(each_item)})
        tf_corpus.append(term_frequency)
    return tf_corpus 



