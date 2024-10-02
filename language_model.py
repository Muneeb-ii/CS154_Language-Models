from helper import get_file_contents




list_of_stop_words: str = get_file_contents("list_of_stop_words.txt")
characters_to_remove = ['!', ',', '.', '?', ':', ';', '-', '_']
translation_table = str.maketrans('', '', ''.join(characters_to_remove))

def preprocess_text(text: str) -> str:
    text_lower: str  = text.lower()
    text_no_punctuation: str = text_lower.translate(translation_table)
    text_tokenized: list[str] = text_no_punctuation.split(" ")
    text_no_stop_words: list[str]=[]

    for each_word in text_tokenized:
        if(each_word in list_of_stop_words):
            pass
        else:
            text_no_stop_words.append(each_word)
    return text_no_stop_words

def get_unique_words(text: list[str]) -> set[str]:
    text_unique: set[str] = set(text)
    return text_unique