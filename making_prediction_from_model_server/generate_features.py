import string
import pandas as pd
from spellchecker import SpellChecker

spell=SpellChecker()


def num_single_quote_error(text: str):
    num_errors = 0
    words = text.split(" ")

    ind1 = 0
    while ind1 < len(words) - 1:
        ind2 = ind1 + 1

        word1 = words[ind1]
        word2 = words[ind2]

        if word1.isalpha() and word2.isalpha():
            word1 = word1.lower()
            word2 = word2.lower()

            if len(word1) > 1 and len(word2) == 1:
                if word2 not in ['a', 'i']:
                    num_errors += 1

            elif word1 == 'i' and (word2 == "ll" or word2 == "d"):
                num_errors += 1

        ind1 += 1

    return num_errors



def num_spacing_error(text: str):
    words = text.split(" ")
    num_errors = words.count("")

    return num_errors


def num_space_absence_after_sentence_completion(text: str):
    num_errors = 0
    words = text.split(" ")

    for word in words:
        if any(punct in word for punct in ['.', '?', '!']):
            if word[-1] not in ['.', '?', '!']:
                num_errors += 1

    return num_errors



def num_capitalized_words(text: str):
    num_cap_words = 0
    words = text.split(" ")

    table = str.maketrans('', '', string.punctuation)

    for word in words:
        word = word.translate(table)
        if word.isalpha():
            if word.isupper():
                if word != "I":
                    num_cap_words += 1

    return num_cap_words



def num_spelling_errors(text: str):
    num_errors = 0
    table = str.maketrans('', '', string.punctuation)

    words = text.split(" ")
    for word in words:
        word = word.translate(table)
        if word.isalpha():
            word = word.lower()
            if word not in spell:
                num_errors += 1

    return num_errors


def num_words(text: str):
    words = text.split(" ")
    num = len(words) - words.count("")
    return num


def num_punctuations(text: str):
    punct = list(string.punctuation)
    num = 0
    words = text.split(" ")

    for word in words:
        if any(p in word for p in punct):
            num += 1

    return num

def num_numeric_values(text: str):
    numeric_values = list("0123456789")
    num = 0
    words = text.split(" ")

    for word in words:
        if any(n in word for n in numeric_values):
            num += 1

    return num


def num_capitalization_absence_after_sentence_completion(text: str):
    num_errors = 0
    punct_to_check_for = ['.', '?', '!']

    for punct in punct_to_check_for:
        sent = text.split(punct)
        sent = [s.strip() for s in sent if s != '']

        for s in sent:
            if s != '':
                first_letter = s[0]
                if first_letter.isalpha():
                    if not first_letter.isupper():
                        num_errors += 1

    return num_errors




def GenerateFeatures(text:str):
    data=pd.DataFrame({
        "text":[text]
    })

    data["num_single_quote_error"] = data["text"].map(
        lambda text: num_single_quote_error(text=text)
    )

    data["num_spacing_error"] = data["text"].map(
        lambda text: num_spacing_error(text=text)

    )

    data["num_space_absence_after_sentence_completion"] = data["text"].map(
        lambda text: num_space_absence_after_sentence_completion(text=text)
    )

    data["num_capitalized_words"] = data["text"].map(
        lambda text: num_capitalized_words(text=text)
    )

    data["num_capitalization_absence_after_sentence_completion"] = data["text"].map(
        lambda text: num_capitalization_absence_after_sentence_completion(text=text)
    )

    data["num_spelling_errors"] = data["text"].map(
        lambda text: num_spelling_errors(text=text)

    )

    data["num_words"] = data["text"].map(
        lambda text: num_words(text=text)
    )

    data["num_punctuations"] = data["text"].map(
        lambda text: num_punctuations(text=text)
    )

    data["num_numeric_values"] = data["text"].map(
        lambda text: num_numeric_values(text=text)
    )

    data["text"] = data["text"].map(
        lambda text: text.strip()
    )


    return data



if __name__=="__main__":
    text="""
    Welcome, to my project everybody, thislnvjgnekwgb  cbhbjhwf 265476i bmnwbvn e     ,mnfhgfuiukdehkhjudgufggfkggwjfgjv
    """

    data=GenerateFeatures(text=text)
    print(data)
    print(f"Num words: {data.iloc[0]['num_words']}")












