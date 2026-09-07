import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import mwparserfromhell

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_wikitext(text):
    if not text:
        return ""
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)

    wikicode = mwparserfromhell.parse(text)
    for link in wikicode.filter_wikilinks():
        title = str(link.title).strip().lower()
        if title.startswith("file:") or title.startswith("image:"):
            if link.text:
                parts = str(link.text).split("|")
                caption = parts[-1].strip()
                wikicode.replace(link, caption)
            else:
                wikicode.replace(link, "")

    clean = wikicode.strip_code()
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def preprocess(text):
    text = text.lower()
    text = re.sub(r"_", " ", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    lemmatized_tokens = []
    for token in tokens:
        if token not in STOP_WORDS:
            lemmatized_tokens.append(LEMMATIZER.lemmatize(token))
    return " ".join(lemmatized_tokens)