import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def preprocess(text):
    # Mengubah teks menjadi lowercase
    text = text.lower()

    # Menghilangkan angka
    text = re.sub(r'\d+', '', text)

    # Menghilangkan tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Menghilangkan whitespace
    text = text.strip()

    # Tokenisasi
    tokens = word_tokenize(text)

    # Menghilangkan stop words
    stop_words = set(stopwords.words('indonesian'))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]

    # Menggabungkan kembali token-token yang sudah diproses
    processed_text = ' '.join(stemmed_tokens)

    return processed_text
