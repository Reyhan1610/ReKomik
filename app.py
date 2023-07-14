from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
import pandas as pd
import re
from data import load_data
from search import search
from preprocessing import preprocess
import mysql.connector
import string
import matplotlib.pyplot as plt
import numpy as np
import base64


app = Flask(__name__)

# Load data
data = load_data()

# Inisialisasi vectorizer dengan menggunakan TF-IDF
tfidf = TfidfVectorizer()

# Menghitung bobot dari masing-masing dokumen
tfidf_matrix = tfidf.fit_transform(data['dokumen'])

# Menghitung cosine similarity dari seluruh pasangan dokumen
cosine_sim = cosine_similarity(tfidf_matrix)

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/komik_diagram')
def komik_diagram_page():
    # Menghitung jumlah data komik Elex dan Webtoon
    elex_count = len(data[data['judul_elex'] != ''])
    webtoon_count = len(data[data['judul_webtoon'] != ''])

    # Visualisasi perbandingan data komik Elex dan Webtoon dengan pie diagram dan bar diagram
    labels = ['Elex Media', 'Webtoon Indonesia']
    counts = [elex_count, webtoon_count]

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(6, 8))

    # Pie diagram
    ax1.pie(counts, labels=labels, autopct='%1.1f%%')
    ax1.set_title('Perbandingan Data Komik Elex Media dan Webtoon')

    # Bar diagram
    x = np.arange(len(labels))
    ax2.bar(x, counts)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Jumlah')
    ax2.set_title('Jumlah Data Komik Elex dan Webtoon')

    # Mengubah diagram menjadi data URI
    img = BytesIO()
    plt.savefig(img, format='png', transparent=True)
    img.seek(0)
    img_uri = base64.b64encode(img.getvalue()).decode()
    
    data['judul'] = data['judul_elex'] + ' ' + data['judul_webtoon']
    
    return render_template('komik_diagram.html', data=data, img_uri=img_uri)

@app.route('/komik_all')
def komik_all_page():
    # Load data
    data['judul'] = data['judul_elex'] + ' ' + data['judul_webtoon']
    data['sinopsis'] = data['sinopsis_elex'] + ' ' + data['sinopsis_webtoon']

    # Mengurutkan data judul
    data['judul'] = sorted(data['judul'])

    return render_template('komik_all.html', data=data)

@app.route('/komik_elex')
def komik_elex_page():
    # Load data
    data['judul'] = data['judul_elex'] 
    data['sinopsis'] = data['sinopsis_elex'] 

    # Mengurutkan data judul
    #data['judul'] = sorted(data['judul'])

    return render_template('komik_elex.html', data=data)

@app.route('/komik_webtoon')
def komik_webtoon_page():
    # Load data
    data['judul'] = data['judul_webtoon'] 
    data['sinopsis'] = data['sinopsis_webtoon'] 

    # Mengurutkan data judul
    data['judul'] = sorted(data['judul'])

    return render_template('komik_webtoon.html', data=data)

@app.route('/search', methods=['GET', 'POST'])
def search_page():
    if request.method == 'POST':
        query = request.form['query']
        preprocessed_query = preprocess(query)

        results = search(preprocessed_query, data, tfidf, cosine_sim, tfidf_matrix, top_k=15)
        
        # Membuat daftar judul dan similarity score yang tidak kosong
        judul = []
        similarity_scores = []
        for result in results:
            if result['judul'] and result['similarity_score'] >= 0.2:
                judul.append(result['judul'])
                similarity_scores.append(result['similarity_score'])

        # Membuat visualisasi histogram similarity score jika terdapat data yang tidak kosong
        if judul and similarity_scores:
            plt.figure()
            plt.bar(judul, similarity_scores, color='#3F72AF')
            plt.xticks(rotation=90)
            plt.grid(True)
            plt.tight_layout()

            # Mengonversi gambar histogram menjadi data biner
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', transparent=True)
            img_buffer.seek(0)
            img_b64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        else:
            img_b64 = None

        # Menampilkan pesan error jika hasil pencarian kosong
        if not judul:
            error_message = f"Hasil pencarian untuk kata kunci '{query}' tidak ditemukan. Silakan coba menggunakan kata kunci yang lain!"
        else:
            error_message = None

        return render_template('search.html', query=query, results=results, img_b64=img_b64, error_message=error_message)
    return render_template('search.html')



def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

app.jinja_env.filters['remove_punctuation'] = remove_punctuation

if __name__ == '__main__':
    app.run(debug=True)
