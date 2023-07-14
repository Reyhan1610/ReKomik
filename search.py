from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def search(query, data, tfidf, cosine_sim, tfidf_matrix, top_k=15): # Menambahkan variabel tfidf_matrix sebagai parameter
    # Menerapkan vectorizer ke query
    query_vector = tfidf.transform([query])

    # Menghitung cosine similarity antara query dan seluruh dokumen
    sim_scores = cosine_similarity(query_vector, tfidf_matrix).flatten() # Menggunakan variabel tfidf_matrix yang diterima dari parameter

    # Mengurutkan dokumen berdasarkan similarity score dari yang tertinggi
    sorted_indexes = sim_scores.argsort()[::-1][:top_k]

    # Mengambil data dari dokumen yang relevan
    results = []
    for i in sorted_indexes:
        result = {
            'judul': data.iloc[i]['judul_elex'] + ' ' + data.iloc[i]['judul_webtoon'],
            'sinopsis': data.iloc[i]['sinopsis_elex'] + ' ' + data.iloc[i]['sinopsis_webtoon'],
            'similarity_score': round(sim_scores[i], 2)
        }
        results.append(result)


    return results
