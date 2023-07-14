from sklearn.model_selection import KFold
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

# Load data
data = load_data()

# Inisialisasi vectorizer dengan menggunakan TF-IDF
tfidf = TfidfVectorizer()

# Menghitung bobot dari masing-masing dokumen
tfidf_matrix = tfidf.fit_transform(data['dokumen'])

# Menghitung cosine similarity dari seluruh pasangan dokumen
cosine_sim = cosine_similarity(tfidf_matrix)

# Membagi data menjadi 5 fold untuk cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# List untuk menyimpan hasil evaluasi
evaluations = []

for train_index, test_index in kf.split(data):
    # Membagi data menjadi train set dan test set berdasarkan fold
    train_data = data.iloc[train_index]
    test_data = data.iloc[test_index]

    # Load data untuk train set
    train_data['judul'] = train_data['judul_elex'] + ' ' + train_data['judul_webtoon']
    train_data['sinopsis'] = train_data['sinopsis_elex'] + ' ' + train_data['sinopsis_webtoon']

    # Train TF-IDF vectorizer
    tfidf_matrix_train = tfidf.fit_transform(train_data['dokumen'])

    # Train cosine similarity
    cosine_sim_train = cosine_similarity(tfidf_matrix_train)

    # Evaluasi pada test set
    total_precision = 0.0
    total_recall = 0.0

    for i, query in test_data.iterrows():
        # Preprocess query
        preprocessed_query = preprocess(query['dokumen'])

        # Search
        results = search(preprocessed_query, train_data, tfidf, cosine_sim_train, tfidf_matrix_train, top_k=15)

        # True relevance judul
        true_judul = query['judul_elex'] + ' ' + query['judul_webtoon']

        # Calculate precision and recall
        precision = 0.0
        recall = 0.0

        for result in results:
            if result['judul'] == true_judul:
                precision = 1.0
                break

        if precision == 1.0:
            total_precision += 1.0
            total_recall += 1.0

    fold_evaluation = {
        'precision': total_precision / len(test_data),
        'recall': total_recall / len(test_data)
    }

    evaluations.append(fold_evaluation)

# Calculate average precision and recall across all folds
avg_precision = sum([evaluation['precision'] for evaluation in evaluations]) / len(evaluations)
avg_recall = sum([evaluation['recall'] for evaluation in evaluations]) / len(evaluations)

# Print evaluation results
print("Average Precision:", avg_precision)
print("Average Recall:", avg_recall)
