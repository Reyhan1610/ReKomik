import unittest
from main import app, data, tfidf, cosine_sim, tfidf_matrix
from preprocessing import preprocess
from search import search

class TestApp(unittest.TestCase):
    
    def test_flask_connection(self):
        tester = app.test_client(self)
        response = tester.get('/')
        self.assertEqual(response.status_code, 200)

    def test_data_loaded(self):
        self.assertIsNotNone(data)
        self.assertGreater(len(data['dokumen']), 0)

    def test_tfidf_vectorizer(self):
        self.assertIsNotNone(tfidf)
        self.assertGreater(tfidf.idf_.size, 0)

    def test_cosine_similarity(self):
        self.assertIsNotNone(cosine_sim)
        self.assertGreater(cosine_sim.shape[0], 0)

    def test_query_preprocessing(self):
        query = " CEK Fungsi Preprocessing pada sistem untuk rekomendasi Komik! "
        self.assertEqual(preprocess(query), "cek fungsi preprocess sistem rekomendasi komik")

    def test_search(self):
        query = "komik lucu"
        preprocessed_query = preprocess(query)
        results = search(preprocessed_query, data, tfidf, cosine_sim, tfidf_matrix, top_k=15)
        self.assertEqual(len(results), 15)

    def test_result_display(self):
        query = "komik lucu"
        tester = app.test_client(self)
        response = tester.post('/search', data=dict(query=query))
        self.assertIn(b'<title>Search</title>', response.data)

if __name__ == '__main__':
    unittest.main()
