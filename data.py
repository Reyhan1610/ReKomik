import mysql.connector
import pandas as pd
from preprocessing import preprocess


def load_data():
    # Koneksi ke database
    cnx = mysql.connector.connect(user='root', password='',
                                  host='localhost',
                                  database='ta1')
    
    # Membaca file csv pertama
    #data1 = pd.read_csv('data1.csv')
    
    # Membaca file csv kedua
    #data2 = pd.read_csv('data2.csv')

    # Menggabungkan data dari kedua file csv
    #data = pd.concat([data1, data2], ignore_index=True)

    # Query untuk mengambil data dari tabel
    query1 = "SELECT e.judul_elex, e.sinopsis_elex FROM elex e"
    
    # Membuat dataframe dari hasil query
    data1 = pd.read_sql(query1, cnx)

    # Query untuk mengambil data dari tabel
    query2 = "SELECT w.judul_webtoon, w.sinopsis_webtoon FROM webtoon_id w"
    
    # Membuat dataframe dari hasil query
    data2 = pd.read_sql(query2, cnx)

    data = pd.concat([data1, data2], ignore_index=True)
    
    # Menggabungkan judul dan sinopsis menjadi satu dokumen
    data = data.fillna('')

    data['dokumen'] = data['judul_elex'] + ' ' + data['sinopsis_elex'] + ' ' + data['judul_webtoon'] + ' ' + data['sinopsis_webtoon']
    data['dokumen'] = data['dokumen'].apply(preprocess)
    # Menutup koneksi ke database
    cnx.close()
    
    return data

