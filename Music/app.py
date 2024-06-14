from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Fungsi untuk memuat dataset
def load_data(file_path):
    # Memuat dataset dari path file yang ditentukan
    return pd.read_csv(file_path, encoding='latin-1')

# Fungsi untuk preprocessing data
def preprocess_data(df):
    # Menggabungkan kolom-kolom yang relevan menjadi satu string teks untuk setiap baris
    columns_to_combine = ['name', 'artist', 'spotify_id', 'preview', 'img', 'danceability', 'energy', 'loudness', 
                          'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'acousticness_artist', 
                          'danceability_artist', 'energy_artist', 'instrumentalness_artist', 'liveness_artist', 'speechiness_artist', 
                          'valence_artist']
    # Menggabungkan kolom-kolom tersebut menjadi satu kolom baru 'combined_features'
    df['combined_features'] = df[columns_to_combine].astype(str).agg(' '.join, axis=1)
    return df

# Fungsi untuk membuat TF-IDF Vectorizer
def create_tfidf_matrix(df):
    # Inisialisasi TF-IDF Vectorizer dengan mengabaikan stop words dalam bahasa Inggris
    tfidf = TfidfVectorizer(stop_words='english')
    # Mengubah data teks menjadi matriks TF-IDF
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    return tfidf, tfidf_matrix

# Fungsi untuk menghitung matriks kesamaan kosinus
def calculate_cosine_similarity(tfidf_matrix):
    # Menghitung kesamaan kosinus dari matriks TF-IDF
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

# Fungsi untuk mendapatkan rekomendasi
def get_recommendations(keyword, df, tfidf, tfidf_matrix):
    # Membuat data input dari keyword yang diberikan
    input_data = {'combined_features': keyword}
    input_df = pd.DataFrame([input_data])
    # Mengubah data input menjadi matriks TF-IDF
    input_tfidf = tfidf.transform(input_df['combined_features'])
    # Menghitung kesamaan kosinus antara data input dan matriks TF-IDF
    cosine_sim_input = cosine_similarity(input_tfidf, tfidf_matrix)
    
    # Mendapatkan skor kesamaan dan mengurutkannya
    sim_scores = list(enumerate(cosine_sim_input[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Mengambil 10 rekomendasi teratas
    sim_scores = sim_scores[1:11]
    song_indices = [i[0] for i in sim_scores]
    return df.iloc[song_indices]

@app.route('/')
def index():
    # Merender template index.html
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def recommendations():
    # Path ke file dataset
    file_path = 'music.csv'
    # Memuat dan memproses data
    df = load_data(file_path)
    df = preprocess_data(df)
    tfidf, tfidf_matrix = create_tfidf_matrix(df)

    # Mengambil keyword dari form yang dikirimkan
    keyword = request.form['keyword']
    # Mendapatkan rekomendasi berdasarkan keyword
    recommendations = get_recommendations(keyword, df, tfidf, tfidf_matrix)
    # Merender template recommendations.html dengan data rekomendasi
    return render_template('recommendations.html', recommendations=recommendations.to_dict('records'))

if __name__ == "__main__":
    # Menjalankan aplikasi Flask dalam mode debug
    app.run(debug=True)
