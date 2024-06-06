# Install necessary libraries
# pip install pandas scikit-learn

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
def load_data(file_path):
    return pd.read_csv(file_path, encoding='latin-1')

# Preprocessing
def preprocess_data(df):
    # Combine relevant features into a single string
    df['combined_features'] = df.apply(lambda row: f"{row['track_name']} {row['artist(s)_name']} {row['released_year']} {row['released_month']} {row['released_day']} {row['streams']}", axis=1)
    return df

# Create TF-IDF Vectorizer
def create_tfidf_matrix(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    return tfidf, tfidf_matrix

# Calculate cosine similarity matrix
def calculate_cosine_similarity(tfidf_matrix):
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get recommendations
def get_recommendations(input_title, df, tfidf, tfidf_matrix):
    input_data = {'track_name': input_title, 'artist(s)_name': '', 'released_year': '', 'released_month': '', 'released_day': '', 'streams': ''}
    input_df = pd.DataFrame([input_data])
    input_df = preprocess_data(input_df)
    input_tfidf = tfidf.transform(input_df['combined_features'])
    cosine_sim_input = cosine_similarity(input_tfidf, tfidf_matrix)
    
    sim_scores = list(enumerate(cosine_sim_input[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 recommendations
    song_indices = [i[0] for i in sim_scores]
    return df[['track_name', 'artist(s)_name']].iloc[song_indices]

def main():
    # Load the dataset
    file_path = 'spotify-2023.csv'  # Update with your file path
    df = load_data(file_path)
    
    # Preprocess the data
    df = preprocess_data(df)
    
    # Create TF-IDF matrix
    tfidf, tfidf_matrix = create_tfidf_matrix(df)
    
    # Calculate cosine similarity matrix
    cosine_sim = calculate_cosine_similarity(tfidf_matrix)
    
    # Input from user
    song_title = input('Enter the Artist keyword: ')
    
    if song_title:
        recommendations = get_recommendations(song_title, df, tfidf, tfidf_matrix)
        
        print('Recommended Songs:')
        for index, row in recommendations.iterrows():
            print(f"{row['track_name']} by {row['artist(s)_name']}")
    else:
        print('Please enter the song title for recommendations.')

if __name__ == "__main__":
    main()
