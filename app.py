import streamlit as st
from content_based_filtering import content_recommendation
from scipy.sparse import load_npz
import pandas as pd
from numpy import load
from hybrid_recommendations import HybridRecommenderSystem

# Load the data
cleaned_data_path = "data/cleaned_data.csv"
songs_data = pd.read_csv(cleaned_data_path)

# Load the transformed data
transformed_data_path = "data/transformed_data.npz"
transformed_data = load_npz(transformed_data_path)

# Load the track ids
track_ids_path = "data/track_ids.npy"
track_ids = load(track_ids_path, allow_pickle=True)

# Load the filtered songs data
filtered_data_path = "data/collab_filtered_data.csv"
filtered_data = pd.read_csv(filtered_data_path)

# Load the interaction matrix
interaction_matrix_path = "data/interaction_matrix.npz"
interaction_matrix = load_npz(interaction_matrix_path)

# Load the transformed hybrid data
transformed_hybrid_data_path = "data/transformed_hybrid_data.npz"
transformed_hybrid_data = load_npz(transformed_hybrid_data_path)

# UI Title and Description
st.title(' Song Recommender')
st.write('Enter a song and artist name to get personalized music recommendations.')

# Inputs
song_name = st.text_input('Enter a song name:').strip().lower()
artist_name = st.text_input('Enter the artist name:').strip().lower()
k = st.selectbox('How many recommendations do you want?', [5, 10, 15, 20], index=1)

# Check song availability
if ((filtered_data["name"] == song_name) & (filtered_data["artist"] == artist_name)).any():
    filtering_type = "Hybrid Recommender System"
    diversity = st.slider("Diversity in Recommendations", 1, 9, 5, 1)
    content_weight = 1 - (diversity / 10)
    
    st.bar_chart(pd.DataFrame({
        "type": ["Personalized", "Diverse"],
        "ratio": [10 - diversity, diversity]
    }))
else:
    filtering_type = "Content-Based Filtering"

# Generate recommendations
if st.button('Get Recommendations'):
    if filtering_type == 'Content-Based Filtering':
        if ((songs_data["name"] == song_name) & (songs_data['artist'] == artist_name)).any():
            st.write(f'Recommendations for **{song_name}** by **{artist_name}**')
            recommendations = content_recommendation(song_name, artist_name, songs_data, transformed_data, k)
        else:
            st.warning(f"Sorry, '{song_name}' by '{artist_name}' not found in database.")
            st.stop()
    else:
        recommender = HybridRecommenderSystem(number_of_recommendations=k, weight_content_based=content_weight)
        try:
            recommendations = recommender.give_recommendations(song_name, artist_name, filtered_data,
                                                                track_ids, transformed_hybrid_data, interaction_matrix)
            st.write(f'Recommendations for **{song_name}** by **{artist_name}**')
        except Exception as e:
            st.error(f"Failed to generate hybrid recommendations: {str(e)}")
            st.stop()

    # Display recommendations
    for i, rec in recommendations.iterrows():
        name = rec['name'].title()
        artist = rec['artist'].title()
        url = rec['spotify_preview_url']
        if i == 0:
            st.subheader("▶ Currently Playing")
        elif i == 1:
            st.subheader("🎶 Next Up")
        else:
            st.markdown(f"**{i}. {name} by {artist}**")
        st.audio(url)
        st.markdown("---")
