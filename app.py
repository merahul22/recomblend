import streamlit as st
from content_based_filtering import content_recommendation
from scipy.sparse import load_npz
import pandas as pd
from numpy import load
from hybrid_recommendations import HybridRecommenderSystem

# --- Load Data ---
@st.cache_data
def load_data():
    songs_data = pd.read_csv("data/cleaned_data.csv")
    filtered_data = pd.read_csv("data/collab_filtered_data.csv")
    transformed_data = load_npz("data/transformed_data.npz")
    transformed_hybrid_data = load_npz("data/transformed_hybrid_data.npz")
    interaction_matrix = load_npz("data/interaction_matrix.npz")
    track_ids = load("data/track_ids.npy", allow_pickle=True)
    return songs_data, filtered_data, transformed_data, transformed_hybrid_data, interaction_matrix, track_ids

songs_data, filtered_data, transformed_data, transformed_hybrid_data, interaction_matrix, track_ids = load_data()

# --- App Header ---
st.set_page_config(page_title="RecoBlend: Song Recommender", layout="wide")
st.title("RecomBlend - AI Powered Song Recommender")
st.markdown("Get personalized music recommendations by entering your favorite song and artist.")

# --- Input Section ---
st.sidebar.header("Input Details")
song_name = st.sidebar.text_input("Enter a song name").strip().lower()
artist_name = st.sidebar.text_input("Enter the artist name").strip().lower()
k = st.sidebar.selectbox("Number of Recommendations", [5, 10, 15, 20], index=1)

# --- Check if song exists in filtered_data ---
is_in_filtered = ((filtered_data["name"] == song_name) & (filtered_data["artist"] == artist_name)).any()

# --- Recommendation Type ---
if is_in_filtered:
    filtering_type = "Hybrid Recommender System"
    st.sidebar.markdown("Song found in collaborative dataset.")
    diversity = st.sidebar.slider("Diversity in Recommendations", 1, 9, 5)
    content_weight = 1 - (diversity / 10)

    st.sidebar.subheader("Recommendation Mix")
    st.sidebar.bar_chart(pd.DataFrame({
        "Type": ["Personalized", "Diverse"],
        "Ratio": [10 - diversity, diversity]
    }))
else:
    filtering_type = "Content-Based Filtering"
    st.sidebar.warning("Song not found in collaborative dataset. Using content-based filtering.")

# --- Button to Generate Recommendations ---
if st.sidebar.button("Get Recommendations"):
    st.markdown("---")
    if filtering_type == "Content-Based Filtering":
        if ((songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)).any():
            st.subheader(f"Recommendations for '{song_name.title()}' by '{artist_name.title()}'")
            recommendations = content_recommendation(song_name, artist_name, songs_data, transformed_data, k)
        else:
            st.error(f"Song '{song_name}' by '{artist_name}' not found in dataset.")
            st.stop()
    else:
        recommender = HybridRecommenderSystem(number_of_recommendations=k, weight_content_based=content_weight)
        try:
            recommendations = recommender.give_recommendations(
                song_name, artist_name, filtered_data,
                track_ids, transformed_hybrid_data, interaction_matrix
            )
            st.subheader(f"Hybrid Recommendations for '{song_name.title()}' by '{artist_name.title()}'")
        except Exception as e:
            st.error(f"Failed to generate hybrid recommendations: {str(e)}")
            st.stop()

    # --- Display Recommendations ---
    for i, rec in recommendations.iterrows():
        name = rec['name'].title()
        artist = rec['artist'].title()
        url = rec['spotify_preview_url']

        if i == 0:
            st.subheader("Currently Playing")
        elif i == 1:
            st.subheader("Next Up")
        else:
            st.markdown(f"**{i}. {name} by {artist}**")
        st.audio(url)
        st.markdown("---")
