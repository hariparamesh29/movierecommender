import streamlit as st
import google.generativeai as genai
from PIL import Image
import requests
from io import BytesIO

# --- CONFIGURATION ---
st.set_page_config(page_title="🎬 AI Movie Recommender", layout="centered")
st.title("🎬 AI Movie Recommender")
st.write("Get personalized movie recommendations based on your favorite genres, actors, or plots.")

# --- GEMINI API SETUP ---
GEMINI_API_KEY = "AIzaSyAEZnp0jG8nWkiw3iPZ6Zf4DEDMzMTb5QI"  # Replace with your actual API Key
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# --- USER INPUT ---
with st.form("user_input_form"):
    genres = st.text_input("Favorite Genres (e.g., Sci-Fi, Comedy, Thriller)")
    actors = st.text_input("Favorite Actors (e.g., Tom Hanks, Emma Watson)")
    plot = st.text_area("Plot Preferences (e.g., time travel, love triangle, underdog story)")
    submitted = st.form_submit_button("Get Recommendations")

# --- FUNCTION TO GET POSTER FROM IMAGES API ---
def get_movie_poster(title):
    try:
        response = requests.get(f"https://api.duckduckgo.com/?q={title}+movie+poster&format=json")
        data = response.json()
        image_url = data.get("Image", "")
        return image_url if image_url else None
    except:
        return None

# --- GENERATE RECOMMENDATIONS ---
if submitted and (genres or actors or plot):
    with st.spinner("Generating recommendations with Gemini..."):
        prompt = f"""
        Act like a smart movie critic and recommend 3 great movies based on these preferences:
        - Genres: {genres}
        - Favorite Actors: {actors}
        - Plot Themes: {plot}

        For each movie, include:
        1. Title
        2. Short description
        3. Reason why it matches the user's preferences
        4. Mention a famous actor in it
        Return the answer in a clean format like:
        Title: ...
        Description: ...
        Match Reason: ...
        Actor: ...
        """

        response = model.generate_content(prompt)
        output = response.text

        # Split into movie chunks
        movies = output.split("Title:")
        for movie in movies[1:]:
            lines = movie.strip().split("\n")
            title = lines[0].strip()
            desc = next((l.replace("Description:", "").strip() for l in lines if "Description:" in l), "")
            reason = next((l.replace("Match Reason:", "").strip() for l in lines if "Match Reason:" in l), "")
            actor = next((l.replace("Actor:", "").strip() for l in lines if "Actor:" in l), "")

            # Get poster
            poster_url = get_movie_poster(title)
            st.subheader(title)
            cols = st.columns([1, 2])
            if poster_url:
                try:
                    img_data = requests.get(poster_url).content
                    img = Image.open(BytesIO(img_data))
                    cols[0].image(img, use_column_width=True)
                except:
                    cols[0].write("🎞️ [Poster unavailable]")
            else:
                cols[0].write("🎞️ [Poster unavailable]")

            with cols[1]:
                st.markdown(f"**Description:** {desc}")
                st.markdown(f"**Why You'll Like It:** {reason}")
                st.markdown(f"**Famous Actor:** {actor}")
else:
    st.info("Please enter at least one preference to get started.")
