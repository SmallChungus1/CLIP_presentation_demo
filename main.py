import streamlit as st
from clip_inference import ClipInference
import torch
import os 
from dotenv import load_dotenv
from streamlit import session_state as state

load_dotenv()

print("Main.py reloaded")

st.title("CLIP Demo on Classification and Semantic Search")
collection_name=os.getenv("IMAGE_SEARCH_EMBEDDING_COLLECTION_NAME")

#store clip_inference instance in session state, if this prevent re-init every reload on button press?
if "clip_inference" not in state:
    state.clip_inference = ClipInference(db_collection=collection_name)

st.subheader("Semantic Search Demo")

if st.button("Clear vector DB"):
        state.clip_inference.clear_collection()
        st.success("ChromaDB collection cleared!")

if st.button("Store media embeddings from demo_data"):
    state.clip_inference.store_media_embeddings("demo_data")
    st.success("Media embeddings stored in ChromaDB!")

user_query_input = st.text_input("Enter some text:")

if st.button("Submit"):
    results = state.clip_inference.query_media(user_query_input)

    st.write("Query Results:")
    media_paths = results["documents"][0]
    cols = st.columns(len(media_paths))
    for idx, media_path in enumerate(media_paths):
        with cols[idx]:
            if any(media_path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
                st.image(media_path, width=300)
            elif any(media_path.lower().endswith(ext) for ext in ['.mp4', '.webm', '.avi']):
                st.video(media_path, width=300)

st.subheader("Image Classification Demo")

user_image_input = st.text_input("Enter image URL:")
if st.button("Classify Image"):
    try:
        class_name = state.clip_inference.classify_image(user_image_input)
        st.write(f"Predicted Class: :blue[{class_name}]")
        st.image(user_image_input, caption="Input Image")
    except Exception as e:
        st.error(f"Error processing image: {e}")

