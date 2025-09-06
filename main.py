import streamlit as st
from clip_inference import ClipInference
import torch
import os 
from dotenv import load_dotenv

load_dotenv()

st.title("CLIP Demo on Classification and Semantic Search")
collection_name=os.getenv("IMAGE_SEARCH_EMBEDDING_COLLECTION_NAME")

clip_inference = ClipInference(db_collection=collection_name)

st.subheader("Semantic Search Demo")

if st.button("Clear vector DB"):
        clip_inference.clear_collection()
        st.success("ChromaDB collection cleared!")

if st.button("Store media embeddings from demo_data"):
    clip_inference.store_media_embeddings("demo_data")
    st.success("Media embeddings stored in ChromaDB!")

user_query_input = st.text_input("Enter some text:")

if st.button("Submit"):
    results = clip_inference.query_media(user_query_input)
    st.write("Query Results:")
    st.write(results)

st.subheader("Image Classification Demo")

user_image_input = st.text_input("Enter image URL:")
if st.button("Classify Image"):
    try:
        class_name = clip_inference.classify_image(user_image_input)
        st.write(f"Predicted Class: {class_name}")
        st.image(user_image_input, caption="Input Image", use_container_width=True)
    except Exception as e:
        st.error(f"Error processing image: {e}")

