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
    if user_image_input:
        image_features = clip_inference.encode_image(user_image_input)
        categories = ["a photo of a cat", "a photo of a dog", "a photo of a bird", "a photo of a car", "a photo of a tree"]
        category_features = [clip_inference.encode_text(cat) for cat in categories]
        category_features = torch.stack(category_features)
        image_features = image_features.unsqueeze(0)
        similarities = (image_features @ category_features.T).squeeze(0)
        best_idx = similarities.argmax().item()
        st.write(f"The image is classified as: {categories[best_idx]}")
    else:
        st.write("Please enter a valid image URL.")
