import streamlit as st
from clip_inference import ClipInference

st.title("CLIP Demo on Classification and Semantic Search")

clip_inference = ClipInference()

if st.button("Clear vector DB"):
        clip_inference.clear_collection("test_collection")
        st.success("ChromaDB collection cleared!")

if st.button("Store media embeddings from demo_data"):
    clip_inference.store_media_embeddings("demo_data", collection_name="test_collection")
    st.success("Media embeddings stored in ChromaDB!")

user_input = st.text_input("Enter some text:")

if st.button("Submit"):
    results = clip_inference.query_media(user_input, collection_name="test_collection")
    st.write("Query Results:")
    st.write(results)