import streamlit as st
from clip_inference import ClipInference

st.title("CLIP Demo on Classification and Semantic Search")

clip_inference = ClipInference()

user_input = st.text_input("Enter some text:")
if st.button("Submit"):
    text_features = clip_inference.encode_text(user_input)
    st.write(f"Text features: {text_features.shape}") 
    image_features = clip_inference.encode_image("demo_data/boat1.png")
    st.write(f"Image features: {image_features.shape}")
    video_features = clip_inference.encode_video("demo_data/ship1.webm", max_frames=5)
    st.write(f"Video features: {video_features.shape}")