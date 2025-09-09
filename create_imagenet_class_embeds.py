from clip_inference import ClipInference
import torch
import chromadb
import json
from PIL import Image
import os 
from dotenv import load_dotenv
load_dotenv()


def create_imagenet_class_embeddings(clear_existing=True):
    collection_name = os.getenv("IMAGENET_EMBEDDING_COLLECTION_NAME")
    clip_inference = ClipInference(db_collection=collection_name)

    if clear_existing:
        clip_inference.clear_imagenet_collection()

    clip_inference.store_imagenet_class_embeddings("imageNet_classes.json")


if __name__ == "__main__": 
    create_imagenet_class_embeddings()   