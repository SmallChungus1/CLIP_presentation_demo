import torch
from transformers import pipeline
import av
from PIL import Image
import os
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer
import chromadb
from dotenv import load_dotenv
import json
import requests
from io import BytesIO
import validators

load_dotenv()

class ClipInference():
    #keeping imagenet collection location hidden from the user for now
    def __init__(self, model_name="openai/clip-vit-base-patch32", db_collection=os.getenv("IMAGE_SEARCH_EMBEDDING_COLLECTION_NAME")):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.torch_dtype = torch.float16  # fast on CUDA
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.torch_dtype = torch.float32  # MPS prefers fp32
        else:
            self.device = torch.device("cpu")
            self.torch_dtype = torch.float32
        
        print(f"Using device: {self.device}")
        self.model = CLIPModel.from_pretrained(
            model_name,
            dtype=self.torch_dtype,
            low_cpu_mem_usage=False,  #avoiding the cannot copy out of meta tensor error?
            device_map=None,
            cache_dir=os.getenv("CACHE_DIR")
        )
        self.model.to(self.device)
        self.model.eval()

        self.processor = CLIPProcessor.from_pretrained(model_name, cache_dir=os.getenv("CACHE_DIR"))

        #chromadb client and collections for image search, imagenet-classification
        self.chromaDB_client_base_path = "./chroma_local_db"
        self.chroma_client = chromadb.PersistentClient(path=self.chromaDB_client_base_path)
        self.collection_name = db_collection
        self.chroma_image_search_collection = self.chroma_client.get_collection(name=db_collection) if db_collection in [col.name for col in self.chroma_client.list_collections()] else self.chroma_client.create_collection(name=db_collection)
        self.chroma_imagenet_collection = os.getenv("IMAGENET_EMBEDDING_COLLECTION_NAME")
        self.chroma_imagenet_embed_collec = self.chroma_client.get_collection(name=self.chroma_imagenet_collection) 

        print("init clip_inference")
    #Compute image/video embeddings and store them locally in chromadb
    def store_media_embeddings(self, media_dir):

        for filename in os.listdir(media_dir):
            file_path = os.path.join(media_dir, filename)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                features = self.encode_image(file_path).cpu().numpy()
            elif filename.lower().endswith(('.mp4', '.webm', '.avi')):
                #just like encoding images, but we sample frames and avg embeds
                features = self.encode_video(file_path).cpu().numpy()
            else:
                continue
            self.chroma_image_search_collection.add(
                ids=[filename],
                documents=[file_path],
                embeddings=[features],
                metadatas=[{"filename": filename}],
            )

        print(f"succesfully stored media embeddings to collection {self.collection_name}")

    def query_media(self, query_text):
        #need to be converted to numpy for chromadb
        text_features = self.encode_text(query_text).cpu().numpy()
        results = self.chroma_image_search_collection.query(
            query_embeddings=[text_features],
            n_results=5
        )
        return results
    
    def store_imagenet_class_embeddings(self, class_list_json):

        with open(class_list_json, "r") as f:
            class_list = json.load(f)

            for _, cls_name in class_list.items():
                class_prompt = f"a photo or video of a {cls_name}"
                class_embed = self.encode_text(class_prompt).cpu().numpy()
                self.chroma_imagenet_embed_collec.add(
                    ids=[cls_name],
                    embeddings=[class_embed])
        print(f"succesfully stored imagenet class embeddings to collection {self.chroma_imagenet_collection}")

    #Image classification function
    def classify_image(self, image_url):
        image_features = self.encode_image(image_url).cpu().numpy()
        result = self.chroma_imagenet_embed_collec.query(
            query_embeddings=[image_features],
            n_results=1
        )
        return result["ids"][0][0]

    def clear_collection(self):
        if self.collection_name in [col.name for col in self.chroma_client.list_collections()]:
            self.chroma_client.delete_collection(name=self.collection_name)
            print(f"Collection {self.collection_name} deleted.")
        else:
            print(f"Collection {self.collection_name} does not exist.")


    #Need l2 normalization, per clip's paper
    @staticmethod
    def _l2norm(x, dim=-1, eps=1e-12):
        return x / (x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps))
    
    @torch.inference_mode()
    def encode_image(self, image_url_or_obj):

        #we can pass in url or PIL image object
        if isinstance(image_url_or_obj, str):
            #if url
            if validators.url(image_url_or_obj):
                image = Image.open(BytesIO(requests.get(image_url_or_obj).content))
            else:
                image = Image.open(image_url_or_obj)
        else:
            image = image_url_or_obj.convert("RGB")

        image_inputs = self.processor(images=image, return_tensors="pt")
        #need to move tensors to correct device, since defaults to cpu
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
        image_features = self.model.get_image_features(**image_inputs)
        #its good to l2-normalize the embeddings
        image_features = self._l2norm(image_features.to(torch.float32)).squeeze(0)  # [D]

        print(image_features.shape)
        return image_features
    
    @torch.inference_mode()
    def encode_text(self, text):
        text_inputs = self.processor(text=text, return_tensors="pt")
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        text_features = self.model.get_text_features(**text_inputs)
        text_features = self._l2norm(text_features.to(torch.float32)).squeeze(0)
        return text_features

    @torch.inference_mode()
    def encode_video(self, video_url, max_frames=10):
        frames = self.extract_keyframes(video_url, max_frames=max_frames)
        video_features = []

        for frame in frames:
            frame_features = self.encode_image(frame)
            video_features.append(frame_features)

        #average embeddings
        video_feat_avg = torch.stack(video_features, dim=0).mean(dim=0, keepdim=True)
        video_feat_avg = self._l2norm(video_feat_avg).squeeze(0)
        return video_feat_avg

    #using PyAV for key-frame extraction
    def extract_keyframes(self, video_path, max_frames=10):
        container = av.open(video_path)
        frames = []
        for frame in container.decode(video=0):
            if frame.key_frame:  # keep only I-frames
                frames.append(Image.fromarray(frame.to_ndarray(format="rgb24")))
                if max_frames and len(frames) >= max_frames:
                    break
        container.close()

        #if no frames extracted using keyframe defaulting to uniform sampling
        if len(frames) == 0:
            frames = self.extract_uniform_frames(video_path, num_frames=max_frames)
        return frames

    #using av for uniform frame extraction
    def extract_uniform_frames(self, video_path, num_frames=10):
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        interval = max(total_frames // num_frames, 1)
        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if i % interval == 0:
                frames.append(Image.fromarray(frame.to_ndarray(format="rgb24")))
                if len(frames) >= num_frames:
                    break
        container.close()
        return frames
    
