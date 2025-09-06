import torch
from transformers import pipeline
import av
from PIL import Image
import os
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer
import chromadb

class ClipInference():
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.torch_dtype = torch.float16  # fast on CUDA
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.torch_dtype = torch.float32  # MPS prefers fp32
        else:
            self.device = torch.device("cpu")
            self.torch_dtype = torch.float32

        self.model = CLIPModel.from_pretrained(model_name).to(self.device, dtype=self.torch_dtype)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    #Compute image/video embeddings and store them locally in chromadb
    def store_media_embeddings(self, media_dir, collection_name="test_collection"):
        client = chromadb.Client()
        if collection_name in [col.name for col in client.list_collections()]:
            collection = client.get_collection(name=collection_name)
        else:
            collection = client.create_collection(name=collection_name)

        for filename in os.listdir(media_dir):
            file_path = os.path.join(media_dir, filename)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                features = self.encode_image(file_path).cpu().numpy()
            elif filename.lower().endswith(('.mp4', '.webm', '.avi')):
                #just like encoding images, but we sample frames and avg embeds
                features = self.encode_video(file_path).cpu().numpy()
            else:
                continue
            collection.add(
                ids=[filename],
                documents=[file_path],
                embeddings=[features],
                metadatas=[{"filename": filename}],
            )

        print(f"succesfully stored media embeddings to collection {collection_name}")

    def clear_collection(self, collection_name="test_collection"):
        client = chromadb.Client()
        if collection_name in [col.name for col in client.list_collections()]:
            client.delete_collection(name=collection_name)
            print(f"Collection {collection_name} deleted.")
        else:
            print(f"Collection {collection_name} does not exist.")

    def query_media(self, query_text, collection_name="test_collection"):
        text_features = self.encode_text(query_text).cpu().numpy()
        client = chromadb.Client()
        collection = client.get_collection(name=collection_name)
        results = collection.query(
            query_embeddings=[text_features],
            n_results=5
        )
        return results
    
    #Need l2 normalization, per clip's paper
    @staticmethod
    def _l2norm(x, dim=-1, eps=1e-12):
        return x / (x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps))
    
    @torch.inference_mode()
    def encode_image(self, image_url_or_obj):

        #we can pass in url or PIL image object
        if isinstance(image_url_or_obj, str):
            image = Image.open(image_url_or_obj).convert("RGB")
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
    
