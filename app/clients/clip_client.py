from sentence_transformers import SentenceTransformer

# One model, two encoders under the hood — the same .encode() call handles
# both text strings and PIL images, landing both in the same 512-dim space.
clip_model = SentenceTransformer("clip-ViT-B-32")
