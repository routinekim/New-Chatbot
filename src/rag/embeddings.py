import os


def get_embeddings():
    if os.getenv("UPSTAGE_API_KEY"):
        from langchain_upstage import UpstageEmbeddings
        return UpstageEmbeddings(model="solar-embedding-1-large")
    # 폴백: Upstage API 키 없을 때 HuggingFace bge-m3-korean (로컬, 무료)
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="upskyy/bge-m3-korean")
