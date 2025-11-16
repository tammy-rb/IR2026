from s_06_encode_chuncks import build_embeddings_from_saved_chunks

CHUNKS_DIR = "docs_chuncks"
SBERT_OUT_DIR = "vectors/SBERT_origin"
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    """
    Build SBERT document embeddings using precomputed chunks.
    """
    build_embeddings_from_saved_chunks(
        chunks_dir=CHUNKS_DIR,
        model_name=SBERT_MODEL_NAME,
        out_dir=SBERT_OUT_DIR,
        batch_size=16,
        max_chunks_per_call=32,
        prefix="sbert_doc_embeddings",
    )


if __name__ == "__main__":
    main()
