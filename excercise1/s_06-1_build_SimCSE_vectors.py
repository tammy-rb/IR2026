from s_06_encode_chuncks import build_embeddings_from_saved_chunks

CHUNKS_DIR = "docs_chuncks"
SIMCSE_OUT_DIR = "vectors/simcse_raw"
SIMCSE_MODEL_NAME = "princeton-nlp/unsup-simcse-bert-base-uncased"


def main():
    """
    Build SimCSE document embeddings using precomputed chunks.
    """
    build_embeddings_from_saved_chunks(
        chunks_dir=CHUNKS_DIR,
        model_name=SIMCSE_MODEL_NAME,
        out_dir=SIMCSE_OUT_DIR,
        batch_size=16,
        max_chunks_per_call=32,
        prefix="simcse_doc_embeddings",
    )


if __name__ == "__main__":
    main()
