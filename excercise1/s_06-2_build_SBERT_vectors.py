from s_06_build_sentence_vectors import build_chunked_embeddings_for_xml

RAW_DIR = "docs"
SBERT_OUT_DIR = "vectors/SBERT_origin"
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    """
    Build SBERT document embeddings for the SAME raw XML debates,
    using the same chunking strategy as SimCSE.
    """
    build_chunked_embeddings_for_xml(
        raw_dir=RAW_DIR,
        model_name=SBERT_MODEL_NAME,
        out_dir=SBERT_OUT_DIR,
        xml_pattern="*.xml",
        chunk_max_tokens=256,
        batch_size=32,
        prefix="sbert_doc_embeddings",
    )


if __name__ == "__main__":
    main()
