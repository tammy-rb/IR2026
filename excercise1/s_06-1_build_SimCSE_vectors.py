from s_06_build_sentence_vectors import build_chunked_embeddings_for_xml

RAW_DIR = "docs"
SIMCSE_OUT_DIR = "vectors/simcse_raw"
SIMCSE_MODEL_NAME = "princeton-nlp/unsup-simcse-bert-base-uncased"


def main():
    """
    Build SimCSE document embeddings for the raw XML debates.
    Uses the generic chunked XML pipeline from embeddings_base.
    """
    build_chunked_embeddings_for_xml(
        raw_dir=RAW_DIR,
        model_name=SIMCSE_MODEL_NAME,
        out_dir=SIMCSE_OUT_DIR,
        xml_pattern="*.xml",
        chunk_max_tokens=256,
        batch_size=32,  
        prefix="simcse_doc_embeddings",
    )


if __name__ == "__main__":
    main()
