#
strategies of fitering by weeks:
bm25:
[text](RAG_retriever/prefilter/chuncks_selector.py)
get a filter function and chunckl file or list of chuncks.
get chucnkl and retun chunckd ids or vector of rows (1 for matching and 0 removed) of chuncks passed the filter, 
so we can score only passed chuncks.
dense/openai
save in qdrant.
docker compose up the qdrant, and sae with ..
so we can prefilter by ..
built in in qdirant - good for filtering.

week_retrieving:

please update the retiever so becuase there is no text in the payload, it will make the chunck bu take it from the jsonl.. go to jsonl, take the fields from it... so first: make a script that take the jsonl, and make a map - any uuid -> jsonl file path, byte_offset. so after retrieve the dense vectors ids - go in o(1) to the map and retrieve their fields to the Chucnk model (including texts).
1. make a .py script that take a jsonl file and make map id->jsonlfile, byte_offset.
2. dense retrieval will go into this map after vectors retirving and use it to implement the chuncks fileds in o(1) time.


