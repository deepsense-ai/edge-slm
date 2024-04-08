# Document splitter

This portion of the repository contains the sources of a simple script used to prepare the chunked document database. The documents which are supported are `.md` markdown formatted elements and a simple `.txt` files.

## Usage

```
usage: Document Splitter [-h] [--output OUTPUT] [--chunk_size CHUNK_SIZE] [--chunk_overlap CHUNK_OVERLAP] [--size_count_method {CHARACTERS,TOKENS}]
                         [--embedding_model_name EMBEDDING_MODEL_NAME]
                         input

A script to run the chunking process for the given documents.

positional arguments:
  input                 Path to the input location of the files. Can be pointing at directory or at a single file.

options:
  -h, --help            show this help message and exit
  --output OUTPUT       Path to the output location of the document chunks. Stored in JSON format
  --chunk_size CHUNK_SIZE
                        Maximum length of single chunk
  --chunk_overlap CHUNK_OVERLAP
                        Chunk overlap
  --size_count_method {CHARACTERS,TOKENS}
  --embedding_model_name EMBEDDING_MODEL_NAME
```

The output file can be directly fed to demo application for embedding calculation and building the indexed knowledge base.
