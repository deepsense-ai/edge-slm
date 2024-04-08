import argparse
import json
from pathlib import Path
from enum import Enum

from typing import Callable

from pydantic import BaseModel

from doc_splitter.document_reader import DocumentReader
from doc_splitter.splitter import split_documents
from doc_splitter.utils import TokensLengthCalculator


class SizeCountMethod(Enum):
    """Indicates the method of choice for calculating chunk size."""
    CHARACTERS="CHARACTERS"
    TOKENS="TOKENS"


class Configuration(BaseModel):
    """Holds the script run configuration."""
    input: Path
    output: Path
    embedding_model_name: str
    chunk_size: int
    chunk_overlap: int
    size_count_method: SizeCountMethod


def parse_input_configuration() -> Configuration:
    """
    Function parses input arguments.

    Returns:
        Configuration: parsed configuration
    """
    parser = argparse.ArgumentParser(
        prog="Document Splitter",
        description="A script to run the chunking process for the given documents.",
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Path to the input location of the files. Can be pointing at directory or at a single file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="document_chunks.json",
        help="Path to the output location of the document chunks. Stored in JSON format",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="Maximum length of single chunk",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=0,
        help="Chunk overlap",
    )
    parser.add_argument(
        "--size_count_method",
        type=str,
        choices=["CHARACTERS", "TOKENS"],
        default="CHARACTERS"
    )
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="thenlper/gte-base"
    )


    args = parser.parse_args()
    return Configuration(
        input=args.input,
        output=args.output,
        embedding_model_name=args.embedding_model_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        size_count_method=SizeCountMethod(args.size_count_method)
    )


def get_length_calculator_fn(config: Configuration) -> Callable:
    """A function that returns the length calculator callable based on configuration.

    Args:
        config (Configuration): A script configuration.

    Returns:
        Callable: A function that calculates the length of the chunk
    """
    method = config.size_count_method
    if method == SizeCountMethod.CHARACTERS:
        return len
    if method == SizeCountMethod.TOKENS:
        return TokensLengthCalculator(config.embedding_model_name)
    raise ValueError(f"Size count method: {method} is not supported")


if __name__ == "__main__":
    configuration = parse_input_configuration()
    reader = DocumentReader(configuration.input)
    documents = reader.load()
    chunks = split_documents(
        documents,
        chunk_size=configuration.chunk_size,
        chunk_overlap=configuration.chunk_overlap,
        length_function = get_length_calculator_fn(configuration),
        md_separators = True)

    with open(configuration.output, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks}, f, default=lambda x: x.dict())
