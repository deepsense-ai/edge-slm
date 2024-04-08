import itertools
from pathlib import Path
from typing import Any, Callable, List

from langchain.docstore.document import Document
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

from doc_splitter.document_context import DocumentContext, DocumentContextMetadata



def split_document_into_chunks(
    document: Document,
    chunk_size: int = 1000,
    chunk_overlap: int = 0,
    length_function: Callable = len,
    md_separators: bool = True,
) -> list[Document]:
    """Splits a LangChain Document into chunks of size `chunk_size` with `chunk_overlap`
    overlap.

    Args:
        document: LangChain Document.
        chunk_size: Maximum size of chunks.
        chunk_overlap: Number of characters that subsequent chunks share.
        length_function: Function used by RecursiveCharacterTextSplitter to get the length of chunk.
        md_separators: If false, disables using markdown specific separators

    Returns:
        List of LangChain Documents.
    """

    if (Path(document.metadata["source"]).suffix == ".md") & md_separators:
        text_splitter = RecursiveCharacterTextSplitter.from_language(
            Language.MARKDOWN,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
        )
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
        )

    document_chunks = text_splitter.split_documents([document])

    return document_chunks


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 0,
    length_function: Callable = len,
    md_separators: bool = True,
) -> List[DocumentContext]:
    """
    Splits LangChain documents into chunks and convert them into DocumentContext.

    Args:
        documents (List[Document]): List of LangChain documents to split
        chunk_size (int, optional): The maximal chunk size. Defaults to 1000.
        chunk_overlap (int, optional): Number of characters that subsequent chunks share. Defaults to 0.
        length_function (Callable, optional): Function used by RecursiveCharacterTextSplitter to get
                                              the length of chunk.. Defaults to len.
        md_separators (bool, optional): If false, disables using markdown specific separators. Defaults to True.

    Returns:
        List[DocumentContext]: List of DocumentContext objects.
    """
    chunk_id = 0
    langchain_documents = list(itertools.chain.from_iterable(
        [
            split_document_into_chunks(
                document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=length_function,
                md_separators=md_separators,
            )
            for document in documents
        ]
    ))

    for document_chunk in langchain_documents:
        document_chunk.metadata["chunk_id"] = chunk_id
        chunk_id += 1

    document_contexts = [
        DocumentContext(
            metadata=DocumentContextMetadata(
                source=document.metadata["source"].as_posix(),
                chunk_id=document.metadata["chunk_id"],
            ),
            content=document.page_content,
        )
        for document in langchain_documents
    ]
    return document_contexts
