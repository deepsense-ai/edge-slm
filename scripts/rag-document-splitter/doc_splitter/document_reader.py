from pathlib import Path
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader


class DocumentReader:
    """Class that is responsible to load all the documents from the specified location."""
    def __init__(self, path: Path):
        if path.is_file():
            self._document_paths = [path]
        else:
            self._document_paths = [filepath for filepath in list(path.glob("**/*")) if filepath.is_file()]

    @staticmethod
    def _load_single_document(path: Path) -> Document:
        if path.suffix in {".md", ".txt"}:
            return TextLoader(path).load()[0]
        raise ValueError(f"Not supported extension. File path {path}")

    def load(self) -> List[Document]:
        """Loads the documents and returns them as LangChain documents.

        Returns:
            List[Document]: All the loaded documents in LangChain format.
        """
        documents = [self._load_single_document(path) for path in self._document_paths]
        return documents
