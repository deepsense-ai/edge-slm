from pydantic import BaseModel


class DocumentContextMetadata(BaseModel):
    """Model representing Document metadata"""

    source: str
    chunk_id: int


class DocumentContext(BaseModel):
    """Model representing Document context - content and metadata"""

    metadata: DocumentContextMetadata
    content: str
