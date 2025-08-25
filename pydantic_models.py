from pydantic import BaseModel, Field
from typing import Union
from enum import Enum
from datetime import datetime, timezone
import typing

class ModelName(str, Enum):
    GEMINI_FLASH = "gemini-2.0-flash"
    GEMINI_FLASH_MINI = "gemini-2.0-flash-lite"

class QueryInput(BaseModel):
    question: str
    session_id: Union[str, int] = Field(default=None)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model: ModelName = Field(default=ModelName.GEMINI_FLASH)

class QueryResponse(BaseModel):
    answer: str
    session_id: Union[str, int]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model: ModelName

class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime
    file_size: typing.Optional[int] = None
    content_type: typing.Optional[str] = None

class DeleteFileRequest(BaseModel):
    file_id: int

class DeleteFileResponse(BaseModel):
    message: str
