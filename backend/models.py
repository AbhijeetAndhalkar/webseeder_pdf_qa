from pydantic import BaseModel, Field
from typing import List

class AskRequest(BaseModel):
    document_id: str = Field(..., description="The unique ID of the uploaded document.")
    question: str = Field(..., description="The user's question regarding the document.")

class AskResponse(BaseModel):
    answer: str = Field(..., description="The generated answer based strictly on the document context.")
    sources: List[int] = Field(..., description="A list of source chunk numbers used to generate the answer.")
