from pydantic import BaseModel, Field, EmailStr, field_validator, field_serializer, ConfigDict
from typing import Optional, List, Dict, Any

class TokenClassificationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

# class TokenClassificationResponse(BaseModel):
#     text: str
#     tokens: List[str]
#     labels: List[str]
#     scores: List[float]
#     start: List[int]
#     end: List[int]
#     model_config = ConfigDict(from_attributes=True)
    

class Entity(BaseModel):
    entity_group: str
    score: float
    word: str
    start: int
    end: int
