"""Request and response models.

These are the contract the UI codes against, and FastAPI generates the OpenAPI
spec from them, so the documentation cannot drift from what is enforced.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=512, gt=0, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    @field_validator("messages")
    @classmethod
    def _reject_assistant_only(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        # A conversation with nothing from the user has nothing to answer.
        if all(m.role == "assistant" for m in v):
            raise ValueError("conversation must contain at least one user or system message")
        return v


class ErrorBody(BaseModel):
    """Structured error payload.

    The UI renders `detail` directly, so it must stay human-readable — this is
    what a user sees in the error banner.
    """

    error: str
    detail: str
    request_id: str
