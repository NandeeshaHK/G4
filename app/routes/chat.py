from fastapi import APIRouter
from app.multimodal.builder import build_message
from app.core.inference import generate_response

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat(req: dict):
    msg = req["messages"][-1]["content"]

    messages = build_message(msg)

    response = generate_response(messages)

    return {
        "id": "gemma4",
        "object": "chat.completion",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": response["text"]
                }
            }
        ]
    }