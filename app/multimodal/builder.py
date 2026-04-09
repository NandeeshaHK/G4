def build_message(data):
    content = []

    # ORDER MATTERS → media FIRST (HF rule)
    
    if data["type"] == "audio":
        content.append({
            "type": "audio",
            "audio": data["content"]  # URL or path
        })

    elif data["type"] == "image":
        content.append({
            "type": "image",
            "url": data["content"]
        })

    elif data["type"] == "video":
        content.append({
            "type": "video",
            "video": data["content"]
        })

    # Always append text LAST
    content.append({
        "type": "text",
        "text": data.get("prompt", "")
    })

    return [
        {
            "role": "user",
            "content": content
        }
    ]