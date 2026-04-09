from app.core.model import model, processor


def generate_response(messages):
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=512
    )

    decoded = processor.decode(
        outputs[0][input_len:], 
        skip_special_tokens=False
    )

    return processor.parse_response(decoded)