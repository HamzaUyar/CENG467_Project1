import torch


def predict_model(model, tokenizer, messages, configuration=None):
    # Configuration defaults
    cfg = configuration or {}
    temperature = float(cfg.get("temperature", 0.1))
    max_new_tokens = int(cfg.get("max_token_limit", 2000))
    do_sample = temperature > 0.0

    # Prepare chat prompt using tokenizer's chat template (Qwen supports this)
    if hasattr(tokenizer, "apply_chat_template"):
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
    else:
        # Fallback: simple concatenation if no chat template is available
        rendered = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            rendered.append(f"<{role}>: {content}")
        prompt = "\n".join(rendered) + "\n<assistant>:"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Device placement
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    # Generation settings
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": do_sample,
    }

    # Ensure valid padding/eos ids where applicable
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        gen_kwargs["pad_token_id"] = tokenizer.eos_token_id
    if getattr(tokenizer, "eos_token_id", None) is not None:
        gen_kwargs["eos_token_id"] = tokenizer.eos_token_id

    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            **gen_kwargs,
        )

    # Only decode the newly generated tokens (exclude the prompt)
    gen_only = output_ids[0, input_ids.shape[-1]:]
    text = tokenizer.decode(gen_only, skip_special_tokens=True)
    return text.strip()
    """
    This function, `predict_model`, is designed to interact with QWEN models to generate predictions
    based on a conversation history. 

    Args:
        model: The pre-trained language model to be used for generating responses.
        tokenizer: the tokenizer corresponding to the model.
        messages: A list of dictionaries representing the conversation history,
                  where each dictionary has a "role" (e.g., "system", "user", or "assistant") 
                  and "content" (the message text).
        configuration: initially, the model used should be max_token_limit of 2000, with temperature of 0.1
        The assessment would mainly be assessed the correctness of the implementation, rather than the performance

    Returns:
        The model's response as a string.
    """


def model_evaluation(model_type, model, tokenizer, system_content, question, formatted_options, configuration=None):
    if model_type == "qwen2" or model_type == "qwen3":
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Question: {question}\n\nOptions:\n{formatted_options}"}
        ]
        model_result = predict_model(model, tokenizer, messages, configuration)
    else: 
        raise ValueError(f"Unknown model_type: {model_type}")

    #  print(f"Model result: {model_result}")
    return model_result
