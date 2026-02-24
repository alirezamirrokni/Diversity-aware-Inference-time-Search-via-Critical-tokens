from typing import Dict, List, Tuple

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def detect_flags(model_name: str) -> Tuple[bool, bool, bool]:
    # Detect whether a model name indicates Gemma, Qwen, or Phi.
    s = (model_name or "").lower()
    is_gemma = "gemma" in s
    is_qwen = "qwen" in s
    is_phi = ("phi-3" in s) or ("phi3" in s) or ("/phi" in s)
    return is_gemma, is_qwen, is_phi


def load_tokenizer_and_model(model_name: str):
    # Load tokenizer + model, with Qwen tie-embedding workaround and Phi trust_remote_code.
    is_gemma, is_qwen, is_phi = detect_flags(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=is_phi)

    # Ensure a pad token exists (some chat tokenizers omit it; generation can warn/error otherwise).
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Always load config for Qwen and Phi so we can apply compatibility fixes.
    config = None
    if is_qwen or is_phi:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=is_phi)

        # Qwen: avoid tied embeddings issues in some versions.
        if is_qwen:
            config.tie_word_embeddings = False

        # Phi-3 (remote code / older transformers): some configs store RoPE scaling under
        # "rope_type" instead of "type". Remote modeling code expects rope_scaling["type"].
        if is_phi and getattr(config, "rope_scaling", None) is not None:
            rs = config.rope_scaling
            if isinstance(rs, dict) and "type" not in rs:
                patched = dict(rs)
                patched["type"] = patched.get("rope_type", "longrope")
                config.rope_scaling = patched

    if config is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=is_phi,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=is_phi,
        )

    model.eval()
    return tokenizer, model


def build_chat_text(tokenizer, messages: List[Dict[str, str]]) -> str:
    # Build a chat prompt using tokenizer.apply_chat_template if available, else fallback.
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    parts = []
    for m in messages:
        role = (m.get("role") or "user").upper()
        content = (m.get("content") or "").strip()
        parts.append(f"{role}: {content}")
    parts.append("ASSISTANT:")
    return "\n".join(parts)
