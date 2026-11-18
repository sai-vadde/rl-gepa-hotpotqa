"""Local Qwen3 through Ollama. One request at a time, thinking off."""
import os

import dspy

API_BASE = os.environ.get("OLLAMA_HOST_URL", "htt<path>")


def _patch_litellm_think():
    """litellm 1.77 puts an unknown `think` kwarg inside Ollama `options`, where
    Ollama ignores it. Move it to the top level of the /api/chat body so that
    think=False really switches Qwen3 thinking off."""
    from litellm.llms.ollama.chat.transformation import OllamaChatConfig
    if getattr(OllamaChatConfig, "_think_patched", False):
        return
    orig = OllamaChatConfig.transform_request

    def patched(self, model, messages, optional_params, litellm_params, headers):
        think = optional_params.pop("think", None)
        data = orig(self, model, messages, optional_params, litellm_params, headers)
        if think is not None:
            data["think"] = think
        return data

    OllamaChatConfig.transform_request = patched
    OllamaChatConfig._think_patched = True


def make_lm(model="qwen3:4b", max_tokens=256, temperature=0.6, cache=False):
    _patch_litellm_think()
    return dspy.LM(
        f"ollama_chat/{model}",
        api_base=API_BASE,
        api_key="",
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
        cache=cache,
        num_ctx=12288,
        top_k=20,
        think=False,
    )


def usage_totals(lm, start=0):
    """Sum model calls and tokens from lm.history entries after index `start`."""
    calls = pt = ct = 0
    for h in lm.history[start:]:
        calls += 1
        u = h.get("usage") or {}
        pt += u.get("prompt_tokens", 0) or 0
        ct += u.get("completion_tokens", 0) or 0
    return {"calls": calls, "prompt_tokens": pt, "completion_tokens": ct}
