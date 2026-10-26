import os
import json
import time
import requests
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# ============================================================
# Configuration Loading
# ============================================================

def load_config():
    """Load configuration from config.json if present."""
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", "config.json"),
        os.path.join(os.getcwd(), "config.json"),
        "config.json"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return {}

_CONFIG = load_config()

# Environment / Config Defaults
VLLM_BASE_URL = os.environ.get(
    "VLLM_BASE_URL",
    _CONFIG.get("VLLM_BASE_URL", os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1"))
)
VLLM_API_KEY = os.environ.get(
    "VLLM_API_KEY",
    _CONFIG.get("VLLM_API_KEY", os.environ.get("LLM_API_KEY", "local-key"))
)
DEFAULT_VLLM_MODEL = os.environ.get(
    "VLLM_MODEL",
    _CONFIG.get("VLLM_MODEL", os.environ.get("LLM_MODEL", "Qwen/Qwen3-32B"))
)

OLLAMA_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL",
    _CONFIG.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
)
OLLAMA_API_KEY = os.environ.get(
    "OLLAMA_API_KEY",
    _CONFIG.get("OLLAMA_API_KEY", "ollama")
)
DEFAULT_OLLAMA_MODEL = os.environ.get(
    "OLLAMA_MODEL",
    _CONFIG.get("OLLAMA_MODEL", "qwen2.5:1.5b")
)

OPENAI_API_KEY = os.environ.get(
    "OPENAI_API_KEY",
    _CONFIG.get("OPENAI_API_KEY", "")
)
OPENAI_BASE_URL = os.environ.get(
    "OPENAI_BASE_URL",
    _CONFIG.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
)


def resolve_backend(backend=None, logprobs=False, model=None):
    """
    Resolve target backend, base_url, api_key, and default model.
    - 'vllm': High-performance OpenAI-compatible serving with token logprobs (MCTS requirement).
    - 'ollama': Local lightweight serving for Symbolic LLM goal decomposition.
    - 'openai': Official OpenAI API.
    - 'auto' / None: If logprobs requested -> vLLM; otherwise Ollama if available or vLLM.
    """
    if backend is None or backend == "auto":
        if logprobs:
            backend = "vllm"
        else:
            backend = "ollama"

    backend = backend.lower()

    if backend == "vllm":
        base_url = VLLM_BASE_URL
        api_key = VLLM_API_KEY
        if model and model != DEFAULT_OLLAMA_MODEL:
            resolved_model = model
        else:
            resolved_model = DEFAULT_VLLM_MODEL
    elif backend == "ollama":
        base_url = OLLAMA_BASE_URL
        api_key = OLLAMA_API_KEY
        if model and model != DEFAULT_VLLM_MODEL:
            resolved_model = model
        else:
            resolved_model = DEFAULT_OLLAMA_MODEL
    elif backend == "openai":
        base_url = OPENAI_BASE_URL
        api_key = OPENAI_API_KEY
        resolved_model = model or "gpt-4o"
    else:
        # Fallback to general LLM configuration
        base_url = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
        api_key = os.environ.get("LLM_API_KEY", "local-key")
        resolved_model = model or DEFAULT_VLLM_MODEL

    return backend, base_url, api_key, resolved_model


# ============================================================
# Chat Completion API
# ============================================================

def get_session_completion(
    chat_history,
    model=None,
    max_tokens=4095,
    temperature=0.0,
    logprobs=False,
    top_logprobs=None,
    backend=None,
    base_url=None,
    api_key=None,
    timeout=300
):
    """
    Send chat completion request to the chosen LLM backend (vLLM, Ollama, or OpenAI).
    
    For MCTS, vLLM should be used with logprobs=True to capture per-token sequence probabilities.
    For Symbolic LLM, Ollama is used for fast subgoal extraction without requiring logprobs.
    """
    resolved_backend, default_url, default_key, resolved_model = resolve_backend(
        backend=backend, logprobs=logprobs, model=model
    )

    url = (base_url or default_url).rstrip("/")
    key = api_key or default_key

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }

    def fetch_completion(messages):
        use_logprobs = logprobs
        payload = {
            "model": resolved_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if use_logprobs:
            payload["logprobs"] = True
            if top_logprobs is not None:
                payload["top_logprobs"] = top_logprobs

        endpoint = f"{url}/chat/completions"
        max_retries = 8
        attempt = 0
        while attempt < max_retries:
            try:
                response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                # If logprobs caused an error on a backend that doesn't support it, remove and retry immediately
                if use_logprobs and hasattr(e, "response") and e.response is not None and e.response.status_code in [400, 422]:
                    payload.pop("logprobs", None)
                    payload.pop("top_logprobs", None)
                    use_logprobs = False
                    continue

                # Handle rate limits (HTTP 429)
                if hasattr(e, "response") and e.response is not None and e.response.status_code == 429:
                    attempt += 1
                    if attempt < max_retries:
                        # Check for Retry-After header
                        retry_after = e.response.headers.get("Retry-After")
                        if retry_after and retry_after.isdigit():
                            sleep_time = int(retry_after) + 1
                        else:
                            sleep_time = min(30, (2 ** attempt) * 2)
                        print(f"[Rate limit 429] Waiting {sleep_time}s for API cooldown (attempt {attempt}/{max_retries})...", flush=True)
                        time.sleep(sleep_time)
                        continue

                raise RuntimeError(f"HTTP request to {endpoint} failed (backend={resolved_backend}): {e}") from e

    result = fetch_completion(chat_history)

    # Handle token limit continuation if output was truncated
    if result.get("choices") and result["choices"][0].get("finish_reason") == "length":
        assistant_content = result["choices"][0]["message"].get("content", "")
        chat_history.append({
            "role": "assistant",
            "content": assistant_content
        })
        chat_history.append({
            "role": "user",
            "content": "Continue from exactly where you stopped. Output only PDDL plan actions."
        })

        next_result = fetch_completion(chat_history)
        next_content = next_result["choices"][0]["message"].get("content", "")

        result["choices"][0]["message"]["content"] = assistant_content + "\n" + next_content

        # Combine logprobs if present
        if "logprobs" in result["choices"][0] and result["choices"][0]["logprobs"]:
            if "logprobs" in next_result["choices"][0] and next_result["choices"][0]["logprobs"]:
                current_lp_content = result["choices"][0]["logprobs"].get("content", [])
                next_lp_content = next_result["choices"][0]["logprobs"].get("content", [])
                if current_lp_content and next_lp_content:
                    current_lp_content.append({'token': '\n', 'logprob': -3.0e-05, 'bytes': [10], 'top_logprobs': []})
                    current_lp_content.extend(next_lp_content)

    return result


# ============================================================
# Action Sequence Probability Calculation (seq_prob)
# ============================================================

def seq_prob(completion):
    """
    Calculate sequence probabilities for generated PDDL action lines.
    
    Given the model completion with token log probabilities:
      p(action_line) = exp( sum_{t in action_line} log p(t) )
      
    Returns a list of tuples: [(action_string, probability), ...]
    """
    if "choices" not in completion or not completion["choices"]:
        print("Invalid completion structure:", json.dumps(completion, indent=2))
        raise KeyError("The completion response does not contain valid 'choices' key")

    response_text = completion["choices"][0]["message"].get("content", "").strip()
    logprobs_data = completion["choices"][0].get("logprobs")

    # Split into clean action lines (ignoring comments and empty lines)
    action_lines = []
    for line in response_text.split("\n"):
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        action_lines.append(line)

    if not action_lines:
        return []

    # If logprobs are not available (e.g. backend lacks logprob support), return fallback uniform weights
    if not logprobs_data or not logprobs_data.get("content"):
        print("[Notice] Logprobs not present in completion. Using uniform action probability 1.0.")
        return [(action, 1.0) for action in action_lines]

    logprobs_content = logprobs_data["content"]
    token_logprobs = [token["logprob"] for token in logprobs_content]
    tokens = [token["token"] for token in logprobs_content]

    action_logprobs = []
    current_action_logprob = 0.0
    action_idx = 0

    for token, logprob in zip(tokens, token_logprobs):
        current_action_logprob += logprob

        # A newline inside the token or a line delimiter indicates the end of an action line
        if "\n" in token or token.endswith(",") or token.endswith("\r"):
            if action_idx < len(action_lines):
                action_logprobs.append(current_action_logprob)
                current_action_logprob = 0.0
                action_idx += 1

    # Add remaining logprob for the last action line if present
    if current_action_logprob != 0.0 and action_idx < len(action_lines):
        action_logprobs.append(current_action_logprob)

    # Convert log-probabilities to likelihoods: p = exp(sum log p(t))
    action_probabilities = [float(np.exp(lp)) for lp in action_logprobs]

    # Align length with action_lines
    if len(action_probabilities) > len(action_lines):
        action_probabilities = action_probabilities[:len(action_lines)]

    while len(action_probabilities) < len(action_lines):
        action_probabilities.append(1.0)

    return list(zip(action_lines, action_probabilities))