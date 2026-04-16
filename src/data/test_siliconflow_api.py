from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GENERATOR_SCRIPT_PATH = PROJECT_ROOT / "src" / "data" / "generate_synthetic_low_count_cases.py"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "Pro/deepseek-ai/DeepSeek-V3.2"
API_KEY_PLACEHOLDER = "PASTE_YOUR_SILICONFLOW_API_KEY_HERE"
DEFAULT_API_KEY = API_KEY_PLACEHOLDER
DEFAULT_API_KEY_ENV = "SILICONFLOW_API_KEY"
DEFAULT_TIMEOUT_SEC = 60
MAX_BODY_PREVIEW_CHARS = 1200


def _mask_secret(secret: str) -> str:
    if len(secret) <= 10:
        return "*" * len(secret)
    return f"{secret[:6]}...{secret[-4:]}"


def _preview_text(text: str) -> str:
    text = text.strip()
    if len(text) <= MAX_BODY_PREVIEW_CHARS:
        return text
    return text[:MAX_BODY_PREVIEW_CHARS] + "\n...[truncated]..."


def _normalize_base_url(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base[: -len("/chat/completions")]
    if base.endswith("/models"):
        return base[: -len("/models")]
    return base


def _load_api_key_from_generator(script_path: Path) -> str:
    if not script_path.exists():
        return ""
    text = script_path.read_text(encoding="utf-8")
    match = re.search(r'^DEFAULT_API_KEY\s*=\s*["\'](.*)["\']\s*$', text, re.MULTILINE)
    if not match:
        return ""
    api_key = match.group(1).strip()
    if not api_key or api_key == API_KEY_PLACEHOLDER:
        return ""
    return api_key


def _resolve_api_key(args: argparse.Namespace) -> tuple[str, str]:
    direct_key = str(args.api_key).strip()
    if direct_key and direct_key != API_KEY_PLACEHOLDER:
        return direct_key, "test_script"

    generator_key = _load_api_key_from_generator(args.generator_script_path)
    if generator_key:
        return generator_key, f"generator_script:{args.generator_script_path}"

    env_key = os.getenv(args.api_key_env, "").strip()
    if env_key:
        return env_key, f"env:{args.api_key_env}"

    raise EnvironmentError(
        "Missing SiliconFlow API key. "
        "Please fill DEFAULT_API_KEY in this file, or fill DEFAULT_API_KEY in "
        f"{args.generator_script_path.name}, or set {args.api_key_env}."
    )


def _request_json(
    url: str,
    api_key: str,
    timeout_sec: int,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any] | str]:
    body: bytes | None = None
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        response_text = response.read().decode("utf-8", errors="ignore")
        try:
            parsed: dict[str, Any] | str = json.loads(response_text)
        except json.JSONDecodeError:
            parsed = response_text
        return int(response.status), parsed


def _extract_http_error(exc: urllib.error.HTTPError) -> tuple[int, str]:
    body = exc.read().decode("utf-8", errors="ignore")
    return int(exc.code), body.strip() or str(exc)


def _select_model(requested_model: str, models_payload: dict[str, Any] | str) -> tuple[str, list[str]]:
    if not isinstance(models_payload, dict):
        return requested_model, []

    items = models_payload.get("data", [])
    if not isinstance(items, list):
        return requested_model, []

    model_ids: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id", "")).strip()
        if model_id:
            model_ids.append(model_id)

    if requested_model in model_ids:
        return requested_model, model_ids

    for keyword in ("DeepSeek-V3.2", "deepseek"):
        for model_id in model_ids:
            if keyword.lower() in model_id.lower():
                return model_id, model_ids

    if model_ids:
        return model_ids[0], model_ids

    return requested_model, model_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test SiliconFlow API connectivity and authentication.")
    parser.add_argument("--api_key", default=DEFAULT_API_KEY)
    parser.add_argument("--api_key_env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--generator_script_path", type=Path, default=DEFAULT_GENERATOR_SCRIPT_PATH)
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout_sec", type=int, default=DEFAULT_TIMEOUT_SEC)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key, api_key_source = _resolve_api_key(args)
    base_url = _normalize_base_url(str(args.base_url))
    models_url = base_url + "/models"
    chat_url = base_url + "/chat/completions"

    print("=== SiliconFlow API Test ===")
    print(f"base_url={base_url}")
    print(f"api_key_source={api_key_source}")
    print(f"api_key_masked={_mask_secret(api_key)}")
    print(f"requested_model={args.model}")

    try:
        status_code, models_payload = _request_json(
            url=models_url,
            api_key=api_key,
            timeout_sec=int(args.timeout_sec),
            method="GET",
        )
        selected_model, available_models = _select_model(str(args.model), models_payload)
        print(f"[models] status={status_code}")
        print(f"[models] available_model_count={len(available_models)}")
        if available_models:
            print(f"[models] first_models={available_models[:10]}")
        print(f"[models] selected_model={selected_model}")
    except urllib.error.HTTPError as exc:
        status_code, body = _extract_http_error(exc)
        print(f"[models] status={status_code}")
        print(f"[models] body={_preview_text(body)}")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"[models] exception={type(exc).__name__}: {exc}")
        sys.exit(1)

    payload = {
        "model": selected_model,
        "messages": [
            {
                "role": "user",
                "content": "Reply with exactly one word: OK",
            }
        ],
        "temperature": 0.1,
        "max_tokens": 16,
    }

    try:
        status_code, chat_payload = _request_json(
            url=chat_url,
            api_key=api_key,
            timeout_sec=int(args.timeout_sec),
            method="POST",
            payload=payload,
        )
        print(f"[chat] status={status_code}")
        print(f"[chat] body={_preview_text(json.dumps(chat_payload, ensure_ascii=False))}")
        print("RESULT=PASS")
    except urllib.error.HTTPError as exc:
        status_code, body = _extract_http_error(exc)
        print(f"[chat] status={status_code}")
        print(f"[chat] body={_preview_text(body)}")
        print("RESULT=FAIL")
        sys.exit(2)
    except Exception as exc:  # noqa: BLE001
        print(f"[chat] exception={type(exc).__name__}: {exc}")
        print("RESULT=FAIL")
        sys.exit(2)


if __name__ == "__main__":
    main()
