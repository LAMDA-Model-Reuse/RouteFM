#!/usr/bin/env python3
"""Submit and monitor a SinapisAI video-generation task.

Credentials are read only from SINAPISAI_API_KEY. The key is never printed or
written to disk. Load a local .env.video with `set -a; source .env.video; set +a`
before invoking this script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError as exc:  # pragma: no cover - depends on the caller's env
    raise SystemExit(
        "Missing dependency: install it with `python3 -m pip install requests`."
    ) from exc


SUCCESS_STATES = {"completed", "done", "succeeded", "success"}
FAILURE_STATES = {"cancelled", "canceled", "error", "failed"}


def env_number(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be a number, got {raw!r}.") from exc


def require_api_key() -> str:
    api_key = os.getenv("SINAPISAI_API_KEY", "").strip()
    if not api_key or api_key == "replace_me":
        raise SystemExit(
            "SINAPISAI_API_KEY is missing. Load it with "
            "`set -a; source .env.video; set +a`."
        )
    return api_key


def response_json(response: requests.Response) -> dict[str, Any]:
    response.raise_for_status()
    try:
        payload = response.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise RuntimeError("SinapisAI returned a non-JSON response.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("SinapisAI returned JSON that is not an object.")
    return payload


def save_result(output_dir: Path, task_id: str, payload: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{task_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def task_state(payload: dict[str, Any]) -> str:
    return str(payload.get("status") or payload.get("state") or "unknown").lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="Text-to-video prompt.")
    parser.add_argument(
        "--model",
        default=os.getenv(
            "SINAPISAI_VIDEO_MODEL", "doubao/doubao-seedance-2-0-260128"
        ),
        help="Provider model identifier.",
    )
    parser.add_argument(
        "--api-base",
        default=os.getenv("SINAPISAI_API_BASE", "https://api.sinapisai.com/v1"),
        help="SinapisAI API base URL.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=env_number("SINAPISAI_POLL_INTERVAL", 10.0),
        help="Seconds between status requests.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=env_number("SINAPISAI_TIMEOUT", 1800.0),
        help="Maximum seconds to wait for a terminal state.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sinapisai"),
        help="Directory for task metadata JSON.",
    )
    parser.add_argument(
        "--submit-only",
        action="store_true",
        help="Create the task and exit without polling.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.poll_interval <= 0 or args.timeout <= 0:
        raise SystemExit("--poll-interval and --timeout must be positive.")

    api_key = require_api_key()
    api_base = args.api_base.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"}

    with requests.Session() as session:
        created = response_json(
            session.post(
                f"{api_base}/videos",
                headers=headers,
                json={"model": args.model, "prompt": args.prompt},
                timeout=60,
            )
        )
        task_id = str(created.get("id") or "").strip()
        if not task_id:
            raise RuntimeError("SinapisAI create response does not contain an id.")

        result_path = save_result(args.output_dir, task_id, created)
        print(f"Submitted task {task_id}")
        print(f"Metadata: {result_path}")
        if args.submit_only:
            return 0

        started = time.monotonic()
        previous_state: str | None = None
        while True:
            result = response_json(
                session.get(f"{api_base}/videos/{task_id}", headers=headers, timeout=60)
            )
            result_path = save_result(args.output_dir, task_id, result)
            state = task_state(result)
            if state != previous_state:
                print(f"Status: {state}")
                previous_state = state

            if state in SUCCESS_STATES:
                print(f"Completed. Provider response: {result_path}")
                return 0
            if state in FAILURE_STATES:
                print(f"Task ended with status {state}. Details: {result_path}", file=sys.stderr)
                return 1
            if state == "unknown":
                print(
                    "The response has no recognized status/state field; "
                    "saved it without further polling."
                )
                return 0
            if time.monotonic() - started >= args.timeout:
                print(
                    f"Timed out while waiting; latest provider response: {result_path}",
                    file=sys.stderr,
                )
                return 2
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except requests.RequestException as exc:
        print(f"SinapisAI request failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
