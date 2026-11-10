"""Đọc environment variables, không hardcode endpoint container trong code."""

import os


def env(name: str, default: str) -> str:
    return os.getenv(name, default).rstrip("/")


def registry_url() -> str:
    return env("REGISTRY_URL", "http://127.0.0.1:9000")
