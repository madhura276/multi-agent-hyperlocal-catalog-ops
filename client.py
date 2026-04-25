from __future__ import annotations

from typing import Optional

import requests

from models import MultiAgentAction


class MultiAgentHyperlocalCatalogOpsClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict:
        response = requests.get(f"{self.base_url}/health", timeout=30)
        response.raise_for_status()
        return response.json()

    def reset(self, payload: Optional[dict] = None) -> dict:
        response = requests.post(
            f"{self.base_url}/reset",
            json=payload or {},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def step(self, action: MultiAgentAction) -> dict:
        response = requests.post(
            f"{self.base_url}/step",
            json={"action": action.model_dump()},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def state(self) -> dict:
        response = requests.get(f"{self.base_url}/state", timeout=30)
        response.raise_for_status()
        return response.json()
