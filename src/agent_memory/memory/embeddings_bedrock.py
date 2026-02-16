"""Embedding provider using AWS Bedrock Titan."""

from __future__ import annotations

import json

import boto3

from .embeddings import normalize


class BedrockEmbeddings:
    """Generate text embeddings via AWS Bedrock Titan."""

    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v2:0",
        region: str = "us-east-1",
        dimensions: int = 1024,
    ) -> None:
        self._client = boto3.client("bedrock-runtime", region_name=region)
        self._model_id = model_id
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns a normalized vector."""
        body = json.dumps({"inputText": text, "dimensions": self._dimensions})
        response = self._client.invoke_model(modelId=self._model_id, body=body)
        result = json.loads(response["body"].read())
        vec = result["embedding"]
        return normalize(vec)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Titan doesn't support native batching, so we loop."""
        return [self.embed(t) for t in texts]
