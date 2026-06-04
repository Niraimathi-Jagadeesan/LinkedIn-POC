"""
ChromaDB-backed vector store for LinkedIn post embeddings.
Supports two embedding backends:
  - openai   : OpenAI text-embedding-3-small (needs OPENAI_API_KEY)
  - local    : ChromaDB built-in ONNX model  (no API key, downloads ~22 MB once)
"""

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

from config.settings import get_settings


class LinkedInVectorStore:
    _COLLECTION = "linkedin_posts"

    def __init__(self) -> None:
        settings = get_settings()
        Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._use_local = settings.embedding_provider == "local"

        if self._use_local:
            # Uses a bundled ONNX sentence-transformer model — no API key needed.
            self._local_ef = embedding_functions.DefaultEmbeddingFunction()
            self._collection = self._client.get_or_create_collection(
                name=self._COLLECTION,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self._local_ef,
            )
        else:
            from openai import OpenAI
            self._openai = OpenAI(api_key=settings.openai_api_key)
            self._embed_model = settings.embedding_model
            self._collection = self._client.get_or_create_collection(
                name=self._COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        if self._use_local:
            # DefaultEmbeddingFunction returns a list of embeddings; take the first
            return self._local_ef([text])[0]
        resp = self._openai.embeddings.create(model=self._embed_model, input=text)
        return resp.data[0].embedding

    # ── Indexing ──────────────────────────────────────────────────────────────

    def add_posts(self, raw_posts: list[dict]) -> int:
        """
        Embed and store LinkedIn UGC post objects.
        Returns the number of *new* posts indexed (skips duplicates by URN).
        """
        added = 0
        for post in raw_posts:
            urn = post.get("id", "")
            try:
                text = post["specificContent"]["com.linkedin.ugc.ShareContent"][
                    "shareCommentary"
                ]["text"]
            except (KeyError, TypeError):
                continue

            if not text or not urn:
                continue

            # Skip posts already in the store
            if self._collection.get(ids=[urn])["ids"]:
                continue

            if self._use_local:
                # Let ChromaDB handle embedding via the collection's embedding_function
                self._collection.add(
                    documents=[text],
                    ids=[urn],
                    metadatas=[{"author": post.get("author", ""), "created": str(post.get("created", {}).get("time", ""))}],
                )
            else:
                self._collection.add(
                    documents=[text],
                    embeddings=[self._embed(text)],
                    ids=[urn],
                    metadatas=[{"author": post.get("author", ""), "created": str(post.get("created", {}).get("time", ""))}],
                )
            added += 1
        return added

    def add_text(
        self, text: str, post_id: str, metadata: Optional[dict] = None
    ) -> None:
        """Store a single text entry (e.g. a freshly published post)."""
        if self._use_local:
            self._collection.add(
                documents=[text],
                ids=[post_id],
                metadatas=[metadata or {}],
            )
        else:
            self._collection.add(
                documents=[text],
                embeddings=[self._embed(text)],
                ids=[post_id],
                metadatas=[metadata or {}],
            )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def search_similar(self, query: str, k: int = 3) -> list[str]:
        """Return up to k most relevant post texts for the given query."""
        total = self._collection.count()
        if total == 0:
            return []
        if self._use_local:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(k, total),
            )
        else:
            results = self._collection.query(
                query_embeddings=[self._embed(query)],
                n_results=min(k, total),
            )
        return results["documents"][0] if results["documents"] else []

    def count(self) -> int:
        return self._collection.count()

    def get_most_recent(self) -> str:
        """
        Return the text of the post with the latest 'created' timestamp stored in
        the collection. Returns empty string if nothing is indexed.
        """
        total = self._collection.count()
        if total == 0:
            return ""
        result = self._collection.get(include=["documents", "metadatas"])
        pairs = list(zip(result["documents"], result["metadatas"]))
        # Sort descending by the 'created' timestamp stored as a string
        pairs.sort(
            key=lambda p: int(p[1].get("created", "0") or "0"),
            reverse=True,
        )
        return pairs[0][0] if pairs else ""
