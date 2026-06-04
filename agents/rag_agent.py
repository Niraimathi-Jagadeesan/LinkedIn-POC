"""
RAG Agent — fetches previous LinkedIn posts, indexes them in ChromaDB,
and retrieves relevant context for new post generation.
"""

from embeddings.vectorstore import LinkedInVectorStore
from linkedin.api_client import LinkedInAPIClient


class RAGAgent:
    def __init__(
        self, api_client: LinkedInAPIClient, vectorstore: LinkedInVectorStore
    ) -> None:
        self._api = api_client
        self._vs = vectorstore

    def index_past_posts(self, person_urn: str) -> int:
        """Fetch all posts from LinkedIn and embed them into ChromaDB."""
        print("Fetching your LinkedIn posts for context indexing...")
        posts = self._api.fetch_posts(person_urn)
        count = self._vs.add_posts(posts)
        print(f"Indexed {count} new posts into memory ({len(posts)} total fetched).")
        return count

    def get_last_post_text(self, person_urn: str) -> str:
        """
        Return the text of the most recently published LinkedIn post.
        Tries the live API first; falls back to the ChromaDB vectorstore
        (which holds previously indexed posts) if the API returns nothing.
        """
        text = self._api.fetch_last_post(person_urn)
        if text:
            return text
        # API returned nothing (scope not granted, token expired, mock, etc.)
        # Fall back to whatever is already indexed in ChromaDB.
        text = self._vs.get_most_recent()
        if text:
            print("  [INFO] Live API returned no posts — using most recent indexed post from local store.")
        return text

    def retrieve_context(self, topic: str, user_context: str) -> str:
        """Return a formatted string of the 3 most relevant past posts."""
        query = f"{topic} {user_context}".strip()
        similar = self._vs.search_similar(query, k=3)

        if not similar:
            return "No previous posts found — writing fresh content."

        lines = ["Here are some of your previous LinkedIn posts on similar topics:\n"]
        for i, text in enumerate(similar, 1):
            lines.append(f"--- Post {i} ---\n{text}\n")
        return "\n".join(lines)
