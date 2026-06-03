"""
Content Agent — generates LinkedIn post text, hashtags, and visual prompts.
Supports multiple LLM providers: OpenAI, Groq (free), Ollama (local).
"""

import json

from openai import OpenAI

from config.settings import get_settings
from linkedin.models import PostFormat


def _build_llm_client(settings) -> tuple["OpenAI", str]:
    """
    Return (OpenAI-compatible client, model name) for the configured provider.
    Groq and Ollama both expose an OpenAI-compatible REST API.
    """
    if settings.llm_provider == "groq":
        return (
            OpenAI(
                api_key=settings.groq_api_key,
                base_url="https://api.groq.com/openai/v1",
            ),
            settings.llm_model,
        )
    if settings.llm_provider == "ollama":
        return (
            OpenAI(
                api_key="ollama",          # Ollama ignores the key
                base_url=settings.ollama_base_url,
            ),
            settings.llm_model,
        )
    # Default: OpenAI
    return OpenAI(api_key=settings.openai_api_key), settings.llm_model


class ContentAgent:
    _SYSTEM_PROMPT = (
        "You are an expert LinkedIn content creator. "
        "Study the user's past posts to match their writing style, tone, and voice. "
        "Always write in first person. Keep content professional yet conversational. "
        "Output ONLY valid JSON — no markdown fences, no extra commentary."
    )

    def __init__(self) -> None:
        settings = get_settings()
        self._client, self._model = _build_llm_client(settings)
        self._provider = settings.llm_provider

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_post(
        self,
        topic: str,
        user_context: str,
        past_posts_context: str,
        post_format: PostFormat,
    ) -> dict:
        """
        Returns a dict with keys:
          text, hashtags, image_prompt
          + flyer_headline, flyer_subtitle  (FLYER only)
          + slides: [{title, body}, ...]    (CAROUSEL only)
        """
        user_prompt = self._build_prompt(
            topic, user_context, past_posts_context, post_format
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        result = json.loads(response.choices[0].message.content)
        return self._normalise(result, topic)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        topic: str,
        user_context: str,
        past_posts_context: str,
        post_format: PostFormat,
    ) -> str:
        format_instructions = {
            PostFormat.TEXT: (
                "Write a compelling LinkedIn text post (max 1 300 characters). "
                "Use short paragraphs and line breaks. End with a question or CTA."
            ),
            PostFormat.IMAGE: (
                "Write a short LinkedIn image caption (max 300 characters) that "
                "complements a visual. Punchy and engaging."
            ),
            PostFormat.FLYER: (
                "Write a LinkedIn post caption (max 200 characters). "
                "Also provide a 'flyer_headline' (max 10 words) and a "
                "'flyer_subtitle' (max 20 words) for the graphic card."
            ),
            PostFormat.CAROUSEL: (
                "Write a LinkedIn post caption (max 200 characters) to introduce "
                "the carousel. Also provide a 'slides' array of exactly 5 objects "
                "each with 'title' (max 8 words) and 'body' (max 40 words)."
            ),
        }

        return f"""Topic: {topic}
Additional context: {user_context or "None"}

{past_posts_context}

Format requirement:
{format_instructions[post_format]}

Also always include:
- "hashtags": list of exactly 5 relevant hashtags (with # prefix)
- "image_prompt": a detailed DALL-E image generation prompt that visually represents the topic

Return valid JSON only."""

    @staticmethod
    def _normalise(raw: dict, topic: str) -> dict:
        """Ensure required keys are present with sensible fallbacks."""
        return {
            "text": raw.get("text", ""),
            "hashtags": raw.get("hashtags", []),
            "image_prompt": raw.get(
                "image_prompt",
                f"Professional LinkedIn graphic about {topic}, clean modern design",
            ),
            "flyer_headline": raw.get("flyer_headline", topic),
            "flyer_subtitle": raw.get("flyer_subtitle", ""),
            "slides": raw.get("slides", []),
        }
