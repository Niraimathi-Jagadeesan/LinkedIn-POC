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
        "You are a world-class LinkedIn content strategist and copywriter with 10+ years "
        "of experience growing professional audiences. Your posts consistently achieve "
        "high engagement through storytelling, insight, and a strong personal voice.\n\n"
        "Rules you ALWAYS follow:\n"
        "1. Open with a powerful HOOK (bold statement, surprising stat, or provocative question) — "
        "the first line must stop the scroll.\n"
        "2. Write in first person, conversational but authoritative tone.\n"
        "3. Use short punchy sentences. Break long ideas into single-line paragraphs.\n"
        "4. Include a real-world insight, data point, or concrete example tied to the topic.\n"
        "5. Expand and elaborate — do NOT just restate the topic. Tell a story around it.\n"
        "6. Build to a clear takeaway or lesson.\n"
        "7. End with a genuine call-to-action (question, challenge, or invitation to comment).\n"
        "8. Never use corporate buzzwords like 'leverage', 'synergy', 'circle back'.\n"
        "9. NEVER use markdown formatting: no **bold**, no ##headings, no bullet dashes (-), "
        "no backticks. Plain text only. Use line breaks (\\n) for structure.\n"
        "10. Study the user's past posts to mirror their vocabulary and voice.\n"
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
                "Write a compelling LinkedIn text post (1 000–1 500 characters).\n"
                "Structure: Hook (1 line) → Context/Story (3-5 short paragraphs) → "
                "Key insight or list of 3-5 takeaways → Strong CTA question.\n"
                "Use blank lines between every paragraph. Use emojis sparingly (1-2 max) "
                "only where they genuinely add clarity."
            ),
            PostFormat.IMAGE: (
                "Write a full LinkedIn post to accompany an image (800–1 200 characters).\n"
                "Structure: Powerful hook (1 bold line) → Story or insight (3-4 short paragraphs) "
                "→ 3-5 key takeaways or bullet points → CTA question.\n"
                "The post must work as a standalone read even without the image.\n"
                "Also provide 'hook_line': the very first sentence only (max 12 words), "
                "used as a text overlay on the image."
            ),
            PostFormat.FLYER: (
                "Write a LinkedIn flyer post caption (300–500 characters).\n"
                "Structure: Hook → 1-2 value sentences → CTA.\n"
                "Also provide:\n"
                "- 'flyer_headline': punchy headline max 8 words (ALL CAPS impact phrase)\n"
                "- 'flyer_subtitle': supporting line max 15 words that expands on the headline"
            ),
            PostFormat.CAROUSEL: (
                "Write a LinkedIn carousel post caption (300–500 characters) that teases "
                "the value inside ('Swipe to discover...' style hook).\n"
                "Also provide a 'slides' array of exactly 6 objects:\n"
                "  - Slide 1: Cover — 'title': bold hook title, 'body': 1-line sub-headline\n"
                "  - Slides 2-5: Content — 'title': clear point (max 7 words), "
                "'body': 40-60 word elaboration with a concrete example or stat\n"
                "  - Slide 6: CTA — 'title': call-to-action phrase, 'body': engaging closing question"
            ),
        }

        return f"""You are writing a LinkedIn post for a professional in this niche.

TOPIC: {topic}
ADDITIONAL CONTEXT / KEY POINTS: {user_context or "None provided"}

PAST POSTS FOR STYLE REFERENCE:
{past_posts_context}

FORMAT REQUIREMENT:
{format_instructions[post_format]}

ALSO ALWAYS INCLUDE IN YOUR JSON:
- "hashtags": list of exactly 5 highly relevant hashtags (with # prefix, mix popular + niche)
- "image_prompt": a vivid, detailed image generation prompt (describe scene, style, mood, colors) — suitable for generating a professional LinkedIn visual

CRITICAL REMINDERS:
- The "text" field must be a FULLY WRITTEN, well-elaborated LinkedIn post — not a placeholder or topic restatement.
- NO markdown in text fields: no **bold**, no ##, no dashes, no backticks. Plain text + newlines only.
- Write as if this will be copy-pasted directly into LinkedIn.

Return ONLY a valid JSON object. No explanation outside the JSON."""

    @staticmethod
    def _clean_text(text: str) -> str:
        """Strip markdown formatting that leaks into LinkedIn post text."""
        import re
        # Remove **bold** and *italic*
        text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
        # Remove ##headings
        text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
        # Remove backticks
        text = re.sub(r'`+', '', text)
        # Convert markdown bullet dashes at line start to a clean bullet
        text = re.sub(r'^\s*[-*]\s+', '• ', text, flags=re.MULTILINE)
        # Collapse 3+ newlines to 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def _normalise(raw: dict, topic: str) -> dict:
        """Ensure required keys are present with sensible fallbacks."""
        clean = ContentAgent._clean_text
        return {
            "text": clean(raw.get("text", "")),
            "hashtags": raw.get("hashtags", []),
            "image_prompt": raw.get(
                "image_prompt",
                f"Professional LinkedIn graphic about {topic}, clean modern design",
            ),
            "hook_line": clean(raw.get("hook_line", "")),
            "flyer_headline": clean(raw.get("flyer_headline", topic)),
            "flyer_subtitle": clean(raw.get("flyer_subtitle", "")),
            "slides": raw.get("slides", []),
        }
