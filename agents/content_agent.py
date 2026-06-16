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

    def analyze_tone(self, post_text: str) -> str:
        """
        Use LLM to extract a tone and style profile from the given post text.
        Returns a structured plain-text profile used to guide new post generation.
        """
        if not post_text.strip():
            return "No previous post available — use a confident, direct, professional tone with short punchy sentences."
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert writing coach. Analyze the LinkedIn post provided "
                        "and extract a precise tone and style profile. Be specific and actionable "
                        "— this profile will be used to write a new post that matches the author's voice exactly."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Analyze the tone and style of this LinkedIn post:\n\n{post_text}\n\n"
                        "Provide a concise profile covering these 7 dimensions (one line each):\n"
                        "1. Tone: (e.g., motivational, analytical, conversational, authoritative)\n"
                        "2. Sentence structure: (e.g., short punchy, longer narrative, mixed)\n"
                        "3. Vocabulary: (e.g., simple/accessible, technical, inspirational, jargon-free)\n"
                        "4. Formatting: (e.g., numbered lists, bullet points, plain paragraphs, emoji usage)\n"
                        "5. Hook style: (e.g., bold statement, open question, surprising statistic)\n"
                        "6. CTA style: (e.g., open reflective question, direct challenge, community invitation)\n"
                        "7. Visual/design mood: (e.g., bold energetic, clean minimalist, warm professional)"
                    ),
                },
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def generate_post(
        self,
        topic: str,
        user_context: str,
        tone_profile: str,
        post_format: PostFormat,
    ) -> dict:
        """
        Returns a dict with keys:
          text, hashtags, image_prompt
          + flyer_headline, flyer_subtitle  (FLYER only)
          + slides: [{title, body}, ...]    (CAROUSEL only)
        """
        user_prompt = self._build_prompt(
            topic, user_context, tone_profile, post_format
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
        tone_profile: str,
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
                "Also provide a 'slides' array of exactly 6 objects.\n"
                "Each slide object MUST have ALL of these keys:\n"
                "  - 'title': clear point (max 7 words)\n"
                "  - 'body': 40-60 word elaboration with a concrete example or stat\n"
                "  - 'key_stat': a bold short fact, number, or outcome (max 8 words, e.g. '73% faster deployments', 'Saves 4 hrs/week'). Empty string if not applicable.\n"
                "  - 'icon_concept': 1-3 words describing a relevant icon or symbol (e.g. 'cloud arrow', 'bar chart', 'rocket', 'shield check')\n"
                "  - 'layout': one of: 'split-left' (text left, visual right) | 'stat-callout' (centered big stat) | 'cover' | 'cta'\n"
                "  - 'slide_image_prompt': a vivid DALL-E prompt (40-60 words) describing an infographic-style illustration for this slide. "
                "Focus on diagrams, icons, charts, architecture flows, or process visuals. No people. No text in image. Clean professional style.\n\n"
                "Slide rules:\n"
                "  - Slide 1: Cover — layout='cover', big hook title, 1-line sub-headline as body\n"
                "  - Slides 2-5: Content — layout='split-left' or 'stat-callout'\n"
                "  - Slide 6: CTA — layout='cta', call-to-action phrase, engaging closing question as body"
            ),
            PostFormat.PPTX: (
                "Generate a professional PowerPoint deck structure for a LinkedIn audience.\n"
                "Return the usual 'text' (LinkedIn caption, 300-500 chars) and 'hashtags'.\n"
                "Also return a 'slides' array with EXACTLY these slide objects in order:\n\n"
                "1. ONE slide with type='title':\n"
                "   - title: Punchy deck title (max 8 words)\n"
                "   - subtitle: One-line value statement (max 14 words)\n\n"
                "2. ONE slide with type='overview':\n"
                "   - title: Framework/approach name (e.g. 'Implementation Approach')\n"
                "   - subtitle: Key message tagline\n"
                "   - steps: Array of 5-7 process step objects, each with:\n"
                "       * number: integer (1, 2, 3…)\n"
                "       * heading: SHORT ALL-CAPS label, max 3 words (e.g. 'ASSESSMENT')\n"
                "       * title: Step name, 3-5 words\n"
                "       * bullets: 3-4 concise action-oriented bullet points (plain text, no dashes)\n"
                "       * outcome: Short outcome phrase, 4-6 words\n\n"
                "3. TWO TO FOUR slides with type='detail', one per key phase/topic:\n"
                "   - section_number: integer matching the overview step this expands\n"
                "   - title: Section heading\n"
                "   - subtitle: Optional sub-heading (can be empty string)\n"
                "   - bullets: 4-6 detailed bullet points (plain text, no markdown)\n"
                "   - key_stat: One bold metric or outcome phrase (e.g. '4 Sprint Cycles', '30% faster') — empty string if none\n"
                "   - outcome: Outcome label (4-6 words) — empty string if none\n"
                "   - mermaid: Mermaid diagram code for this step's process flow.\n"
                "     RULES: Use 'flowchart LR' syntax. 4-6 nodes max. Node labels 2-4 words.\n"
                "     Example: 'flowchart LR\\n    A[Design] --> B[Build]\\n    B --> C[Test]\\n    C --> D[Release]'\n"
                "     Keep the code simple and valid — no subgraphs, no special chars in labels.\n\n"
                "4. ONE slide with type='summary':\n"
                "   - title: 'Key Risks & Success Factors'\n"
                "   - risks: 4-5 short risk phrases (plain text, no dashes)\n"
                "   - success_factors: 4-5 short success factor phrases (plain text, no dashes)\n"
                "   - tagline: One powerful closing statement (max 15 words)\n"
            ),
        }

        return f"""You are writing a LinkedIn post for a professional.

TOPIC: {topic}
ADDITIONAL CONTEXT / KEY POINTS: {user_context or "None provided"}

AUTHOR'S TONE & STYLE PROFILE (extracted from their most recent LinkedIn post):
{tone_profile}

CRITICAL INSTRUCTION: Mirror the EXACT tone, vocabulary, sentence structure, hook style, \
and CTA style described in the profile above. The new post must feel like it was written \
by the same person in the same voice — not a generic LinkedIn post.

FORMAT REQUIREMENT:
{format_instructions[post_format]}

ALSO ALWAYS INCLUDE IN YOUR JSON:
- "hashtags": list of exactly 5 highly relevant hashtags (with # prefix, mix popular + niche)
- "image_prompt": a vivid, detailed image generation prompt that matches the visual/design mood \
from the style profile above (describe scene, style, mood, colors) — suitable for generating \
a professional LinkedIn visual

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

        raw_slides = raw.get("slides", [])

        # PPTX slides have a 'type' field — pass them through without carousel normalisation
        if raw_slides and isinstance(raw_slides[0], dict) and raw_slides[0].get("type"):
            normalised_slides = raw_slides
        else:
            # Carousel slide normalisation — guarantee all infographic keys exist
            normalised_slides = []
            for i, s in enumerate(raw_slides):
                layout_default = "cover" if i == 0 else ("cta" if i == len(raw_slides) - 1 else "split-left")
                normalised_slides.append({
                    "title": clean(s.get("title", f"Point {i}")),
                    "body": clean(s.get("body", "")),
                    "key_stat": clean(s.get("key_stat", "")),
                    "icon_concept": s.get("icon_concept", ""),
                    "layout": s.get("layout", layout_default),
                    "slide_image_prompt": s.get(
                        "slide_image_prompt",
                        f"Professional infographic illustration for '{s.get('title', topic)}', "
                        "clean diagram style, no text, corporate blue palette.",
                    ),
                })

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
            "slides": normalised_slides,
        }
