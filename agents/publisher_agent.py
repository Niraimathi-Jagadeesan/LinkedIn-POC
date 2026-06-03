"""
Publisher Agent — uploads media assets to LinkedIn and creates the UGC post.
In dry-run mode (DRY_RUN=true) no API calls are made; output is saved locally.
"""

import json
import uuid
from pathlib import Path

from config.settings import get_settings
from linkedin.api_client import LinkedInAPIClient
from linkedin.models import GeneratedContent, PostFormat


class PublisherAgent:
    def __init__(self, api_client: LinkedInAPIClient) -> None:
        self._api = api_client
        settings = get_settings()
        self._dry_run = settings.dry_run
        self._visibility = settings.post_visibility  # PUBLIC | CONNECTIONS | LOGGED_IN

    def publish(self, person_urn: str, content: GeneratedContent) -> dict:
        """Upload media (if any) and publish the post. Returns the API response."""
        full_text = content.text
        if content.hashtags:
            full_text = f"{content.text}\n\n{' '.join(content.hashtags)}"

        if self._dry_run:
            return self._dry_run_save(person_urn, content, full_text)

        fmt = content.format

        if fmt == PostFormat.TEXT:
            return self._api.create_text_post(person_urn, full_text, self._visibility)

        if fmt == PostFormat.IMAGE:
            try:
                asset_urn = self._upload_image(person_urn, content.image_path)
                return self._api.create_image_post(
                    person_urn, full_text, asset_urn, content.topic, self._visibility
                )
            except Exception as exc:
                print(f"[WARN] Image upload failed ({exc}). Falling back to text post.")
                return self._api.create_text_post(person_urn, full_text, self._visibility)

        if fmt == PostFormat.FLYER:
            try:
                asset_urn = self._upload_image(person_urn, content.flyer_path)
                return self._api.create_image_post(
                    person_urn, full_text, asset_urn, content.topic, self._visibility
                )
            except Exception as exc:
                print(f"[WARN] Flyer upload failed ({exc}). Falling back to text post.")
                return self._api.create_text_post(person_urn, full_text, self._visibility)

        if fmt == PostFormat.CAROUSEL:
            try:
                asset_urn = self._upload_document(person_urn, content.carousel_path)
                return self._api.create_document_post(
                    person_urn, full_text, asset_urn, content.topic, self._visibility
                )
            except Exception as exc:
                print(f"[WARN] Document upload not available ({exc}).")
                print("[WARN] Falling back to formatted text post with slide content.")
                slides_text = self._slides_as_text(content)
                return self._api.create_text_post(person_urn, slides_text, self._visibility)

        raise ValueError(f"Unknown post format: {fmt}")

    # ── Dry-run ───────────────────────────────────────────────────────────────

    def _dry_run_save(self, person_urn: str, content: GeneratedContent, full_text: str) -> dict:
        """Save generated content to a local JSON file instead of posting."""
        out_dir = Path("./data/dry_run")
        out_dir.mkdir(parents=True, exist_ok=True)

        fake_id = f"dry-run-{uuid.uuid4().hex[:8]}"
        payload = {
            "id": fake_id,
            "dry_run": True,
            "author": person_urn,
            "format": content.format.value,
            "text": full_text,
            "hashtags": content.hashtags,
            "image_prompt": content.image_prompt,
            "image_path": content.image_path,
            "flyer_path": content.flyer_path,
            "carousel_path": content.carousel_path,
        }
        out_path = out_dir / f"{fake_id}.json"
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"[DRY RUN] Post NOT sent to LinkedIn.")
        print(f"[DRY RUN] Output saved to: {out_path}")
        return payload

    # ── Private helpers ───────────────────────────────────────────────────────

    def _upload_image(self, person_urn: str, file_path: str) -> str:
        info = self._api.init_image_upload(person_urn)
        self._api.upload_image(info["uploadUrl"], file_path)
        return info["image"]   # urn:li:image:xxx

    def _upload_document(self, person_urn: str, file_path: str) -> str:
        info = self._api.init_document_upload(person_urn)
        self._api.upload_document(info["uploadUrl"], file_path)
        return info["document"]   # urn:li:document:xxx

    def _slides_as_text(self, content: "GeneratedContent") -> str:
        """Format carousel slides as a structured text post when PDF upload unavailable."""
        icons = ["💡", "🔹", "🔸", "✅", "🚀", "📌", "💼", "🎯"]
        lines = []
        slides = content.slides or []
        for i, slide in enumerate(slides):
            icon = icons[i % len(icons)]
            title = slide.get("title", f"Point {i + 1}")
            body = slide.get("body", "")
            lines.append(f"{icon} {title}")
            if body:
                lines.append(body)
            lines.append("")
        if content.hashtags:
            lines.append(" ".join(content.hashtags))
        return "\n".join(lines).strip()
