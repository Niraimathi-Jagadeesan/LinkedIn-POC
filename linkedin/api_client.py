"""
LinkedIn REST API client — uses the new Content API (2023+).
Old endpoints (/v2/assets, /v2/ugcPosts) are restricted to partner apps;
the new /rest/* endpoints work with the standard "Share on LinkedIn" product.
"""

import urllib.parse
from pathlib import Path

import requests


class LinkedInAPIClient:
    BASE_URL = "https://api.linkedin.com/v2"
    REST_URL = "https://api.linkedin.com/rest"
    _LINKEDIN_VERSION = "202601"

    def __init__(self, access_token: str) -> None:
        self._token = access_token
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {access_token}",
                "X-Restli-Protocol-Version": "2.0.0",
                "LinkedIn-Version": self._LINKEDIN_VERSION,
                "Content-Type": "application/json",
            }
        )

    # ── Profile ──────────────────────────────────────────────────────────────

    def get_profile(self) -> dict:
        # LinkedIn OIDC userinfo endpoint (works with openid + profile scopes)
        resp = self._session.get("https://api.linkedin.com/v2/userinfo", timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_person_urn(self) -> str:
        profile = self.get_profile()
        # OIDC userinfo returns 'sub' as the member ID
        return f"urn:li:person:{profile['sub']}"

    # ── Post Retrieval ────────────────────────────────────────────────────────

    def fetch_posts(self, person_urn: str, count: int = 50) -> list[dict]:
        """Fetch recent posts. Returns empty list if scope not granted."""
        encoded = urllib.parse.quote(person_urn, safe="")
        url = (
            f"{self.BASE_URL}/ugcPosts"
            f"?q=authors&authors=List({encoded})"
            f"&count={count}&sortBy=LAST_MODIFIED"
        )
        try:
            resp = self._session.get(url, timeout=20)
            resp.raise_for_status()
            return resp.json().get("elements", [])
        except Exception:
            return []

    # ── Media Upload — Image ──────────────────────────────────────────────────

    def init_image_upload(self, person_urn: str) -> dict:
        """Initialize an image upload. Returns uploadUrl and image URN."""
        url = f"{self.REST_URL}/images?action=initializeUpload"
        payload = {"initializeUploadRequest": {"owner": person_urn}}
        resp = self._session.post(url, json=payload, timeout=20)
        resp.raise_for_status()
        return resp.json()["value"]   # {"uploadUrl": "...", "image": "urn:li:image:xxx"}

    def upload_image(self, upload_url: str, image_path: str) -> None:
        """PUT image bytes to the pre-signed upload URL."""
        with open(image_path, "rb") as fh:
            data = fh.read()
        resp = requests.put(
            upload_url,
            data=data,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "image/png",
            },
            timeout=60,
        )
        resp.raise_for_status()

    # ── Media Upload — Document (carousel PDF) ────────────────────────────────

    def init_document_upload(self, person_urn: str) -> dict:
        """Initialize a document upload. Returns uploadUrl and document URN."""
        url = f"{self.REST_URL}/documents?action=initializeUpload"
        payload = {"initializeUploadRequest": {"owner": person_urn}}
        resp = self._session.post(url, json=payload, timeout=20)
        resp.raise_for_status()
        return resp.json()["value"]   # {"uploadUrl": "...", "document": "urn:li:document:xxx"}

    def upload_document(self, upload_url: str, pdf_path: str) -> None:
        """PUT PDF bytes to the pre-signed upload URL."""
        with open(pdf_path, "rb") as fh:
            data = fh.read()
        resp = requests.put(
            upload_url,
            data=data,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/octet-stream",
            },
            timeout=120,
        )
        resp.raise_for_status()

    # ── Post Creation (new Content API) ───────────────────────────────────────

    def _post(self, payload: dict) -> dict:
        """Internal helper: POST to /rest/posts and return {id: urn}."""
        resp = self._session.post(f"{self.REST_URL}/posts", json=payload, timeout=30)
        if not resp.ok:
            try:
                print(f"[LinkedIn API error] status={resp.status_code} body={resp.text[:500]}")
                print(f"[LinkedIn API error] headers={dict(resp.headers)}")
            except Exception:
                pass
        resp.raise_for_status()
        post_id = resp.headers.get("x-linkedin-id") or resp.headers.get("x-restli-id", "")
        return {"id": post_id}

    def _base_payload(self, person_urn: str, text: str, visibility: str) -> dict:
        return {
            "author": person_urn,
            "commentary": text,
            "visibility": visibility,
            "distribution": {
                "feedDistribution": "MAIN_FEED",
                "targetEntities": [],
                "thirdPartyDistributionChannels": [],
            },
            "lifecycleState": "PUBLISHED",
            "isReshareDisabledByAuthor": False,
        }

    def create_text_post(self, person_urn: str, text: str, visibility: str = "PUBLIC") -> dict:
        return self._post(self._base_payload(person_urn, text, visibility))

    def create_image_post(
        self, person_urn: str, text: str, asset_urn: str, title: str = "", visibility: str = "PUBLIC"
    ) -> dict:
        payload = self._base_payload(person_urn, text, visibility)
        payload["content"] = {"media": {"title": title or "Image", "id": asset_urn}}
        return self._post(payload)

    def create_document_post(
        self, person_urn: str, text: str, asset_urn: str, title: str = "", visibility: str = "PUBLIC"
    ) -> dict:
        payload = self._base_payload(person_urn, text, visibility)
        payload["content"] = {"media": {"title": title or "Carousel", "id": asset_urn}}
        return self._post(payload)

    def delete_post(self, post_urn: str) -> None:
        """Delete a post by URN. Supports both urn:li:share:xxx and urn:li:ugcPost:xxx."""
        encoded = urllib.parse.quote(post_urn, safe="")
        if "ugcPost" in post_urn:
            resp = self._session.delete(f"{self.BASE_URL}/ugcPosts/{encoded}", timeout=20)
        else:
            resp = self._session.delete(f"{self.REST_URL}/posts/{encoded}", timeout=20)
        resp.raise_for_status()

