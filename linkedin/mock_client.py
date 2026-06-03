"""
Mock LinkedIn API client for testing without a LinkedIn Developer App.
Returns realistic dummy data so the full AI pipeline (RAG + content generation
+ visual creation) can be tested with only an OpenAI API key.
"""

import uuid

MOCK_PERSON_URN = "urn:li:person:mock_test_user"

# Five realistic sample posts used to seed the RAG vector store during testing.
SAMPLE_POSTS = [
    {
        "id": "urn:li:ugcPost:sample001",
        "author": MOCK_PERSON_URN,
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {
                    "text": (
                        "Leadership is not about having all the answers. "
                        "It's about asking the right questions and trusting your team "
                        "to find solutions together. The best leaders I've worked with "
                        "were always the best listeners.\n\n"
                        "What's one leadership lesson that changed how you work?"
                    )
                },
                "shareMediaCategory": "NONE",
            }
        },
        "created": {"time": 1700000000000},
    },
    {
        "id": "urn:li:ugcPost:sample002",
        "author": MOCK_PERSON_URN,
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {
                    "text": (
                        "3 things I wish I knew before starting my tech career:\n\n"
                        "1. Soft skills matter more than technical skills at senior levels\n"
                        "2. Building your network is a long game — invest early\n"
                        "3. Your ability to communicate ideas is your biggest differentiator\n\n"
                        "What would you add to this list?"
                    )
                },
                "shareMediaCategory": "NONE",
            }
        },
        "created": {"time": 1698000000000},
    },
    {
        "id": "urn:li:ugcPost:sample003",
        "author": MOCK_PERSON_URN,
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {
                    "text": (
                        "AI is not replacing jobs. It's replacing tasks.\n\n"
                        "The people who will thrive are those who learn to work "
                        "alongside AI — using it to amplify their unique human skills: "
                        "creativity, empathy, and strategic thinking.\n\n"
                        "Are you upskilling to stay relevant?"
                    )
                },
                "shareMediaCategory": "NONE",
            }
        },
        "created": {"time": 1695000000000},
    },
    {
        "id": "urn:li:ugcPost:sample004",
        "author": MOCK_PERSON_URN,
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {
                    "text": (
                        "The biggest mistake teams make in retrospectives: "
                        "they focus on what went wrong instead of what could go better.\n\n"
                        "Shift from blame to improvement. "
                        "One small process change per sprint beats a long list of problems."
                    )
                },
                "shareMediaCategory": "NONE",
            }
        },
        "created": {"time": 1692000000000},
    },
    {
        "id": "urn:li:ugcPost:sample005",
        "author": MOCK_PERSON_URN,
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {
                    "text": (
                        "Consistency beats perfection every time.\n\n"
                        "Showing up every day — even when your work feels average — "
                        "compounds into mastery over time. "
                        "The professionals who stand out are rarely the most talented. "
                        "They're the most consistent.\n\n"
                        "What's your strategy for staying consistent?"
                    )
                },
                "shareMediaCategory": "NONE",
            }
        },
        "created": {"time": 1690000000000},
    },
]


class MockLinkedInAPIClient:
    """
    Drop-in replacement for LinkedInAPIClient.
    All network calls are no-ops; responses contain plausible fake data.
    """

    def get_profile(self) -> dict:
        return {
            "id": "mock_test_user",
            "localizedFirstName": "Test",
            "localizedLastName": "User",
        }

    def get_person_urn(self) -> str:
        return MOCK_PERSON_URN

    def fetch_posts(self, person_urn: str, count: int = 50) -> list[dict]:
        return SAMPLE_POSTS

    # ── Mock upload methods (all no-ops) ──────────────────────────────────────

    def register_image_upload(self, person_urn: str) -> dict:
        return {
            "value": {
                "uploadMechanism": {
                    "com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest": {
                        "uploadUrl": "https://mock-upload.linkedin.example/image"
                    }
                },
                "asset": f"urn:li:digitalmediaAsset:mock-img-{uuid.uuid4().hex[:8]}",
            }
        }

    def upload_image(self, upload_url: str, image_path: str) -> None:
        pass

    def register_document_upload(self, person_urn: str) -> dict:
        return {
            "value": {
                "uploadMechanism": {
                    "com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest": {
                        "uploadUrl": "https://mock-upload.linkedin.example/document"
                    }
                },
                "asset": f"urn:li:digitalmediaAsset:mock-doc-{uuid.uuid4().hex[:8]}",
            }
        }

    def upload_document(self, upload_url: str, pdf_path: str) -> None:
        pass

    # ── Mock post creation ────────────────────────────────────────────────────

    def create_text_post(self, person_urn: str, text: str) -> dict:
        return {"id": f"urn:li:ugcPost:mock-{uuid.uuid4().hex[:8]}"}

    def create_image_post(
        self, person_urn: str, text: str, asset_urn: str, title: str = ""
    ) -> dict:
        return {"id": f"urn:li:ugcPost:mock-{uuid.uuid4().hex[:8]}"}

    def create_document_post(
        self, person_urn: str, text: str, asset_urn: str, title: str = ""
    ) -> dict:
        return {"id": f"urn:li:ugcPost:mock-{uuid.uuid4().hex[:8]}"}
