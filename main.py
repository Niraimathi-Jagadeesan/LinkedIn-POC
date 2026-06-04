"""
LinkedIn AI Content Agent — Main Orchestrator
=============================================
Uses LangGraph to wire together 5 pipeline steps:

  retrieve_context → generate_content → [create_visual] → human_review → publish

  human_review has a conditional edge: approve → publish
                                       regenerate → generate_content (loop)
                                       cancel → END
"""

import uuid
from typing import List, Optional

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from agents.content_agent import ContentAgent
from agents.design_agent import DesignAgent
from agents.publisher_agent import PublisherAgent
from agents.rag_agent import RAGAgent
from config.settings import get_settings
from embeddings.vectorstore import LinkedInVectorStore
from linkedin.api_client import LinkedInAPIClient
from linkedin.auth import get_access_token
from linkedin.mock_client import MockLinkedInAPIClient, MOCK_PERSON_URN
from linkedin.models import GeneratedContent, PostFormat


def _get_api_client(token: str | None = None):
    """Return real or mock LinkedIn client based on settings."""
    if get_settings().mock_linkedin:
        return MockLinkedInAPIClient()
    return LinkedInAPIClient(token)


# ── State schema ─────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # Inputs
    topic: str
    user_context: str
    post_format: str          # PostFormat value string
    person_urn: str

    # Pipeline data
    past_posts_context: str
    generated_text: str
    hashtags: List[str]
    image_prompt: str
    hook_line: Optional[str]          # first sentence, overlaid on the image
    flyer_headline: Optional[str]
    flyer_subtitle: Optional[str]
    slides: Optional[List[dict]]   # carousel slides

    # Generated asset paths
    image_path: Optional[str]
    flyer_path: Optional[str]
    carousel_path: Optional[str]

    # Control flow
    review_action: Optional[str]   # "approve" | "regenerate" | "cancel"
    approved: bool

    # Output
    post_result: Optional[dict]
    error: Optional[str]


# ── Node functions ────────────────────────────────────────────────────────────

def retrieve_context(state: AgentState) -> AgentState:
    settings = get_settings()
    print("\n[1/5] Reading your past LinkedIn posts for context...")
    if settings.mock_linkedin:
        token = None
    else:
        token = get_access_token()
    api_client = _get_api_client(token)
    vectorstore = LinkedInVectorStore()
    rag = RAGAgent(api_client, vectorstore)

    if vectorstore.count() == 0:
        rag.index_past_posts(state["person_urn"])

    context = rag.retrieve_context(state["topic"], state["user_context"])
    return {**state, "past_posts_context": context}


def generate_content(state: AgentState) -> AgentState:
    print("\n[2/5] Generating content with AI (GPT-4o)...")
    agent = ContentAgent()
    fmt = PostFormat(state["post_format"])
    result = agent.generate_post(
        topic=state["topic"],
        user_context=state["user_context"],
        past_posts_context=state["past_posts_context"],
        post_format=fmt,
    )
    return {
        **state,
        "generated_text": result["text"],
        "hashtags": result["hashtags"],
        "image_prompt": result["image_prompt"],
        "hook_line": result.get("hook_line", ""),
        "flyer_headline": result.get("flyer_headline", state["topic"]),
        "flyer_subtitle": result.get("flyer_subtitle", ""),
        "slides": result.get("slides", []),
        # Reset review state on (re)generation
        "review_action": None,
        "approved": False,
    }


def create_visual(state: AgentState) -> AgentState:
    fmt = PostFormat(state["post_format"])
    agent = DesignAgent()

    if fmt == PostFormat.IMAGE:
        print("\n[3/5] Generating image with FLUX.1-schnell...")
        path = agent.generate_image(state["image_prompt"])
        return {**state, "image_path": path}

    if fmt == PostFormat.FLYER:
        print("\n[3/5] Designing flyer graphic...")
        path = agent.generate_flyer(
            headline=state.get("flyer_headline") or state["topic"],
            subtitle=state.get("flyer_subtitle") or "",
            topic=state["topic"],
        )
        return {**state, "flyer_path": path}

    if fmt == PostFormat.CAROUSEL:
        print("\n[3/5] Building carousel slides...")
        slides = state.get("slides") or []
        if not slides:
            # Fallback: split text into 5 chunks
            words = state["generated_text"].split()
            chunk = max(1, len(words) // 5)
            slides = [
                {"title": f"Point {i+1}", "body": " ".join(words[i*chunk:(i+1)*chunk])}
                for i in range(5)
            ]
        path = agent.generate_carousel(slides, state["topic"])
        return {**state, "carousel_path": path}

    return state  # TEXT — nothing to generate


def human_review(state: AgentState) -> AgentState:
    sep = "=" * 60
    print(f"\n{sep}")
    print("[4/5] REVIEW GENERATED CONTENT")
    print(sep)
    print(f"\nFormat : {state['post_format'].upper()}")
    print(f"\nPost text:\n{state['generated_text']}")
    print(f"\nHashtags: {' '.join(state['hashtags'])}")

    if state.get("image_path"):
        print(f"\nImage   : {state['image_path']}")
    if state.get("flyer_path"):
        print(f"\nFlyer   : {state['flyer_path']}")
    if state.get("carousel_path"):
        print(f"\nCarousel: {state['carousel_path']}")

    print(f"\n{'-' * 60}")
    while True:
        choice = (
            input("Post this to LinkedIn? [yes / no / regenerate]: ")
            .strip()
            .lower()
        )
        if choice in ("yes", "y"):
            return {**state, "review_action": "approve", "approved": True}
        if choice in ("no", "n"):
            return {**state, "review_action": "cancel", "approved": False}
        if choice in ("regenerate", "r", "regen"):
            print("Re-generating content...")
            return {**state, "review_action": "regenerate", "approved": False}
        print("Please enter 'yes', 'no', or 'regenerate'.")


def publish(state: AgentState) -> AgentState:
    if not state.get("approved"):
        print("\nPost cancelled — nothing was published.")
        return {**state, "post_result": None}

    settings = get_settings()
    if settings.mock_linkedin:
        label = "[MOCK] Simulating LinkedIn publish..."
    elif settings.dry_run:
        label = "[DRY RUN] Simulating publish (no post sent)"
    else:
        label = "[5/5] Publishing to LinkedIn..."
    print(f"\n{label}")
    token = None if settings.mock_linkedin else get_access_token()
    api_client = _get_api_client(token)
    publisher = PublisherAgent(api_client)

    content = GeneratedContent(
        text=state["generated_text"],
        hashtags=state["hashtags"],
        image_prompt=state["image_prompt"],
        topic=state["topic"],
        format=PostFormat(state["post_format"]),
        flyer_headline=state.get("flyer_headline"),
        flyer_subtitle=state.get("flyer_subtitle"),
        slides=state.get("slides"),
        image_path=state.get("image_path"),
        flyer_path=state.get("flyer_path"),
        carousel_path=state.get("carousel_path"),
    )

    result = publisher.publish(state["person_urn"], content)

    # Index the newly published post for future RAG context
    vs = LinkedInVectorStore()
    post_id = result.get("id", str(uuid.uuid4()))
    full_text = f"{state['generated_text']}\n\n{' '.join(state['hashtags'])}"
    vs.add_text(full_text, post_id, {"author": state["person_urn"], "created": ""})

    if get_settings().dry_run:
        print(f"\n[DRY RUN] Pipeline complete. Review the saved JSON to verify output.")
    else:
        print(f"\nSuccessfully posted to LinkedIn!")
        print(f"Post ID: {result.get('id', 'unknown')}")
    return {**state, "post_result": result}


# ── Conditional edge routers ──────────────────────────────────────────────────

def _route_visual(state: AgentState) -> str:
    """After generate_content: skip visual step for plain text posts."""
    if state["post_format"] in ("image", "flyer", "carousel"):
        return "create_visual"
    return "human_review"


def _route_review(state: AgentState) -> str:
    """After human_review: approve → publish, regenerate → loop, cancel → end."""
    action = state.get("review_action", "cancel")
    if action == "approve":
        return "publish"
    if action == "regenerate":
        return "generate_content"
    return END


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph():
    wf = StateGraph(AgentState)

    wf.add_node("retrieve_context", retrieve_context)
    wf.add_node("generate_content", generate_content)
    wf.add_node("create_visual", create_visual)
    wf.add_node("human_review", human_review)
    wf.add_node("publish", publish)

    wf.set_entry_point("retrieve_context")
    wf.add_edge("retrieve_context", "generate_content")

    wf.add_conditional_edges(
        "generate_content",
        _route_visual,
        {"create_visual": "create_visual", "human_review": "human_review"},
    )
    wf.add_edge("create_visual", "human_review")

    wf.add_conditional_edges(
        "human_review",
        _route_review,
        {
            "publish": "publish",
            "generate_content": "generate_content",
            END: END,
        },
    )
    wf.add_edge("publish", END)

    return wf.compile()


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("       LinkedIn AI Content Agent")
    print("=" * 60)

    settings = get_settings()
    get_settings.cache_clear()
    settings = get_settings()

    # Validate that the configured LLM provider has a key
    provider_key_missing = (
        (settings.llm_provider == "openai" and not settings.openai_api_key)
        or (settings.llm_provider == "groq" and not settings.groq_api_key)
    )
    if provider_key_missing:
        key_name = "OPENAI_API_KEY" if settings.llm_provider == "openai" else "GROQ_API_KEY"
        print(f"\n{key_name} is missing for provider '{settings.llm_provider}'. Please add it to your .env file.")
        return

    if settings.mock_linkedin:
        print("\n[MOCK MODE] LinkedIn is fully mocked — no LinkedIn credentials needed.")
        print("[MOCK MODE] Using 5 built-in sample posts as your post history.")
        api_client = MockLinkedInAPIClient()
        person_urn = MOCK_PERSON_URN
        name = "Test User (Mock)"
    else:
        if not settings.linkedin_client_id:
            print(
                "\nLinkedIn credentials missing.\n"
                "Set MOCK_LINKEDIN=true in .env to test without a LinkedIn app,\n"
                "or add LINKEDIN_CLIENT_ID and LINKEDIN_CLIENT_SECRET."
            )
            return
        print("\nAuthenticating with LinkedIn...")
        token = get_access_token()
        api_client = LinkedInAPIClient(token)
        try:
            profile = api_client.get_profile()
            # OIDC userinfo returns 'sub' as member ID; 'name' is the full name
            person_urn = f"urn:li:person:{profile['sub']}"
            name = profile.get("name", "").strip()
        except Exception as exc:
            print(f"Failed to retrieve LinkedIn profile: {exc}")
            return

    print(f"Logged in as: {name}")

    # Gather user input
    print("\n" + "-" * 60)
    topic = input("What topic do you want to post about? ").strip()
    if not topic:
        print("Topic cannot be empty.")
        return

    user_context = input(
        "Any additional context or key points to include? (press Enter to skip): "
    ).strip()

    print("\nSelect post format:")
    print("  1. text      — Text-only post")
    print("  2. image     — Post with AI-generated image")
    print("  3. flyer     — Post with a designed graphic card")
    print("  4. carousel  — Multi-slide carousel (PDF)")

    fmt_map = {"1": "text", "2": "image", "3": "flyer", "4": "carousel"}
    while True:
        choice = input("\nEnter choice (1-4): ").strip()
        if choice in fmt_map:
            post_format = fmt_map[choice]
            break
        if choice in fmt_map.values():
            post_format = choice
            break
        print("Please enter a number between 1 and 4.")

    # Run the agent graph
    app = build_graph()
    initial: AgentState = {
        "topic": topic,
        "user_context": user_context,
        "post_format": post_format,
        "person_urn": person_urn,
        "past_posts_context": "",
        "generated_text": "",
        "hashtags": [],
        "image_prompt": "",
        "hook_line": None,
        "flyer_headline": None,
        "flyer_subtitle": None,
        "slides": None,
        "image_path": None,
        "flyer_path": None,
        "carousel_path": None,
        "review_action": None,
        "approved": False,
        "post_result": None,
        "error": None,
    }

    try:
        final = app.invoke(initial)
        if final.get("post_result"):
            print("\nDone! Your post is now live on LinkedIn.")
        else:
            print("\nSession ended — no post was published.")
    except KeyboardInterrupt:
        print("\n\nCancelled.")
    except Exception as exc:
        print(f"\nUnexpected error: {exc}")
        raise


if __name__ == "__main__":
    main()
