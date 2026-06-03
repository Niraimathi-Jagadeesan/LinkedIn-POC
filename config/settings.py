from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LinkedIn OAuth
    linkedin_client_id: str = ""
    linkedin_client_secret: str = ""
    linkedin_redirect_uri: str = "http://localhost:8000/callback"
    linkedin_access_token: str = ""

    # OpenAI (optional — only needed when LLM_PROVIDER=openai or IMAGE_PROVIDER=openai)
    openai_api_key: str = ""

    # Groq (free tier — https://console.groq.com)
    groq_api_key: str = ""

    # Ollama (local, fully offline — https://ollama.com)
    ollama_base_url: str = "http://localhost:11434/v1"

    # Provider selection
    llm_provider: str = "openai"      # openai | groq | ollama
    image_provider: str = "openai"    # openai | pollinations
    embedding_provider: str = "openai" # openai | local

    # Agent config
    llm_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    image_model: str = "dall-e-3"
    image_size: str = "1024x1024"
    chroma_persist_dir: str = "./data/chromadb"

    # Post visibility — controls who sees the post on LinkedIn
    # PUBLIC      → everyone (default for production)
    # CONNECTIONS → only your connections (safe for personal account testing)
    # LOGGED_IN   → any LinkedIn member, not indexed publicly
    post_visibility: str = "PUBLIC"

    # Testing
    dry_run: bool = False         # DRY_RUN=true  → generate content but don't post
    mock_linkedin: bool = False   # MOCK_LINKEDIN=true → no LinkedIn credentials needed

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
