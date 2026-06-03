"""
Seed your past LinkedIn posts into ChromaDB for RAG context.

Usage:
  python seed_posts.py

Add your real post text to the POSTS list below, then run this script once.
The agent will use these posts as context when generating new content.
"""

import uuid
from embeddings.vectorstore import LinkedInVectorStore

# ── Paste your real LinkedIn posts here ───────────────────────────────────────
# Each entry is one post. Copy the full text (no hashtags needed separately).
POSTS = [
    """
    AI in enterprise applications stops being just a buzzword the moment you have to ship it, optimize the token cost, and explain the value to real users and clients.
 
    Had a great opportunity to conduct an AI knowledge-sharing session for my colleagues at the @Kanini Coimbatore location, where I demonstrated our AI POC and discussed some of the real engineering decisions behind building scalable AI-driven solutions.
    
    Some of the topics we explored during the session:
    🔹 RAG (Retrieval-Augmented Generation) concepts and how they improve enterprise AI reliability
    🔹 Working with the Gemini model and practical integration approaches
    🔹 Few-shot prompting techniques to generate more reliable and production-ready AI responses
    🔹 Token optimization strategies using TOON, a compact alternative to JSON that helps reduce token consumption significantly at scale
    🔹 Real-world challenges and learnings while taking AI features from POC to client-ready solutions
    
    One of the biggest learnings we discussed was that many “AI problems” are actually data structure, retrieval quality, and prompt-engineering problems. The model is important, but engineering discipline around cost optimization, reliability, and response quality is what truly makes enterprise AI solutions successful.
    
    It was encouraging to see the team actively engaging in discussions around practical AI adoption and scalable implementation strategies. 
    
    Sessions like these always create opportunities for collaborative learning and innovation.
    
    A big thank you to everyone who participated and made the session interactive and insightful. Looking forward to contributing more towards AI-driven engineering initiatives and continuous learning.
    
    Curious to know — how are others managing token optimization and reliability challenges in production AI applications?
    
    hashtag#GenAI hashtag#ArtificialIntelligence hashtag#RAG hashtag#PromptEngineering hashtag#AIEngineering hashtag#GeminiAI hashtag#ReactJS hashtag#EnterpriseAI hashtag#SoftwareEngineering hashtag#Kanini hashtag#KnowledgeSharing
    """,

    # Add as many as you like...
]
# ─────────────────────────────────────────────────────────────────────────────


def main():
    posts = [p.strip() for p in POSTS if p.strip()]
    if not posts:
        print("No posts found. Add your post text to the POSTS list in seed_posts.py.")
        return

    vs = LinkedInVectorStore()
    before = vs.count()

    for text in posts:
        post_id = f"manual-{uuid.uuid4().hex[:8]}"
        vs.add_text(text, post_id, {"source": "manual_seed"})
        print(f"  Indexed: {text[:80].replace(chr(10), ' ')}...")

    after = vs.count()
    print(f"\nDone. ChromaDB now has {after} posts (was {before}).")
    print("The agent will use these as RAG context on the next run.")


if __name__ == "__main__":
    main()
