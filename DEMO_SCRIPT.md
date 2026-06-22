# LinkedIn AI Content Agent — Team Demo Script

---

## PART 1 — OPENING: What Problem Does This Solve? (2 min)

**Say this:**
> "How much time does it take you to write one good LinkedIn post? Most people say 30–60 minutes — researching, drafting, formatting, finding an image, deciding when to post. Now multiply that by the 2–3 posts per week that LinkedIn's algorithm rewards. That's 2–3 hours every week just on social content.
>
> What we built is an AI pipeline that takes **one sentence from you** — your topic — and produces a complete, publish-ready LinkedIn post in under 2 minutes. It writes in YOUR voice, generates professional infographic visuals, and can post directly to LinkedIn. No manual formatting, no image tools, no context switching."

**Key value points to emphasize:**
- Learns your personal writing voice from your past posts
- Generates 4 post formats: Text, Image, Flyer, Carousel PDF
- One-click publish to LinkedIn with human review checkpoint
- Fully local + free AI options (no OpenAI required)

---

## PART 2 — ARCHITECTURE OVERVIEW (3 min)

**Draw or show this pipeline on screen:**

```
[ Browser UI ] 
      │
      ▼
[ FastAPI + Uvicorn server ]  ← ui_app.py (port 8080)
      │
      ▼
[ LangGraph Orchestrator ]  ← main.py
      │
      ├──► Step 1: RAG Agent        → fetch + index past LinkedIn posts
      ├──► Step 2: Content Agent    → generate post text + hashtags
      ├──► Step 3: Design Agent     → generate infographic visuals
      ├──► Step 4: Human Review     → you approve / regenerate / cancel
      └──► Step 5: Publisher Agent  → post to LinkedIn via API
```

**Say this:**
> "The whole pipeline is orchestrated by LangGraph — a state machine framework built on top of LangChain. Each step is an independent AI agent with a specific job. The state object carries data between agents — the topic goes in, the finished post and image come out. There is a mandatory human review step before anything gets posted — the AI never posts without your approval."

---

## PART 3 — THE AGENTS EXPLAINED (5 min)

### Agent 1: RAG Agent (`agents/rag_agent.py`)
**What it does:**
- Calls the LinkedIn API to fetch your last 50 posts
- Embeds them into **ChromaDB** (a local vector database) using sentence-transformer embeddings
- Retrieves your most recent post and the 3 most similar past posts to the current topic

**Why it matters:**
> "This is the personalisation engine. Without this, the AI generates generic corporate content. With it, the AI studies HOW you write — your sentence length, your tone, your vocabulary — and mirrors it. This is called Retrieval-Augmented Generation (RAG)."

**Technical detail:**
- Embeddings: `text-embedding-3-small` (OpenAI) or `all-MiniLM-L6-v2` (local, free)
- Vector DB: ChromaDB with local persistence at `./data/chromadb/`
- Similarity search: cosine distance, top-3 results

---

### Agent 2: Content Agent (`agents/content_agent.py`)
**What it does:**
- Takes: topic + user context + past post style + tone profile
- Outputs: post text (300–400 words, ALL-CAPS section headings), hashtags, image prompt, hook line, slide content

**The system prompt (Rule 9):**
> "Structure every post as a LinkedIn article with clear visual hierarchy. Open with a ONE-LINE HOOK. Follow with 3–4 section blocks each starting with a SHORT HEADING in ALL CAPS. Never use markdown."

**LLM Provider flexibility — why this matters:**
| Provider | Model | Cost | Speed |
|----------|-------|------|-------|
| OpenAI | GPT-4o | Paid | Fast |
| Groq | LLaMA 3.3 70B | **FREE** | Very fast |
| Google Gemini | Gemini 2.5 Flash | Free tier | Fast |
| Ollama | Any local model | **FREE + offline** | Depends |

**Say this:**
> "We designed the content agent to be provider-agnostic. The same OpenAI-compatible API interface works for Groq, Ollama, and OpenAI. Switch providers by changing one line in the `.env` file — `LLM_PROVIDER=groq`. This means you can run the entire application for FREE using Groq's free tier."

---

### Agent 3: Design Agent (`agents/design_agent.py`)
**What it does — 4 post formats:**

| Format | Pipeline |
|--------|----------|
| **Text** | No visual — post text only |
| **Image** | HTML infographic → Edge headless screenshot → 1080×1080 PNG |
| **Flyer** | HTML landscape infographic → Edge headless → 1200×627 PNG |
| **Carousel** | Gemini batch-expands 6 slides → HTML per slide → Edge screenshot → PDF |

**The carousel pipeline in detail:**
```
Topic
  │
  ▼
Gemini 2.5 Flash (1 API call)
  → Expands ALL 6 slide briefs at once (batch)
  │
  ▼
For each of 6 slides:
  Gemini 2.5 Flash → generates full HTML/CSS (ASG-style infographic)
  Edge headless browser → screenshots HTML at 1080×1080 → PNG
  │
  ▼
fpdf2 → assembles 6 PNGs into a PDF carousel
```

**Why HTML + Edge browser instead of image generation?**
> "Text-to-image models like DALL-E or FLUX cannot reliably render readable text inside images. We tried — the results were blurry and the text was garbled. The breakthrough was generating the slide as HTML/CSS, then using Microsoft Edge in headless mode to take a pixel-perfect screenshot. This gives us designer-quality typography, proper layout, and 100% readable text — every time."

**Fallback chain (resilience):**
```
Gemini HTML → fails? → Local Python HTML renderer → Edge screenshot
Edge browser → fails? → Pillow (offline image)
```

---

### Agent 4: Publisher Agent (`agents/publisher_agent.py`)
**What it does:**
- Uploads image/PDF to LinkedIn's media servers (gets an `asset_urn`)
- Creates the UGC post via LinkedIn REST API
- Supports `DRY_RUN=true` mode — saves everything locally without posting
- Controls post visibility: `PUBLIC`, `CONNECTIONS`, or `LOGGED_IN`

**Say this:**
> "Before anything goes live, the Human Review step shows you the full post text, hashtags, and the visual. You can approve, regenerate (loops back to the content agent), or cancel. Nothing posts without your explicit approval."

---

## PART 4 — PYTHON SETUP & DEPENDENCIES (3 min)

### Project structure:
```
LinkedIn-POC/
├── main.py              # LangGraph orchestrator
├── ui_app.py            # FastAPI server + SSE event stream
├── agents/
│   ├── content_agent.py # LLM post generation
│   ├── design_agent.py  # Visual generation
│   ├── publisher_agent.py # LinkedIn posting
│   └── rag_agent.py     # ChromaDB + past post retrieval
├── linkedin/
│   ├── api_client.py    # LinkedIn REST API calls
│   ├── auth.py          # OAuth 2.0 flow
│   └── models.py        # Data models
├── config/settings.py   # All env vars via Pydantic Settings
├── embeddings/vectorstore.py # ChromaDB wrapper
└── static/index.html    # Frontend (Tailwind CSS)
```

### Key Python libraries (`requirements.txt`):
| Library | Purpose |
|---------|---------|
| `langgraph` | State machine orchestration of agents |
| `langchain-core` | Base abstractions for LLM chains |
| `openai` | OpenAI SDK (also used for Groq/Ollama via compatible API) |
| `google-genai` | Google Gemini SDK (HTML generation, content expansion) |
| `chromadb` | Local vector database for RAG |
| `fastapi + uvicorn` | REST API server + Server-Sent Events |
| `fpdf2` | Assemble PNG slides into PDF carousel |
| `Pillow` | Image fallback renderer |
| `pydantic-settings` | Type-safe `.env` file loading |
| `python-dotenv` | Environment variable management |

### Setup commands:
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Fill in .env with your API keys
.\.venv\Scripts\python.exe ui_app.py
# Open http://localhost:8080
```

---

## PART 5 — LINKEDIN APP SETUP (4 min)

### Step 1: Create the LinkedIn App
1. Go to https://developer.linkedin.com/
2. Click **"Create App"**
3. Fill in: App Name, LinkedIn Page, Logo
4. Under **"Auth"** tab, add Redirect URL: `http://localhost:8000/callback`
5. Copy **Client ID** and **Client Secret** → paste into `.env`

### Step 2: Add Products (CRITICAL)

Go to your app → **"Products"** tab → Request access to these:

| Product | Scopes granted | Used for |
|---------|---------------|----------|
| **Sign In with LinkedIn using OpenID Connect** | `openid`, `profile`, `email` | Get your profile + person URN |
| **Share on LinkedIn** | `w_member_social` | Create posts, upload images/documents |

**These two products are what the app currently uses.**

### Step 3: The Missing Product — Reading Your Posts

**The Problem:** When you see this warning in the terminal:
```
[WARN] Could not fetch posts from LinkedIn API: 403 Client Error: Forbidden
```
This means the app cannot fetch your past posts to learn your writing style.

**Why it fails:** Reading posts requires the `r_member_social` scope. This scope is **not included** in the standard "Share on LinkedIn" product — LinkedIn restricts it.

**The Solution:** In your LinkedIn app → Products → Request **"Share on LinkedIn"** (if not already added), then go to **"OAuth 2.0 Scopes"** and look for `r_member_social`. LinkedIn grants this scope to apps that specifically request it AND have the Share product enabled.

**What to change in code after approval:**
In `linkedin/auth.py`, line:
```python
_SCOPES = "openid profile email w_member_social"
```
Change to:
```python
_SCOPES = "openid profile email w_member_social r_member_social"
```
Then re-run the OAuth flow (`python linkedin/auth.py`) to get a new token with the read scope.

**Workaround in the meantime:** The RAG agent already handles this gracefully — it falls back to ChromaDB (previously indexed posts). Once you've posted at least once, ChromaDB remembers your style.

### Step 4: OAuth Flow
Run once to authenticate:
```bash
.\.venv\Scripts\python.exe linkedin/auth.py
```
- Opens browser → LinkedIn login → Redirects to `localhost:8000/callback`
- Token is automatically saved to `.env` as `LINKEDIN_ACCESS_TOKEN`

---

## PART 6 — LIVE DEMO WALKTHROUGH (5 min)

**Open http://localhost:8080**

### Demo Run 1: Carousel (most impressive)
1. Topic: `"How AI is transforming prior authorization in healthcare"`
2. Format: **Carousel**
3. Click **Generate Post**
4. Show the pipeline steps progressing in real time (SSE events)
5. Point out: "Step 1 fetches your past posts. Step 2 analyzes your writing tone. Step 3 generates the content. Step 4 builds 6 HTML slides and screenshots them with Edge. All automatic."
6. Review panel appears → show the post text + PDF preview
7. Approve → posts to LinkedIn (or show dry-run output)

### Demo Run 2: AI Image (standalone infographic)
1. Topic: `"3 reasons teams fail at digital transformation"`
2. Format: **AI Image**
3. Show the result: hook quote banner + 3 content sections + no "Swipe to explore"
4. Point out: "This is a custom HTML layout generated by Gemini, photographed by Edge browser. Not a stock image, not a generic AI photo — a readable, information-rich infographic."

### Demo Run 3: Flyer (landscape)
1. Topic: `"Why Python is the language of AI"`
2. Format: **Flyer**
3. Show 1200×627 landscape layout with navy left panel + white right panel

---

## PART 7 — ENHANCEMENT ROADMAP (3 min)

### Short-term (1–2 weeks)
| Enhancement | Impact |
|-------------|--------|
| **Scheduling** — add `post_at` datetime, use APScheduler | Post at optimal times (Tue–Thu 8–10am) |
| **Multi-topic queue** — batch generate 5 posts at once | Build a week's content in 10 min |
| **LinkedIn analytics pull** — fetch impressions/reactions | Learn what topics perform best |
| **Template gallery** — save/reuse slide layouts | Brand consistency across posts |

### Medium-term (1 month)
| Enhancement | Impact |
|-------------|--------|
| **Multi-account support** — team members each authenticate | Agency or team use case |
| **Tone profiles** — save "formal", "casual", "technical" presets | Consistent voice per audience |
| **Content calendar** — visual planner with drag-drop | Replace tools like Buffer/Hootsuite |
| **A/B variant generation** — generate 2 versions, pick best | Data-driven content decisions |

### Long-term (2–3 months)
| Enhancement | Impact |
|-------------|--------|
| **Auto-posting with performance loop** — post → collect metrics → improve prompts | Self-optimising pipeline |
| **Competitor monitoring** — RAG over industry posts | Stay relevant to trending topics |
| **Video script generation** — extend to short-form video content | LinkedIn video is high-algorithm reach |
| **Multi-platform export** — same content adapted for Twitter/X, Instagram | Single source, multiple channels |

---

## PART 8 — Q&A PREP

**Q: What does this cost to run?**
> Free option: `LLM_PROVIDER=groq` (LLaMA 3.3 70B, free) + `IMAGE_PROVIDER=gemini` (Gemini 2.5 Flash free tier, 50 req/day). The only paid dependency is OpenAI, which is optional.

**Q: Is our data sent to AI providers?**
> Post content and topic are sent to whichever LLM you configure. With `LLM_PROVIDER=ollama` the model runs 100% locally — nothing leaves your machine. ChromaDB is always local.

**Q: Can multiple people use it?**
> Currently single-user (one LinkedIn token). Adding multi-user support requires a database for token storage — a 1-day addition.

**Q: How is this different from tools like Taplio or Jasper?**
> Three key differences: (1) RAG on YOUR OWN past posts — not generic fine-tuning. (2) Professional infographic carousels — not just text. (3) Direct LinkedIn API publish — no copy-paste. And it's fully custom code you own.

**Q: What's LangGraph vs LangChain?**
> LangChain provides building blocks (LLM calls, prompt templates). LangGraph adds a state machine layer on top — you define nodes (agents) and edges (transitions including conditionals and loops). The human review loop — where you can regenerate — is a LangGraph conditional edge.

---

## QUICK REFERENCE: .env Keys

```env
# LinkedIn
LINKEDIN_CLIENT_ID=your_app_client_id
LINKEDIN_CLIENT_SECRET=your_app_client_secret
LINKEDIN_ACCESS_TOKEN=auto_filled_by_auth.py

# LLM (pick one)
LLM_PROVIDER=groq              # groq = FREE
LLM_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=gsk_...

# Image (Gemini for infographic slides)
IMAGE_PROVIDER=gemini
GEMINI_API_KEY=...

# Behaviour
MOCK_LINKEDIN=false            # true = don't actually post
DRY_RUN=false                  # true = save output locally, no API calls
POST_VISIBILITY=CONNECTIONS    # CONNECTIONS for safe testing
EMBEDDING_PROVIDER=local       # local = no OpenAI key needed for embeddings
```
