"""
Design Agent — generates visuals for LinkedIn posts.
  • IMAGE   → DALL-E 3 (OpenAI) OR Pollinations.ai (free, no key)
  • FLYER   → Pillow-rendered branded graphic card (1200×627)
  • CAROUSEL → Pillow-rendered slides exported as a PDF (1080×1080 per slide)
"""

import platform
import unicodedata
import urllib.parse
from pathlib import Path
from typing import List

import requests
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

from config.settings import get_settings


class DesignAgent:
    # Five visual styles — cycles on each regeneration so every flyer looks different
    _FLYER_PALETTES = [
        {"panel": (8,  22, 50),  "top": (0,  119, 181), "bot": (255, 204,   0), "sub": (180, 215, 245)},
        {"panel": (45, 10, 28),  "top": (210,  30,  85), "bot": (255, 180,   0), "sub": (245, 180, 200)},
        {"panel": (12, 48, 22),  "top": (0,  165,  60), "bot": (155, 255,  80), "sub": (170, 240, 190)},
        {"panel": (55, 32,  5),  "top": (220, 130,   0), "bot": (255, 230,  60), "sub": (255, 220, 165)},
        {"panel": (32,  5, 58),  "top": (118,  40, 210), "bot": (210, 150, 255), "sub": (215, 185, 255)},
    ]
    _BG_MOODS = [
        "dark navy blue gradient, subtle geometric angular lines",
        "deep teal to dark plum gradient, flowing organic curves",
        "dark forest green gradient, abstract botanical shapes",
        "warm charcoal to dark amber gradient, concentric geometric rings",
        "deep indigo to midnight violet gradient, crystalline lattice patterns",
    ]

    def __init__(self) -> None:
        settings = get_settings()
        self._image_provider = settings.image_provider
        if settings.image_provider == "openai":
            self._client = OpenAI(api_key=settings.openai_api_key)
        if settings.image_provider == "gemini":
            from google import genai
            self._gemini_client = genai.Client(api_key=settings.gemini_api_key)
        self._image_model = settings.image_model
        self._image_size = settings.image_size
        self._out = Path("./data/outputs")
        self._out.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_image(
        self, image_prompt: str, filename: str = "post_image.png"
    ) -> str:
        """Generate a LinkedIn-ready image and save it to disk."""
        if self._image_provider == "gemini":
            return self._generate_image_gemini_imagen(image_prompt, filename)
        if self._image_provider == "huggingface":
            return self._generate_image_hf(image_prompt, filename)
        if self._image_provider == "pollinations":
            return self._generate_image_pollinations(image_prompt, filename)
        return self._generate_image_openai(image_prompt, filename)

    def generate_infographic_image(
        self,
        topic: str,
        hook_line: str = "",
        image_prompt: str = "",
        post_text: str = "",
        filename: str = "post_image.png",
    ) -> str:
        """
        Generate a 1080×1080 standalone infographic PNG for the IMAGE post format.
        Dedicated layout: navy header → hook quote banner → 3 content sections → footer.
        NO carousel references. Accommodates the full post content.
        Fallback chain: Gemini HTML → local HTML → Edge → Pillow.
        """
        print("  Generating standalone infographic image (HTML → Edge headless)...")

        # Step 1: Try Gemini for AI-quality HTML
        html = None
        if hasattr(self, "_gemini_client"):
            try:
                html = self._generate_standalone_image_html_with_gemini(
                    topic, hook_line, post_text
                )
            except Exception as exc:
                print(f"  [WARN] Gemini image HTML failed ({exc}). Using local renderer.")

        if html is None:
            html = self._generate_image_html_local(topic, hook_line, post_text)

        # Step 2: Screenshot with Edge
        try:
            return self._screenshot_html(html, filename, 1080, 1080)
        except Exception as exc:
            print(f"  [WARN] Edge screenshot failed ({exc}). Retrying with local HTML.")

        # Step 3: Local HTML + Edge retry (in case Gemini HTML was malformed)
        try:
            html_local = self._generate_image_html_local(topic, hook_line, post_text)
            return self._screenshot_html(html_local, filename, 1080, 1080)
        except Exception as exc2:
            print(f"  [WARN] Local HTML also failed ({exc2}). Using Pillow fallback.")
            return self._generate_image_pillow(image_prompt or topic, filename)

    def _generate_standalone_image_html_with_gemini(
        self,
        topic: str,
        hook_line: str = "",
        post_text: str = "",
    ) -> str:
        """
        Ask Gemini to generate a 1080×1080 standalone infographic HTML.
        Dedicated image layout — NOT a carousel slide.
        """
        from google.genai import types
        import re as _re
        import time as _time

        content = (post_text or hook_line or topic)[:1400]

        system = (
            "You are an expert HTML/CSS infographic designer.\n"
            "Generate a COMPLETE self-contained HTML for a 1080\u00d71080px standalone LinkedIn image.\n\n"
            "LAYOUT (flex column filling exactly 1080px height — NO scrollbars, NO gaps):\n"
            "1. HEADER (height:90px, flex-shrink:0): dark navy (#0D1F59), white title left, "
            "   #hashtag badge right in LinkedIn blue (#0077B5).\n"
            "2. HOOK BANNER (flex-shrink:0): light blue (#eef4ff), left border 6px solid #0077B5, "
            "   italic quote (font-size:20px, padding:22px 44px).\n"
            "3. CONTENT AREA (flex:1, overflow:hidden, padding:14px 44px, gap:12px): "
            "   3 sections each flex:1 — emoji icon + ALL-CAPS bold heading + body text. "
            "   Left border 4px solid #0D1F59, border-radius:12px.\n"
            "4. FOOTER (height:46px, flex-shrink:0): dark navy, hashtag left #0077B5, "
            "   'AI Infographic' right faded white.\n\n"
            "STRICT CSS RULES:\n"
            "- html, body { width:1080px; height:1080px; overflow:hidden; margin:0; padding:0 }\n"
            "- body { display:flex; flex-direction:column; background:#fff }\n"
            "- ::-webkit-scrollbar { display:none }\n"
            "- Footer is the LAST flex child — NOT position:absolute\n"
            "- Font: 'Segoe UI', Arial, sans-serif. No external resources.\n"
            "- NO 'Swipe to explore', NO slide numbers, NO carousel language.\n"
            "- Return ONLY raw HTML, no markdown fences."
        )

        user_msg = (
            f"Topic: {topic}\n"
            f"Hook line: {hook_line}\n"
            f"Full post content:\n{content}\n\n"
            "Generate the complete 1080×1080 standalone infographic. "
            "Extract 3 key insights from the content for the 3 sections. "
            "Make it look like a polished, information-rich LinkedIn image post."
        )

        for attempt in range(3):
            try:
                resp = self._gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=user_msg,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=0.3,
                    ),
                )
                html = resp.text.strip()
                if html.startswith("```"):
                    lines = html.splitlines()
                    lines = lines[1:]
                    if lines and lines[-1].strip().startswith("```"):
                        lines = lines[:-1]
                    html = "\n".join(lines)
                print(f"    Gemini standalone image HTML generated ({len(html)} chars).")
                return html.strip()
            except Exception as exc:
                retryable = "429" in str(exc) or "503" in str(exc) or "UNAVAILABLE" in str(exc)
                if attempt < 2 and retryable:
                    import re as _re2
                    import time as _time2
                    m = _re2.search(r'retry\s*in\s*([\d.]+)s', str(exc))
                    wait = int(float(m.group(1))) + 5 if m else 30
                    print(f"    Gemini busy. Waiting {wait}s (attempt {attempt + 2}/3)...")
                    _time2.sleep(wait)
                else:
                    raise

    def _generate_image_html_local(
        self,
        topic: str,
        hook_line: str = "",
        post_text: str = "",
    ) -> str:
        """
        Generate a 1080×1080 standalone infographic HTML entirely in Python.
        Layout: navy header → hook quote banner → 3 content section cards → navy footer.
        No carousel language. Accommodates full post content.
        """
        import re
        import html as _h

        title    = _h.escape(topic)
        tag      = _h.escape("#" + self._ascii_tag(topic))
        hook_esc = _h.escape(hook_line[:220]) if hook_line else _h.escape(topic)

        # Extract up to 6 key sentences from post_text
        raw_sentences = [
            s.strip() for s in re.split(r'[.;\n]+', post_text or "")
            if len(s.strip()) > 15
        ]

        section_labels = ["KEY INSIGHT", "WHAT THIS MEANS", "THE OPPORTUNITY"]
        section_icons  = ["\U0001f4ca", "\u2699\ufe0f", "\U0001f3af"]

        sections_html = ""
        for i in range(3):
            body = _h.escape(raw_sentences[i][:180]) if i < len(raw_sentences) else _h.escape(topic)
            sections_html += (
                f'<div class="sec">'
                f'<div class="sec-ic">{section_icons[i]}</div>'
                f'<div class="sec-ct">'
                f'<div class="sec-hd">{section_labels[i]}</div>'
                f'<div class="sec-bd">{body}</div>'
                f'</div></div>'
            )

        return (
            "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><style>"
            "*{box-sizing:border-box;margin:0;padding:0}"
            "html{width:1080px;height:1080px;overflow:hidden}"
            "body{width:1080px;height:1080px;overflow:hidden;"
            "font-family:'Segoe UI',Arial,sans-serif;background:#fff;"
            "display:flex;flex-direction:column}"
            "::-webkit-scrollbar{display:none}"
            ".hdr{background:#0D1F59;height:90px;flex-shrink:0;display:flex;align-items:center;"
            "justify-content:space-between;padding:0 44px;border-bottom:5px solid #0077B5}"
            ".hdr h1{color:#fff;font-size:27px;font-weight:800;line-height:1.2;max-width:740px}"
            ".badge{background:#0077B5;color:#fff;font-size:14px;font-weight:700;"
            "padding:7px 18px;border-radius:20px;white-space:nowrap;flex-shrink:0}"
            ".hook{background:#eef4ff;padding:22px 44px;flex-shrink:0;"
            "border-left:6px solid #0077B5}"
            ".hook p{font-size:20px;font-style:italic;color:#1a2a5e;"
            "line-height:1.5;font-weight:500;max-width:950px}"
            ".content{flex:1;padding:14px 44px;display:flex;flex-direction:column;"
            "gap:12px;overflow:hidden}"
            ".sec{flex:1;display:flex;align-items:flex-start;gap:18px;padding:14px 20px;"
            "background:#fff;border:1px solid #dde5f5;border-left:4px solid #0D1F59;"
            "border-radius:12px;overflow:hidden}"
            ".sec-ic{font-size:28px;flex-shrink:0;margin-top:2px}"
            ".sec-ct{flex:1;overflow:hidden}"
            ".sec-hd{font-size:13px;font-weight:800;color:#0D1F59;margin-bottom:6px;"
            "text-transform:uppercase;letter-spacing:0.8px}"
            ".sec-bd{font-size:15px;color:#2a2a3e;line-height:1.55}"
            ".ftr{background:#0D1F59;height:46px;flex-shrink:0;display:flex;"
            "align-items:center;justify-content:space-between;"
            "padding:0 44px;border-top:3px solid #0077B5}"
            ".ftr-tag{color:#0077B5;font-size:15px;font-weight:700}"
            ".ftr-lbl{color:rgba(255,255,255,.35);font-size:13px}"
            f"</style></head><body>"
            f"<div class=\"hdr\"><h1>{title}</h1><span class=\"badge\">{tag}</span></div>"
            f"<div class=\"hook\"><p>&ldquo;{hook_esc}&rdquo;</p></div>"
            f"<div class=\"content\">{sections_html}</div>"
            f"<div class=\"ftr\"><span class=\"ftr-tag\">{tag}</span>"
            "<span class=\"ftr-lbl\">AI Infographic</span></div>"
            "</body></html>"
        )

    def _generate_image_gemini_imagen(self, image_prompt: str, filename: str) -> str:
        """
        Generate via Google Gemini Imagen 3 — produces high-quality infographic images
        that can accurately render text, charts, icons, and structured layouts.
        Falls back to Pillow-rendered image on any error.
        """
        try:
            result = self._gemini_client.models.generate_images(
                model="imagen-4.0-generate-001",
                prompt=image_prompt,
                config={
                    "number_of_images": 1,
                    "aspect_ratio": "1:1",
                    "safety_filter_level": "block_low_and_above",
                    "person_generation": "dont_allow",
                },
            )
            if result.generated_images:
                out_path = self._out / filename
                result.generated_images[0].image.save(str(out_path))
                print(f"  Gemini Imagen 3 image saved: {out_path}")
                return str(out_path)
            raise ValueError("No images returned from Gemini Imagen")
        except Exception as exc:
            print(f"  [WARN] Gemini Imagen generation failed ({exc}). Falling back to local image.")
            return self._generate_image_pillow(image_prompt, filename)

    def _generate_image_hf(self, image_prompt: str, filename: str) -> str:
        """
        Generate via Hugging Face Inference API — free with HF_API_TOKEN.
        Uses FLUX.1-schnell (state-of-the-art, fast, free on HF free tier).
        Falls back to Pillow-rendered image on any error.
        """
        from config.settings import get_settings
        settings = get_settings()
        token = settings.hf_api_token
        if not token:
            print("  [WARN] HF_API_TOKEN not set. Falling back to local image.")
            return self._generate_image_pillow(image_prompt, filename)

        full_prompt = (
            f"{image_prompt}. Professional, high-quality, suitable for LinkedIn. "
            "Clean modern composition, vibrant colors, photorealistic or digital art style."
        )
        api_url = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
        headers = {"Authorization": f"Bearer {token}"}
        payload = {"inputs": full_prompt, "parameters": {"width": 1024, "height": 1024}}

        print("  Generating image with FLUX.1-schnell (Hugging Face)...")
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            out_path = self._out / filename
            out_path.write_bytes(resp.content)
            print(f"  Image saved: {out_path}")
            return str(out_path)
        except Exception as exc:
            print(f"  [WARN] HuggingFace image generation failed ({exc}). Falling back to local image.")
            return self._generate_image_pillow(image_prompt, filename)

    def _generate_image_openai(
        self, image_prompt: str, filename: str
    ) -> str:
        """Generate via DALL-E 3."""
        full_prompt = (
            f"{image_prompt}. "
            "Professional, high-quality, suitable for LinkedIn. "
            "Clean composition, modern design, no text overlays."
        )
        resp = self._client.images.generate(
            model=self._image_model,
            prompt=full_prompt,
            size=self._image_size,
            quality="standard",
            n=1,
        )
        img_url = resp.data[0].url
        img_data = requests.get(img_url, timeout=30).content
        out_path = self._out / filename
        out_path.write_bytes(img_data)
        return str(out_path)

    def _generate_image_pollinations(
        self, image_prompt: str, filename: str
    ) -> str:
        """
        Generate via Pollinations.ai free tier (no key, no premium params).
        Falls back to a Pillow-rendered branded image on any HTTP error.
        """
        # Detect infographic context so the style suffix matches the content
        is_infographic = any(w in image_prompt.lower() for w in
                             ("infographic", "diagram", "chart", "vector", "icon", "flow", "architecture"))
        suffix = (
            "Professional infographic illustration, clean flat design, no text, no letters."
            if is_infographic
            else "Professional LinkedIn graphic, clean modern design."
        )
        encoded = urllib.parse.quote(f"{image_prompt}. {suffix}", safe="")
        url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&seed={abs(hash(image_prompt)) % 9999}"
        print(f"  Requesting image from Pollinations.ai ({'infographic' if is_infographic else 'standard'})...")
        try:
            resp = requests.get(url, timeout=90)
            resp.raise_for_status()
            out_path = self._out / filename
            out_path.write_bytes(resp.content)
            return str(out_path)
        except Exception as exc:
            print(f"  [WARN] Pollinations.ai unavailable ({exc}). Generating branded image locally.")
            return self._generate_image_pillow(image_prompt, filename)

    def add_text_overlay(self, image_path: str, hook_text: str) -> str:
        """
        Overlay the hook sentence on the bottom of an AI-generated image.
        Adds a semi-transparent dark gradient banner with white bold text.
        Returns the path to the modified image (overwrites in-place).
        """
        img = Image.open(image_path).convert("RGBA")
        W, H = img.size

        # Dark gradient overlay at the bottom third
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw_ov = ImageDraw.Draw(overlay)

        banner_h = H // 3
        for i in range(banner_h):
            alpha = int(210 * (i / banner_h))          # 0 → 210 (top → bottom)
            draw_ov.rectangle([(0, H - banner_h + i), (W, H - banner_h + i + 1)],
                              fill=(10, 36, 66, alpha))

        # LinkedIn blue accent line above the text
        draw_ov.rectangle([(0, H - banner_h), (W, H - banner_h + 5)],
                          fill=(0, 119, 181, 220))

        composite = Image.alpha_composite(img, overlay).convert("RGB")
        draw = ImageDraw.Draw(composite)

        # Word-wrap the hook text
        font = self._font(min(60, max(36, W // 18)))
        words = hook_text.split()
        lines, line = [], []
        for w in words:
            trial = " ".join(line + [w])
            bb = draw.textbbox((0, 0), trial, font=font)
            if bb[2] - bb[0] > W - 80 and line:
                lines.append(" ".join(line))
                line = [w]
            else:
                line.append(w)
        if line:
            lines.append(" ".join(line))

        line_h = font.size + 12
        total_text_h = len(lines) * line_h
        y = H - (banner_h // 2) - (total_text_h // 2)

        for ln in lines:
            bb = draw.textbbox((0, 0), ln, font=font)
            x = (W - (bb[2] - bb[0])) // 2
            # Shadow
            draw.text((x + 2, y + 2), ln, font=font, fill=(0, 0, 0, 160))
            draw.text((x, y), ln, font=font, fill=(255, 255, 255))
            y += line_h

        composite.save(image_path, "PNG")
        return image_path

    def _generate_image_pillow(
        self, image_prompt: str, filename: str
    ) -> str:
        """Render a branded 1024×1024 image using Pillow (offline fallback)."""
        W, H = 1024, 1024
        BG      = (15, 76, 129)
        ACCENT  = (0, 119, 181)
        WHITE   = (255, 255, 255)
        YELLOW  = (255, 204, 0)

        img  = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(img)

        # Gradient-ish accent bar at top and bottom
        draw.rectangle([(0, 0), (W, 12)], fill=YELLOW)
        draw.rectangle([(0, H - 12), (W, H)], fill=YELLOW)

        # Centre circle decoration
        draw.ellipse([(W // 2 - 200, H // 2 - 200), (W // 2 + 200, H // 2 + 200)],
                     fill=ACCENT)

        # Prompt text wrapped in the centre
        font_title = self._font(52)
        font_small = self._font(28)
        words  = image_prompt.split()
        lines, line = [], []
        for w in words:
            line.append(w)
            if len(" ".join(line)) > 26:
                lines.append(" ".join(line[:-1]))
                line = [w]
        if line:
            lines.append(" ".join(line))
        lines = lines[:5]

        total_h = len(lines) * 64
        y = H // 2 - total_h // 2
        for ln in lines:
            bbox = draw.textbbox((0, 0), ln, font=font_title)
            x = (W - (bbox[2] - bbox[0])) // 2
            draw.text((x + 2, y + 2), ln, font=font_title, fill=(0, 0, 0, 120))
            draw.text((x, y), ln, font=font_title, fill=WHITE)
            y += 64

        # "AI Generated" watermark at bottom
        draw.text((W // 2 - 80, H - 50), "AI Generated", font=font_small, fill=YELLOW)

        out_path = self._out / filename
        img.save(str(out_path), "PNG")
        return str(out_path)

    def generate_flyer(
        self,
        headline: str,
        subtitle: str,
        topic: str,
        body_text: str = "",
        filename: str = "flyer.png",
    ) -> str:
        """
        AI-powered flyer (1200×627):
          1. FLUX generates a clean abstract background — NO topic words in prompt
             so no stray text appears in the AI image.
          2. Pillow draws a solid opaque text panel on the left half.
          3. Headline phrase + subtitle from the AI content are overlaid cleanly.
        """
        W, H = 1200, 627

        # ── Step 1: Pick a visual style palette then match background mood ────
        pal_idx = abs(hash(headline + body_text + subtitle)) % len(self._FLYER_PALETTES)
        pal     = self._FLYER_PALETTES[pal_idx]
        ACCENT  = pal["top"]
        BOT     = pal["bot"]
        SUB_COL = pal["sub"]

        bg_prompt = (
            f"Abstract professional background, {self._BG_MOODS[pal_idx]}, "
            "soft bokeh light orbs, cinematic depth of field, "
            "clean minimal corporate aesthetic, no text, no letters, no words, "
            "no typography, no logos, no people, 4K quality."
        )
        bg_path = self._generate_image_hf(bg_prompt, "_flyer_bg.png")

        try:
            bg = Image.open(bg_path).convert("RGBA").resize((W, H), Image.LANCZOS)
        except Exception:
            return self._flyer_pillow_fallback(headline, subtitle, topic, filename, body_text)

        # ── Step 2: Solid text panel on left 52% using palette colour ─────────
        PANEL_W   = int(W * 0.52)
        PANEL_RGB = pal["panel"]

        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ov_draw = ImageDraw.Draw(overlay)

        # Solid panel left side
        ov_draw.rectangle([(0, 0), (PANEL_W, H)], fill=(*PANEL_RGB, 230))

        # Soft feather: blend panel edge into the photo over 80 px
        for x in range(80):
            alpha = int(230 * (1 - x / 80))
            ov_draw.rectangle([(PANEL_W + x, 0), (PANEL_W + x + 1, H)],
                              fill=(*PANEL_RGB, alpha))

        # Top and bottom stripes in palette accent colours
        ov_draw.rectangle([(0, 0),     (W, 7)], fill=(*ACCENT, 255))
        ov_draw.rectangle([(0, H - 7), (W, H)], fill=(*BOT,    255))

        bg = Image.alpha_composite(bg, overlay)
        draw = ImageDraw.Draw(bg)

        PAD = 56   # left padding for all text

        # ── Step 3: Topic label (plain rect — avoids cross-platform artefacts) ──
        tag_font = self._font(22)
        tag_text = "#" + self._ascii_tag(topic)
        tb       = draw.textbbox((0, 0), tag_text, font=tag_font)
        tag_w    = tb[2] - tb[0] + 32
        draw.rectangle([(PAD,     26), (PAD + tag_w, 26 + 38)], fill=(*ACCENT, 255))
        draw.rectangle([(PAD,     26), (PAD + 6,     26 + 38)], fill=(*BOT,    255))
        draw.text((PAD + 16, 31), tag_text, font=tag_font, fill=(255, 255, 255))

        # ── Step 4: Headline (large, bold, white) ─────────────────────────────
        # Use a key phrase from the headline — strip filler words for impact
        h_font = self._font(62)
        y = self._draw_wrapped(draw, headline, h_font, (255, 255, 255),
                               W, PAD, H // 4 - 10,
                               max_width=PANEL_W - PAD - 20, align="left",
                               line_gap=10)

        # Accent divider under headline in palette colour
        draw.rectangle([(PAD, y + 10), (PAD + 60, y + 14)],
                       fill=(*BOT, 255))
        y += 34

        # ── Step 5: Subtitle (palette-matched colour) ────────────────────────
        if subtitle:
            s_font = self._font(26)
            y = self._draw_wrapped(draw, subtitle, s_font, SUB_COL,
                                   W, PAD, y,
                                   max_width=PANEL_W - PAD - 20, align="left",
                                   line_gap=8)
        # ── Step 5b: Key insight excerpt from the post body ─────────────────────
        if body_text:
            y += 24
            # Vertical accent bar as quote indicator
            draw.rectangle([(PAD, y), (PAD + 4, y + 90)], fill=(*ACCENT, 200))
            bi_font = self._font(22)
            y = self._draw_wrapped(
                draw, f'“{body_text}”', bi_font, (220, 235, 255),
                W, PAD + 18, y,
                max_width=PANEL_W - PAD - 38, align="left", line_gap=7,
            )
        # ── Step 6: Badge bottom-left ────────────────────────────────────────
        badge_font = self._font(16)
        draw.text((PAD, H - 34), "+ AI Generated",
                  font=badge_font, fill=(*BOT, 200))

        out_path = self._out / filename
        bg.convert("RGB").save(str(out_path), "PNG")
        Path(bg_path).unlink(missing_ok=True)
        return str(out_path)

    def generate_infographic_flyer(
        self,
        headline: str,
        subtitle: str,
        topic: str,
        body_text: str = "",
        filename: str = "flyer.png",
    ) -> str:
        """
        Generate a 1200×627 ASG-style landscape infographic PNG for the FLYER post format.
        Uses HTML → Edge headless.  Fallback: Pillow.
        """
        print("  Generating infographic flyer (HTML → Edge headless, 1200×627)...")
        html = self._generate_flyer_html_local(headline, subtitle, topic, body_text)
        try:
            return self._screenshot_html(html, filename, 1200, 627)
        except Exception as exc:
            print(f"  [WARN] Edge flyer screenshot failed ({exc}). Falling back to Pillow.")
            return self._flyer_pillow_fallback(headline, subtitle, topic, filename, body_text)

    def _generate_flyer_html_local(
        self,
        headline: str,
        subtitle: str,
        topic: str,
        body_text: str = "",
    ) -> str:
        """
        Generate a 1200×627 landscape HTML infographic (ASG style) for the flyer format.
        Left panel (navy): headline + subtitle + hashtag.
        Right panel (white): 3 key-point cards from body_text.
        """
        import re
        import html as _h

        hl = _h.escape(headline)
        sub = _h.escape(subtitle)
        tag = _h.escape("#" + self._ascii_tag(topic))
        body_esc = _h.escape(body_text[:160]) if body_text else ""

        sentences = [
            _h.escape(s.strip())
            for s in re.split(r'[.;\n]+', body_text)
            if len(s.strip()) > 10
        ][:3]
        while len(sentences) < 3:
            sentences.append("Key insight from the post")

        icons = ["&#128200;", "&#9881;", "&#127919;"]
        cards_html = ""
        for i, (icon, sent) in enumerate(zip(icons, sentences)):
            cards_html += (
                f'<div class="card">'
                f'<div class="chdr"><span class="cnum">{icon}</span>'
                f'<span class="ctitle">Key Point {i + 1}</span></div>'
                f'<div class="cbody">{sent[:140]}</div>'
                f'</div>'
            )

        excerpt_html = f'<p class="excerpt">{body_esc}</p>' if body_esc else ""
        return (
            "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><style>"
            "*{box-sizing:border-box;margin:0;padding:0}"
            "body{width:1200px;height:627px;overflow:hidden;font-family:'Segoe UI',Arial,sans-serif}"
            ".wrap{display:flex;width:100%;height:100%}"
            ".left{width:54%;background:linear-gradient(145deg,#0D1F59 0%,#1a3a8a 100%);"
            "padding:48px 52px;display:flex;flex-direction:column;justify-content:center;position:relative}"
            ".bar-top{position:absolute;top:0;left:0;right:0;height:6px;background:#0077B5}"
            ".bar-bot{position:absolute;bottom:0;left:0;right:0;height:6px;background:#0077B5}"
            ".tag{display:inline-block;background:#0077B5;color:#fff;font-size:15px;font-weight:700;"
            "padding:6px 18px;border-radius:20px;margin-bottom:24px}"
            "h1{color:#fff;font-size:44px;font-weight:900;line-height:1.2;margin-bottom:14px;max-width:520px}"
            ".sub{color:rgba(255,255,255,.72);font-size:18px;line-height:1.55;max-width:480px;margin-bottom:20px}"
            ".divider{width:70px;height:5px;background:#0077B5;border-radius:3px;margin-bottom:18px}"
            ".excerpt{color:rgba(255,255,255,.55);font-size:15px;line-height:1.5;max-width:480px;"
            "border-left:3px solid #0077B5;padding-left:14px}"
            ".right{width:46%;background:#fff;padding:36px 40px;display:flex;flex-direction:column;"
            "justify-content:center;gap:12px;border-left:4px solid #0077B5}"
            ".right-title{font-size:17px;font-weight:700;color:#0D1F59;margin-bottom:4px}"
            ".card{border:1px solid #dde5f5;border-radius:10px;overflow:hidden}"
            ".chdr{background:#0D1F59;padding:10px 16px;display:flex;align-items:center;gap:10px}"
            ".cnum{font-size:18px}"
            ".ctitle{color:#fff;font-size:14px;font-weight:700}"
            ".cbody{padding:10px 16px;color:#333;font-size:14px;line-height:1.45}"
            f"</style></head><body><div class=\"wrap\">"
            f"<div class=\"left\"><div class=\"bar-top\"></div>"
            f"<span class=\"tag\">{tag}</span>"
            f"<h1>{hl}</h1>"
            f"<div class=\"divider\"></div>"
            f"<p class=\"sub\">{sub}</p>"
            f"{excerpt_html}"
            f"<div class=\"bar-bot\"></div></div>"
            f"<div class=\"right\"><p class=\"right-title\">Key Takeaways</p>"
            f"{cards_html}</div>"
            "</div></body></html>"
        )

    def generate_carousel(
        self,
        slides: List[dict],
        topic: str,
        post_hook: str = "",
        filename: str = "carousel.pdf",
        infographic_mode: bool = False,
    ) -> str:
        """
        Route to the appropriate carousel renderer based on image provider:
          gemini          → Gemini Imagen 3 full-infographic slides (text baked in, no overlay)
          infographic_mode → Hybrid per-slide AI images + Pillow text
          default          → Classic shared-background carousel
        """
        if self._image_provider == "gemini":
            return self._generate_gemini_carousel(slides, topic, post_hook, filename)
        if infographic_mode:
            return self._generate_infographic_carousel(slides, topic, post_hook, filename)
        return self._generate_classic_carousel(slides, topic, post_hook, filename)

    # ── AI Infographic carousel (Gemini HTML → Edge headless screenshot) ────────

    def _generate_gemini_carousel(
        self,
        slides: List[dict],
        topic: str,
        post_hook: str = "",
        filename: str = "carousel.pdf",
    ) -> str:
        """
        Per-slide flow:
          1. Gemini LLM expands content to 300-400 words (heading + bullets).
          2. Gemini generates a complete self-contained HTML/CSS infographic slide
             (white background, dark navy headers, numbered steps — ASG/Azure style).
          3. Edge headless browser screenshots the HTML at 1080×1080 → PNG.
          Fallback: Pollinations.ai image if HTML/Edge path fails.
          All slides assembled into PDF — NO Pillow text overlay at any point.
        """
        from fpdf import FPDF

        n = len(slides)
        slide_paths: List[str] = []

        print(f"  Building AI infographic carousel ({n} slides)...")

        # ── Step 1: Batch-expand ALL slide content in ONE Gemini call ────────
        print(f"  Step 1 — Batch-expanding content for all {n} slides (1 Gemini call)...")
        all_expanded = self._expand_all_slides_batch(slides, topic, post_hook)

        for i, slide in enumerate(slides):
            img_filename = f"_ai_slide_{i}.png"
            img_path = None
            expanded = (
                all_expanded.get(i)
                or f"# {slide.get('title', topic)}\n\n{slide.get('body', '')}"
            )

            # ── Step 2+3: Gemini HTML → Edge screenshot ───────────────────────
            try:
                print(f"  [Slide {i+1}/{n}] Step 2 — Generating HTML infographic with Gemini...")
                html = self._generate_slide_html_with_gemini(slide, topic, expanded, i, n)
                print(f"  [Slide {i+1}/{n}] Step 3 — Screenshotting HTML with Edge headless...")
                img_path = self._screenshot_html_slide(html, img_filename)
            except Exception as exc:
                print(f"  [WARN] HTML/Edge path failed ({type(exc).__name__}). Trying Pollinations fallback...")

            # ── Fallback: Pollinations.ai ─────────────────────────────────────
            if img_path is None:
                try:
                    img_prompt = self._build_asg_infographic_prompt(slide, topic, expanded, i, n)
                    img_path = self._generate_slide_image_ai(img_prompt, img_filename)
                except Exception as exc2:
                    print(f"  [ERROR] All methods failed for slide {i+1} ({exc2}). Skipping.")
                    continue

            slide_paths.append(img_path)

        if not slide_paths:
            raise RuntimeError("All slide images failed to generate.")

        # ── Assemble into PDF (images are the slides — nothing added on top) ──
        pdf = FPDF(unit="mm", format=(210, 210))
        for sp in slide_paths:
            pdf.add_page()
            pdf.image(sp, x=0, y=0, w=210, h=210)

        out_path = self._out / filename
        pdf.output(str(out_path))

        for sp in slide_paths:
            Path(sp).unlink(missing_ok=True)

        print(f"  AI infographic carousel saved: {out_path}")
        return str(out_path)

    def _generate_slide_html_with_gemini(
        self,
        slide: dict,
        topic: str,
        expanded_content: str,
        slide_idx: int,
        total_slides: int,
    ) -> str:
        """
        Use Gemini to generate a complete self-contained HTML/CSS infographic slide.
        Design target: white background, dark navy headers, numbered sections,
        flow arrows — matching ASG Clinical Platform / Azure Architecture style.
        """
        from google.genai import types

        title    = slide.get("title", topic)
        layout   = slide.get("layout", "split-left")
        key_stat = slide.get("key_stat", "")
        content  = expanded_content[:1200]

        layout_instructions = {
            "cover": (
                f'COVER SLIDE — Topic: "{title}"\n'
                "- Full-width dark navy header (height:90px) with large bold white title\n"
                "- Below header: 3-4 concept boxes in a horizontal row\n"
                "- Each box: navy top bar (40px) with emoji icon + label, white body with 2-3 bullet lines\n"
                "- LinkedIn blue (#0077B5) horizontal accent line below header\n"
                '- Bottom strip: "Swipe to explore →" in LinkedIn blue\n'
            ),
            "stat-callout": (
                f'METRICS SLIDE — Title: "{title}"\n'
                f'Key Stat to highlight prominently: "{key_stat}"\n'
                "- Full-width dark navy header (height:80px) with slide title in white\n"
                f'- Center: large bordered callout box with "{key_stat}" in huge navy bold '
                "text (font-size:96px+), labelled 'Key Metric' below\n"
                "- Below callout: 3 KPI tiles side-by-side, each with a navy header + white body\n"
                "- Supporting bullet points at the bottom\n"
            ),
            "cta": (
                f'CALL-TO-ACTION SLIDE — Title: "{title}"\n'
                "- Full-width dark navy header (height:80px) with slide title in white\n"
                "- Large motivational heading and sub-text in the center\n"
                '- Prominent rounded navy button: "Follow for more →"\n'
                "- CSS-drawn upward-trending arrow graphic on the right\n"
            ),
        }.get(layout, (
            f'CONTENT SLIDE — Title: "{title}"\n'
            "- Full-width dark navy header bar (height:80px) with white title text\n"
            "- Main area split into TWO panels with a thin gray vertical divider:\n"
            "  LEFT PANEL (48% width):\n"
            f'    • If stat available ("{key_stat}"): bordered stat box with stat in '
            "large bold navy text at top\n"
            "    • 3-4 bullet points with filled navy circle bullets (●)\n"
            "  RIGHT PANEL (48% width):\n"
            "    • 3 numbered step cards stacked vertically\n"
            "    • Each card: navy header strip with step number (01/02/03) + step title in white\n"
            "    • White card body with brief 1-2 line description\n"
            "    • Downward arrow (▼) between each card, centered, in navy\n"
        ))

        system = (
            "You are an expert HTML/CSS infographic designer for corporate presentations.\n"
            "Generate a COMPLETE, self-contained HTML file for a 1080×1080px LinkedIn carousel slide.\n\n"
            "STRICT RULES:\n"
            "1. Root div: width:1080px; height:1080px; overflow:hidden; box-sizing:border-box\n"
            "2. White background (#FFFFFF) for the page body and content panels\n"
            "3. Dark navy (#0D1F59) for ALL header bars, section headers, numbered badges\n"
            "4. Accent blue (#0077B5) for highlights, bullet dots, arrows, borders\n"
            "5. Font: 'Segoe UI', Arial, sans-serif — NO external fonts\n"
            "6. All CSS in a single <style> block — NO external stylesheets\n"
            "7. NO JavaScript. NO external images. NO SVG from URLs.\n"
            "8. Use emoji for icons (📊 ⚙️ ✅ 🔹 📈 💡 🎯 etc.) — no <img> tags\n"
            "9. All text clearly readable: min font-size 15px, dark text on white, white on navy\n"
            "10. Return ONLY the raw HTML — no markdown fences, no explanation\n"
        )

        user_msg = (
            f"Slide {slide_idx + 1} of {total_slides}.\n\n"
            f"Layout spec:\n{layout_instructions}\n\n"
            f"Content to visualise:\n{content}\n\n"
            "Generate the complete 1080×1080 HTML infographic. "
            "Make it look like a polished McKinsey/ASG consulting slide. "
            "Every element must fit within the 1080×1080 box."
        )

        import re as _re
        import time as _time

        for attempt in range(3):
            try:
                resp = self._gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=user_msg,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=0.3,
                    ),
                )
                html = resp.text.strip()
                if html.startswith("```"):
                    lines = html.splitlines()
                    lines = lines[1:]
                    if lines and lines[-1].strip().startswith("```"):
                        lines = lines[:-1]
                    html = "\n".join(lines)
                print(f"    Gemini HTML generated ({len(html)} chars).")
                return html.strip()
            except Exception as exc:
                retryable = "429" in str(exc) or "503" in str(exc) or "UNAVAILABLE" in str(exc)
                if attempt < 2 and retryable:
                    m = _re.search(r'retry\s*in\s*([\d.]+)s', str(exc))
                    wait = int(float(m.group(1))) + 5 if m else 30
                    print(f"    Gemini busy ({type(exc).__name__}). Waiting {wait}s (attempt {attempt + 2}/3)...")
                    _time.sleep(wait)
                else:
                    print(f"    Gemini HTML failed ({type(exc).__name__}). Using local renderer.")
                    return self._generate_slide_html_local(slide, topic, slide_idx, total_slides)

    def _generate_slide_html_local(
        self,
        slide: dict,
        topic: str,
        slide_idx: int,
        total_slides: int,
    ) -> str:
        """
        Generate a professional ASG-style infographic HTML slide entirely in Python
        — no API call required.  Used when Gemini is unavailable/rate-limited.
        White background, dark navy headers, numbered step cards, readable text.
        """
        import re
        import html as _h

        title     = _h.escape(slide.get("title", topic))
        body      = slide.get("body", "")
        key_stat  = _h.escape(slide.get("key_stat", ""))
        layout    = slide.get("layout", "split-left")
        tag       = _h.escape("#" + self._ascii_tag(topic))
        num_label = f"{slide_idx + 1}&nbsp;/&nbsp;{total_slides}"

        # Parse bullet sentences from body
        sentences = [
            _h.escape(s.strip())
            for s in re.split(r'[.;]+', body)
            if len(s.strip()) > 8
        ]

        # ── COVER ────────────────────────────────────────────────────────────
        if layout == "cover":
            sub = _h.escape(body[:260]) if body else ""
            boxes = "".join(
                f'<div class="box"><div class="bi">{ic}</div>'
                f'<div class="bt">{s[:70]}</div></div>'
                for ic, s in zip(["&#9881;", "&#128200;", "&#127919;", "&#128161;"], sentences[:4])
            )
            return (
                "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><style>"
                "*{box-sizing:border-box;margin:0;padding:0}"
                "body{width:1080px;height:1080px;overflow:hidden;font-family:'Segoe UI',Arial,sans-serif}"
                ".wrap{width:100%;height:100%;background:linear-gradient(155deg,#0D1F59 0%,#1a3a8a 55%,#0a2246 100%);display:flex;flex-direction:column}"
                ".bar{height:6px;background:#0077B5;flex-shrink:0}"
                ".main{flex:1;padding:52px 64px;display:flex;flex-direction:column;justify-content:center}"
                ".top{display:flex;justify-content:space-between;align-items:center;margin-bottom:28px}"
                ".tag{background:#0077B5;color:#fff;font-size:18px;font-weight:700;padding:8px 22px;border-radius:22px}"
                ".num{color:rgba(255,255,255,.4);font-size:15px}"
                "h1{color:#fff;font-size:62px;font-weight:900;line-height:1.15;max-width:760px;margin-bottom:20px}"
                ".sub{color:rgba(255,255,255,.72);font-size:20px;line-height:1.55;max-width:720px;margin-bottom:38px}"
                ".boxes{display:flex;gap:16px;margin-bottom:34px}"
                ".box{flex:1;background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.18);border-radius:12px;padding:18px 14px}"
                ".bi{font-size:26px;margin-bottom:8px}"
                ".bt{color:rgba(255,255,255,.84);font-size:14px;line-height:1.45;font-weight:500}"
                ".swipe{color:#0077B5;font-size:18px;font-weight:700}"
                f"</style></head><body><div class=\"wrap\"><div class=\"bar\"></div>"
                f"<div class=\"main\"><div class=\"top\"><span class=\"tag\">{tag}</span>"
                f"<span class=\"num\">{num_label}</span></div>"
                f"<h1>{title}</h1><p class=\"sub\">{sub}</p>"
                f"<div class=\"boxes\">{boxes}</div>"
                "<p class=\"swipe\">Swipe to explore &rarr;</p>"
                "</div><div class=\"bar\"></div></div></body></html>"
            )

        # ── CTA ──────────────────────────────────────────────────────────────
        elif layout == "cta":
            body_esc = _h.escape(body[:300]) if body else ""
            return (
                "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><style>"
                "*{box-sizing:border-box;margin:0;padding:0}"
                "body{width:1080px;height:1080px;overflow:hidden;font-family:'Segoe UI',Arial,sans-serif}"
                ".wrap{width:100%;height:100%;background:linear-gradient(155deg,#0D1F59 0%,#1a3a8a 55%,#0a2246 100%);display:flex;flex-direction:column}"
                ".bar{height:6px;background:#0077B5;flex-shrink:0}"
                ".main{flex:1;padding:72px;display:flex;flex-direction:column;justify-content:center}"
                ".num{color:rgba(255,255,255,.35);font-size:15px;margin-bottom:20px}"
                "h1{color:#fff;font-size:58px;font-weight:900;line-height:1.18;max-width:880px;margin-bottom:16px}"
                ".div{width:110px;height:6px;background:#0077B5;border-radius:3px;margin:20px 0 28px}"
                ".sub{color:rgba(255,255,255,.70);font-size:23px;line-height:1.55;max-width:740px;margin-bottom:56px}"
                ".btn{background:#0077B5;color:#fff;font-size:26px;font-weight:700;padding:22px 54px;border-radius:40px;display:inline-block}"
                f"</style></head><body><div class=\"wrap\"><div class=\"bar\"></div>"
                f"<div class=\"main\"><div class=\"num\">{num_label}</div>"
                f"<h1>{title}</h1><div class=\"div\"></div>"
                f"<p class=\"sub\">{body_esc}</p>"
                "<div class=\"btn\">Follow for more &rarr;</div>"
                "</div><div class=\"bar\"></div></div></body></html>"
            )

        # ── STAT CALLOUT ──────────────────────────────────────────────────────
        elif layout == "stat-callout":
            stat_box = (
                f'<div class="sbox"><div class="snum">{key_stat}</div>'
                '<div class="slbl">Key Metric</div></div>'
            ) if key_stat else ""
            bullets = "".join(
                f'<div class="bl"><span class="dot">&#9679;</span><span>{s[:140]}</span></div>'
                for s in sentences[:3]
            )
            return (
                "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><style>"
                "*{box-sizing:border-box;margin:0;padding:0}"
                "body{width:1080px;height:1080px;overflow:hidden;font-family:'Segoe UI',Arial,sans-serif;background:#fff}"
                ".hdr{background:#0D1F59;height:88px;display:flex;align-items:center;padding:0 36px;border-bottom:5px solid #0077B5}"
                ".hdr h1{color:#fff;font-size:30px;font-weight:700;flex:1}"
                ".hdr .num{color:rgba(255,255,255,.45);font-size:15px}"
                ".body{padding:48px 60px;display:flex;flex-direction:column;align-items:center}"
                ".sbox{border:3px solid #0077B5;border-radius:20px;padding:36px 80px;text-align:center;margin-bottom:40px;min-width:500px}"
                ".snum{font-size:110px;font-weight:900;color:#0D1F59;line-height:1}"
                ".slbl{font-size:19px;color:#666;margin-top:8px;font-weight:600;text-transform:uppercase;letter-spacing:1px}"
                ".bls{width:100%;max-width:840px}"
                ".bl{display:flex;align-items:flex-start;margin-bottom:18px;font-size:19px;color:#2a2a4a;line-height:1.5}"
                ".dot{color:#0077B5;font-size:11px;margin-right:16px;margin-top:5px;flex-shrink:0}"
                f"</style></head><body>"
                f"<div class=\"hdr\"><h1>{title}</h1><span class=\"num\">{num_label}</span></div>"
                f"<div class=\"body\">{stat_box}<div class=\"bls\">{bullets}</div></div>"
                "</body></html>"
            )

        # ── SPLIT-LEFT (default content slide) ────────────────────────────────
        else:
            stat_html = (
                f'<div class="sbox">'
                f'<div class="snum">{key_stat}</div>'
                '<div class="slbl">Key Metric</div></div>'
            ) if key_stat else ""

            bullets_html = "".join(
                f'<div class="bl"><span class="dot">&#9679;</span><span>{s[:110]}</span></div>'
                for s in sentences[:4]
            )

            step_titles = []
            for s in sentences[:3]:
                words = s.split()
                step_titles.append((" ".join(words[:5]) + "&hellip;") if len(words) > 5 else s)
            while len(step_titles) < 3:
                step_titles.append(f"Step {len(step_titles) + 1}")

            step_descs = (sentences[1:4] + ["", "", ""])[:3]

            steps_html = ""
            for j in range(3):
                steps_html += (
                    f'<div class="card">'
                    f'<div class="chdr"><span class="cnum">0{j+1}</span>'
                    f'<span class="ctitle">{step_titles[j]}</span></div>'
                    f'<div class="cbody">{step_descs[j][:110]}</div>'
                    f'</div>'
                )
                if j < 2:
                    steps_html += '<div class="arr">&#9660;</div>'

            return (
                "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><style>"
                "*{box-sizing:border-box;margin:0;padding:0}"
                "body{width:1080px;height:1080px;overflow:hidden;font-family:'Segoe UI',Arial,sans-serif;background:#fff}"
                ".hdr{background:#0D1F59;height:88px;display:flex;align-items:center;padding:0 36px;border-bottom:5px solid #0077B5}"
                ".badge{background:#0077B5;color:#fff;font-size:17px;font-weight:700;width:44px;height:44px;border-radius:50%;display:flex;align-items:center;justify-content:center;margin-right:16px;flex-shrink:0}"
                ".hdr h1{color:#fff;font-size:27px;font-weight:700;flex:1}"
                ".hdr .num{color:rgba(255,255,255,.45);font-size:15px}"
                ".panels{display:flex;height:992px}"
                ".left{width:50%;padding:30px 30px 30px 40px;border-right:2px solid #e0e8f5;overflow:hidden}"
                ".right{width:50%;padding:30px 36px 30px 30px;display:flex;flex-direction:column;overflow:hidden}"
                ".sbox{border:2px solid #0077B5;border-radius:14px;padding:16px 20px;margin-bottom:20px;background:#f0f5ff;display:flex;align-items:center;gap:18px}"
                ".snum{font-size:50px;font-weight:900;color:#0D1F59;line-height:1;white-space:nowrap}"
                ".slbl{font-size:15px;color:#555;font-weight:600;line-height:1.3}"
                ".bl{display:flex;align-items:flex-start;margin-bottom:13px;font-size:17px;color:#2a2a4a;line-height:1.5}"
                ".dot{color:#0077B5;font-size:10px;margin-right:12px;margin-top:4px;flex-shrink:0}"
                ".card{background:#fff;border:1px solid #dde5f5;border-radius:10px;overflow:hidden;margin-bottom:6px}"
                ".chdr{background:#0D1F59;padding:11px 16px;display:flex;align-items:center;gap:12px}"
                ".cnum{background:#0077B5;color:#fff;font-size:14px;font-weight:700;width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0}"
                ".ctitle{color:#fff;font-size:15px;font-weight:700}"
                ".cbody{padding:11px 16px;color:#333;font-size:15px;line-height:1.5}"
                ".arr{text-align:center;color:#0077B5;font-size:20px;line-height:1;margin:3px 0}"
                f"</style></head><body>"
                f"<div class=\"hdr\"><div class=\"badge\">{slide_idx+1}</div>"
                f"<h1>{title}</h1><span class=\"num\">{num_label}</span></div>"
                f"<div class=\"panels\">"
                f"<div class=\"left\">{stat_html}{bullets_html}</div>"
                f"<div class=\"right\">{steps_html}</div>"
                "</div></body></html>"
            )

    def _screenshot_html(
        self, html_content: str, filename: str, width: int = 1080, height: int = 1080
    ) -> str:
        """
        Save HTML to a temp file then capture a screenshot using Edge/Chrome headless
        at the given pixel dimensions.  Returns the path of the saved PNG.
        """
        import subprocess

        tmp_html = self._out / f"_tmp_{filename}.html"
        tmp_html.write_text(html_content, encoding="utf-8")
        out_path = self._out / filename

        browser_candidates = [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ]
        browser = next((p for p in browser_candidates if Path(p).exists()), None)
        if not browser:
            tmp_html.unlink(missing_ok=True)
            raise RuntimeError("No headless browser (Edge/Chrome) found.")

        file_uri = tmp_html.resolve().as_uri()
        cmd = [
            browser,
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-software-rasterizer",
            "--force-device-scale-factor=1",
            f"--window-size={width},{height}",
            f"--screenshot={str(out_path.resolve())}",
            file_uri,
        ]

        result = subprocess.run(cmd, capture_output=True, timeout=45)
        tmp_html.unlink(missing_ok=True)

        if not out_path.exists() or out_path.stat().st_size < 2000:
            raise RuntimeError(
                f"Edge screenshot failed. "
                f"stderr: {result.stderr.decode(errors='replace')[:400]}"
            )
        print(f"    HTML rendered by Edge headless ({width}\u00d7{height}) \u2713")
        return str(out_path)

    def _screenshot_html_slide(self, html_content: str, filename: str) -> str:
        """1080\u00d71080 screenshot \u2014 delegates to _screenshot_html for carousel slides."""
        return self._screenshot_html(html_content, filename, 1080, 1080)

    def _expand_all_slides_batch(
        self,
        slides: List[dict],
        topic: str,
        post_hook: str = "",
    ) -> dict:
        """
        Single Gemini call that returns expanded content (200-300 words) for ALL
        slides at once.  Returns {slide_index: content_string}.  Falls back to
        empty dict so callers use the original slide brief instead.
        """
        try:
            from google.genai import types
            import json as _json

            specs = [
                {
                    "idx":      i,
                    "title":    s.get("title", topic),
                    "body":     s.get("body", "")[:200],
                    "key_stat": s.get("key_stat", ""),
                    "layout":   s.get("layout", "split-left"),
                }
                for i, s in enumerate(slides)
            ]

            user_msg = (
                f"Topic: {topic}\nPost hook: {post_hook or 'N/A'}\n\n"
                "Expand each slide into 200-300 words of structured professional content.\n"
                "For each slide include: 1 main heading, 2-3 sub-sections with sub-headings, "
                "concise bullet points, and any key stat prominently.\n\n"
                f"Slides:\n{_json.dumps(specs, indent=2)}\n\n"
                f"Return ONLY a JSON object mapping each slide index (as a string) to its "
                f"expanded content string.  Example: {{\"0\": \"# Title\\n\\nContent...\", \"1\": \"...\"}}"
            )

            resp = self._gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_msg,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.5,
                ),
            )
            raw = resp.text.strip()
            if raw.startswith("```"):
                raw = raw.split("```", 2)[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.rsplit("```", 1)[0]
            data = _json.loads(raw)
            result = {}
            for k, v in data.items():
                try:
                    result[int(k)] = str(v)
                except (ValueError, TypeError):
                    pass
            print(f"  Batch content generated for {len(result)}/{len(slides)} slides.")
            return result
        except Exception as exc:
            print(f"  [WARN] Batch content expansion failed ({exc}). Using slide briefs.")
            return {}

    def _expand_slide_content_with_llm(
        self,
        slide: dict,
        topic: str,
        post_hook: str = "",
    ) -> str:
        """
        Step 1: Use Gemini LLM to expand one slide into 300-400 words of structured
        content (heading + sub-sections + bullets). This content informs the image prompt.
        """
        try:
            from google.genai import types

            title    = slide.get("title", topic)
            body     = slide.get("body", "")
            key_stat = slide.get("key_stat", "")
            layout   = slide.get("layout", "split-left")

            layout_hint = {
                "cover": (
                    "This is the COVER slide. Write a compelling introduction overview "
                    "that sets the stage for the full carousel."
                ),
                "stat-callout": (
                    f"This is a METRICS slide. Feature the key statistic prominently: '{key_stat}'. "
                    "Back it up with context, trend data, and 2-3 supporting data points."
                ),
                "cta": (
                    "This is the CALL-TO-ACTION slide. Write an engaging closing that "
                    "motivates readers to follow, share, or comment."
                ),
            }.get(layout, (
                "This is a CONTENT slide. Structure with 3 numbered steps or clear "
                "sub-sections that professionals can act on immediately."
            ))

            user_msg = (
                f"Topic: {topic}\n"
                f"Slide title: {title}\n"
                f"Brief content: {body}\n"
                f"Key statistic: {key_stat or 'N/A'}\n"
                f"Post hook: {post_hook or 'N/A'}\n\n"
                f"{layout_hint}\n\n"
                "Write a professional 300-400 word infographic content piece.\n"
                "Structure:\n"
                "- 1 main heading\n"
                "- 2-3 sub-sections each with a sub-heading\n"
                "- Concise bullet points or numbered steps under each section\n"
                "- Include the key stat if provided\n"
                "Be concise, authoritative, and suitable for a LinkedIn professional audience."
            )

            resp = self._gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_msg,
                config=types.GenerateContentConfig(temperature=0.6),
            )
            content = resp.text.strip()
            print(f"    Content generated ({len(content.split())} words).")
            return content
        except Exception as exc:
            print(f"  [WARN] Content expansion failed ({exc}). Using slide brief.")
            return f"# {slide.get('title', topic)}\n\n{slide.get('body', '')}"

    def _build_asg_infographic_prompt(
        self,
        slide: dict,
        topic: str,
        expanded_content: str,
        slide_idx: int,
        total_slides: int,
    ) -> str:
        """
        Step 2: Build a detailed image prompt for a professional white-background
        infographic diagram in the style of ASG Clinical Platform / Azure Architecture:
        numbered sections, dark navy headers, flow arrows, clean flat icons, white bg.
        """
        title    = slide.get("title", topic)
        layout   = slide.get("layout", "split-left")
        key_stat = slide.get("key_stat", "")

        # Trim expanded content to key context for the prompt
        content_ctx = " ".join(expanded_content.split()[:100])

        base_style = (
            "Professional corporate infographic diagram, pure white background (#FFFFFF). "
            "Consulting firm slide deck quality — McKinsey / Accenture / Azure Architecture Center style. "
            "Dark navy blue (#0D1F59) header bars and section title labels. "
            "Numbered boxes with colored navy/blue headers and white content panels. "
            "Flow arrows connecting elements. Flat design icons. "
            "Clear numbered sections (01, 02, 03). "
            "Clean sans-serif typography. White background throughout — no dark backgrounds. "
            "Vector illustration style, no photography, no people. "
        )

        layout_spec = {
            "cover": (
                f"Cover infographic slide for the topic: '{title}'. "
                "Full-width dark navy header bar at top with large bold white title. "
                "Below header: 3-4 horizontally arranged concept boxes with icons, "
                "each with a colored header (navy/blue) and white body. "
                "Arrows connecting the concept boxes left to right. "
                "Professional overview diagram layout."
            ),
            "stat-callout": (
                f"Metrics infographic with '{key_stat}' as the central highlighted KPI. "
                "Large bold metric number inside a prominent bordered callout box. "
                "3 supporting KPI tiles arranged in a row below the main metric. "
                "Bar chart or trend line as a secondary visual element. "
                "Full-width dark navy header bar at top with slide title."
            ),
            "cta": (
                "Professional call-to-action closing infographic. "
                "Upward-trending growth arrows and chart element on the right side. "
                "Clean navy and blue color accent sections. "
                "Professional closing slide with clear visual emphasis."
            ),
        }.get(layout, (
            f"Multi-section process flow infographic about '{title}'. "
            "Full-width dark navy header bar at top containing the slide title in white. "
            "LEFT PANEL (50% width): information panel with a key stats box at top "
            "and 3-4 bullet point lines below. "
            "RIGHT PANEL (50% width): 3 numbered vertical step boxes, each with a "
            "dark navy header strip containing the step number (01, 02, 03) and step title, "
            "and white body content below. Downward arrows between each step box. "
            "Thin vertical gray divider line between left and right panels."
        ))

        return (
            f"{base_style}"
            f"{layout_spec} "
            f"Content context: {content_ctx}. "
            "1080x1080 square format. "
            "Sharp edges, high resolution, professional LinkedIn publication quality."
        )

    def _generate_slide_image_ai(self, prompt: str, filename: str) -> str:
        """
        Step 3: Generate the slide image using HuggingFace FLUX.1-schnell (primary)
        or Pollinations.ai FLUX model (fallback). No Pillow rendering — pure AI images.
        """
        from config.settings import get_settings
        settings = get_settings()
        token = settings.hf_api_token

        # ── Primary: HuggingFace FLUX.1-schnell ──────────────────────────────
        if token:
            try:
                api_url = (
                    "https://router.huggingface.co/hf-inference/models/"
                    "black-forest-labs/FLUX.1-schnell"
                )
                headers = {"Authorization": f"Bearer {token}"}
                payload = {
                    "inputs": prompt,
                    "parameters": {"width": 1024, "height": 1024},
                }
                resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
                resp.raise_for_status()
                out_path = self._out / filename
                out_path.write_bytes(resp.content)
                print(f"    Image generated via FLUX.1-schnell (HuggingFace).")
                return str(out_path)
            except Exception as exc:
                print(f"  [WARN] HuggingFace FLUX failed ({exc}), trying Pollinations...")

        # ── Fallback: Pollinations.ai FLUX model ─────────────────────────────
        try:
            encoded = urllib.parse.quote(prompt, safe="")
            seed    = abs(hash(prompt)) % 99999
            url     = (
                f"https://image.pollinations.ai/prompt/{encoded}"
                f"?width=1024&height=1024&model=flux&seed={seed}&nologo=true"
            )
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            out_path = self._out / filename
            out_path.write_bytes(resp.content)
            print(f"    Image generated via Pollinations.ai.")
            return str(out_path)
        except Exception as exc:
            print(f"  [ERROR] All image providers failed for this slide ({exc}).")
            raise


    # ── Infographic carousel (AI-generated images + Pillow text overlay) ────────

    def _generate_infographic_carousel(
        self,
        slides: List[dict],
        topic: str,
        post_hook: str = "",
        filename: str = "carousel.pdf",
    ) -> str:
        """
        Hybrid infographic carousel:
          • Each slide (except CTA) requests one AI image via IMAGE_PROVIDER.
            Set IMAGE_PROVIDER=openai + OPENAI_API_KEY for DALL-E 3 quality;
            keeps pollinations/huggingface as free fallbacks.
          • Cover:         full-bleed AI image + dark gradient panel on left for text
          • Content slides: clean dark text panel (left 58%) + AI image panel (right 38%)
          • Stat-callout:   AI image strip (top 36%) + circular gauge + body below
          • CTA:            pure Pillow, no AI image needed
          Falls back to Pillow-drawn bar chart when image generation fails.
        """
        from fpdf import FPDF
        import math

        W = H = 1080
        PAD = 72

        ACCENT_COLORS = [
            (0,   119, 181),  # LinkedIn blue — cover
            (0,   180, 140),  # teal          — slide 2
            (130,  60, 210),  # purple        — slide 3
            (0,   150, 210),  # sky           — slide 4
            (210, 120,   0),  # amber         — slide 5
            (220,  50,  80),  # coral         — CTA
        ]
        BG_COLORS = [
            (8,  18, 55),
            (5,  38, 45),
            (20, 10, 48),
            (5,  28, 52),
            (42, 22,  5),
            (48, 10, 22),
        ]
        WHITE = (255, 255, 255)
        DIM   = (178, 208, 238)
        PALE  = (110, 150, 195)

        slide_paths: List[str] = []
        n        = len(slides)
        provider = getattr(self, "_image_provider", "pollinations")
        print(f"  Building hybrid infographic carousel ({n} slides, '{provider}' images)...")

        for i, slide in enumerate(slides):
            accent   = ACCENT_COLORS[min(i, len(ACCENT_COLORS) - 1)]
            bg       = BG_COLORS[min(i, len(BG_COLORS) - 1)]
            is_cover = (i == 0)
            is_cta   = (i == n - 1)
            layout   = slide.get("layout", "split-left")

            # ── 1. Generate AI image for this slide (skip CTA) ───────────────
            ai_image = None
            if not is_cta:
                raw_hint   = slide.get("slide_image_prompt", "")
                title_txt  = slide.get("title", topic)
                img_prompt = (
                    f"{raw_hint}. " if raw_hint else ""
                ) + (
                    f"Professional infographic illustration, topic: {title_txt}. "
                    f"Theme: {topic}. Clean flat vector style, corporate design, "
                    "no text, no words, no letters, high detail, 8K quality."
                )
                print(f"  [Slide {i+1}/{n}] Requesting AI image ({provider})...")
                ai_tmp = self.generate_image(img_prompt, f"_infographic_ai_tmp_{i}.png")
                try:
                    ai_image = Image.open(ai_tmp).convert("RGB")
                except Exception as exc:
                    print(f"  [Slide {i+1}] Image load failed ({exc}), using Pillow fallback.")
                finally:
                    try:
                        Path(ai_tmp).unlink(missing_ok=True)
                    except Exception:
                        pass

            # ── 2. Base canvas (gradient + dot grid) ─────────────────────────
            img  = Image.new("RGB", (W, H), bg)
            draw = ImageDraw.Draw(img)
            for row in range(H):
                t  = row / H
                rc = tuple(max(0, int(bg[c] * (1.15 - t * 0.25))) for c in range(3))
                draw.line([(0, row), (W, row)], fill=rc)
            dc = tuple(min(255, c + 13) for c in bg)
            for gy in range(0, H + 50, 50):
                for gx in range(0, W + 50, 50):
                    draw.ellipse([(gx - 1, gy - 1), (gx + 1, gy + 1)], fill=dc)

            # ════════════════════════════════════════════════════════════════
            if is_cover:
                # Full-bleed AI image + left-side dark gradient for text ──────
                if ai_image:
                    img.paste(ai_image.resize((W, H), Image.LANCZOS), (0, 0))

                # Dark gradient overlay: opaque on far left → transparent at 65%
                TEXT_END = int(W * 0.65)
                FEATHER  = 140
                overlay  = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                od       = ImageDraw.Draw(overlay)
                for x in range(TEXT_END + FEATHER):
                    if x <= TEXT_END - FEATHER:
                        alpha = 215
                    else:
                        t = (x - (TEXT_END - FEATHER)) / FEATHER
                        alpha = max(0, int(215 * (1.0 - t)))
                    od.line([(x, 0), (x, H)], fill=(*bg, alpha))
                img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
                draw = ImageDraw.Draw(img)

                # Frame
                draw.rectangle([(0, 0),     (W, 9)],    fill=accent)
                draw.rectangle([(0, H - 6), (W, H)],    fill=accent)
                draw.rectangle([(0, 9),     (6, H - 6)], fill=accent)
                draw.text((W - PAD, 28), f"1 / {n}",
                           font=self._font(22), fill=PALE, anchor="rs")

                # Topic badge
                tag   = "#" + self._ascii_tag(topic)
                tb    = draw.textbbox((0, 0), tag, font=self._font(24))
                tag_w = tb[2] - tb[0] + 36
                draw.rounded_rectangle([(PAD, 28), (PAD + tag_w, 76)],
                                        radius=24, fill=accent)
                draw.text((PAD + 18, 35), tag, font=self._font(24), fill=WHITE)

                # Large title
                max_tw = TEXT_END - PAD - 40
                y = self._draw_wrapped(draw, slide.get("title", topic),
                                       self._font(82), WHITE, W,
                                       PAD + 8, H // 3,
                                       max_width=max_tw, align="left", line_gap=18)
                draw.rectangle([(PAD + 8, y + 22), (PAD + 130, y + 30)], fill=accent)

                sub = post_hook or slide.get("body", "")
                if sub:
                    self._draw_wrapped(draw, sub, self._font(30), DIM, W,
                                       PAD + 8, y + 48,
                                       max_width=max_tw, align="left", line_gap=10)

                draw.text((TEXT_END - 20, H - PAD - 6), "Swipe to explore →",
                           font=self._font(24), fill=accent, anchor="rs")

            # ════════════════════════════════════════════════════════════════
            elif is_cta:
                # Pure Pillow CTA ─────────────────────────────────────────────
                draw.rectangle([(0, 0),     (W, 9)],    fill=accent)
                draw.rectangle([(0, H - 6), (W, H)],    fill=accent)
                draw.rectangle([(0, 9),     (6, H - 6)], fill=accent)
                draw.text((W - PAD, 28), f"{n} / {n}",
                           font=self._font(22), fill=PALE, anchor="rs")

                for ci, (cx_off, bright) in enumerate([(-90, 22), (-38, 32), (14, 45)]):
                    ax = W - 200 + cx_off
                    for seg in range(5):
                        sc = tuple(min(255, c + bright + seg * 8) for c in bg)
                        y1 = H - 130 - seg * 62
                        draw.line([(ax, y1), (ax, y1 - 48)], fill=sc, width=4)
                    draw.polygon([(ax-13, H-130-5*62), (ax+13, H-130-5*62),
                                   (ax,   H-130-5*62-20)], fill=accent)

                y = self._draw_wrapped(draw, slide.get("title", "Let's Connect"),
                                       self._font(72), WHITE, W,
                                       PAD + 8, H // 5,
                                       max_width=W - 270, align="left", line_gap=14)
                draw.rectangle([(PAD + 8, y + 18), (PAD + 110, y + 26)], fill=accent)
                body = slide.get("body", "")
                if body:
                    self._draw_wrapped(draw, body, self._font(30), DIM, W,
                                       PAD + 8, y + 44,
                                       max_width=W - 270, align="left", line_gap=10)
                btn_txt = "Follow for more →"
                bb      = draw.textbbox((0, 0), btn_txt, font=self._font(30))
                btn_w   = bb[2] - bb[0] + 64
                bx      = (W - btn_w) // 2
                draw.rounded_rectangle([(bx, H - 192), (bx + btn_w, H - 120)],
                                        radius=38, fill=accent)
                draw.text((bx + 32, H - 182), btn_txt, font=self._font(30), fill=WHITE)

            # ════════════════════════════════════════════════════════════════
            elif layout == "stat-callout":
                # AI image strip (top) + circular gauge + body ────────────────
                draw.rectangle([(0, 0),     (W, 9)],    fill=accent)
                draw.rectangle([(0, H - 6), (W, H)],    fill=accent)
                draw.rectangle([(0, 9),     (6, H - 6)], fill=accent)
                draw.text((W - PAD, 28), f"{i+1} / {n}",
                           font=self._font(22), fill=PALE, anchor="rs")

                if ai_image:
                    # Top image strip
                    strip_h = int(H * 0.34)
                    strip_w = W - PAD * 2
                    strip   = ai_image.resize((strip_w, strip_h), Image.LANCZOS)
                    mask    = Image.new("L", (strip_w, strip_h), 0)
                    ImageDraw.Draw(mask).rounded_rectangle(
                        [(0, 0), (strip_w - 1, strip_h - 1)], radius=18, fill=255)
                    img.paste(strip, (PAD, PAD + 18), mask)
                    ImageDraw.Draw(img).rounded_rectangle(
                        [(PAD-2, PAD+16), (PAD+strip_w+2, PAD+18+strip_h+2)],
                        radius=20, outline=accent, width=3)
                    draw    = ImageDraw.Draw(img)
                    text_y  = PAD + strip_h + 36
                    gauge_cy = text_y + 48 + 170
                else:
                    text_y   = PAD + 18
                    gauge_cy = H // 2 + 30

                # Step badge + title
                draw.ellipse([(PAD, text_y), (PAD + 54, text_y + 54)], fill=accent)
                nb = draw.textbbox((0, 0), str(i), font=self._font(28))
                draw.text((PAD + 27 - (nb[2]-nb[0])//2,
                            text_y + 27 - (nb[3]-nb[1])//2 - 2),
                           str(i), font=self._font(28), fill=WHITE)
                self._draw_wrapped(draw, slide.get("title", ""),
                                   self._font(46), WHITE, W,
                                   PAD + 8, text_y + 62,
                                   max_width=W - PAD * 2 - 60, align="left", line_gap=9)

                # Compact circular gauge
                R  = 155
                cx = W // 2
                gb = tuple(min(255, c + 22) for c in bg)
                draw.arc([(cx-R, gauge_cy-R), (cx+R, gauge_cy+R)],
                          start=-215, end=35, fill=gb, width=24)
                draw.arc([(cx-R, gauge_cy-R), (cx+R, gauge_cy+R)],
                          start=-215, end=-25, fill=accent, width=24)
                draw.ellipse([(cx-R+28, gauge_cy-R+28), (cx+R-28, gauge_cy+R-28)],
                              fill=tuple(min(255, c + 8) for c in bg))
                key_stat = slide.get("key_stat", "")
                if key_stat:
                    sf = self._font(80)
                    sb = draw.textbbox((0, 0), key_stat, font=sf)
                    sx = cx - (sb[2]-sb[0]) // 2
                    sy = gauge_cy - (sb[3]-sb[1]) // 2 - 8
                    draw.text((sx+3, sy+3), key_stat, font=sf,
                               fill=tuple(max(0, c-10) for c in bg))
                    draw.text((sx, sy), key_stat, font=sf, fill=WHITE)
                body = slide.get("body", "")
                if body:
                    self._draw_wrapped(draw, body, self._font(26), DIM, W,
                                       PAD + 8, gauge_cy + R + 22,
                                       max_width=W - PAD * 2, align="left", line_gap=8)

            # ════════════════════════════════════════════════════════════════
            else:
                # Content slide: text left panel + AI image right panel ───────
                VIZ_X  = int(W * 0.60)
                TEXT_W = VIZ_X - PAD - 20

                draw.rectangle([(0, 0),     (W, 9)],    fill=accent)
                draw.rectangle([(0, H - 6), (W, H)],    fill=accent)
                draw.rectangle([(0, 9),     (6, H - 6)], fill=accent)
                draw.text((W - PAD, 28), f"{i+1} / {n}",
                           font=self._font(22), fill=PALE, anchor="rs")

                # Step badge
                draw.ellipse([(PAD, PAD + 4), (PAD + 60, PAD + 64)], fill=accent)
                nb = draw.textbbox((0, 0), str(i), font=self._font(30))
                draw.text((PAD + 30 - (nb[2]-nb[0])//2,
                            PAD + 32 - (nb[3]-nb[1])//2 - 2),
                           str(i), font=self._font(30), fill=WHITE)

                # Title
                y = self._draw_wrapped(draw, slide.get("title", ""),
                                       self._font(52), WHITE, W,
                                       PAD + 8, PAD * 2 + 14,
                                       max_width=TEXT_W, align="left", line_gap=11)
                draw.rectangle([(PAD + 8, y + 14), (PAD + 80, y + 20)], fill=accent)
                y += 36

                # Key-stat callout box
                key_stat = slide.get("key_stat", "")
                if key_stat:
                    sf  = self._font(42)
                    sb  = draw.textbbox((0, 0), key_stat, font=sf)
                    kx1 = PAD + 8
                    kx2 = min(kx1 + (sb[2]-sb[0]) + 48, VIZ_X - 20)
                    ky2 = y + 76
                    draw.rounded_rectangle([(kx1+4, y+4), (kx2+4, ky2+4)], radius=14,
                                            fill=tuple(max(0, c-8) for c in bg))
                    draw.rounded_rectangle([(kx1, y), (kx2, ky2)],
                                            radius=14, fill=accent)
                    draw.text((kx1 + 20, y + 16), key_stat, font=sf, fill=WHITE)
                    y = ky2 + 28

                body = slide.get("body", "")
                if body:
                    self._draw_wrapped(draw, body, self._font(27), DIM, W,
                                       PAD + 8, y,
                                       max_width=TEXT_W, align="left", line_gap=9)

                # Right panel: AI image (with rounded corners) or bar-chart fallback
                px = VIZ_X + 10
                py = PAD + 18
                pw = W - px - PAD + 8
                ph = H - py - PAD - 16

                if ai_image:
                    panel = ai_image.resize((pw, ph), Image.LANCZOS)
                    mask  = Image.new("L", (pw, ph), 0)
                    ImageDraw.Draw(mask).rounded_rectangle(
                        [(0, 0), (pw - 1, ph - 1)], radius=22, fill=255)
                    img.paste(panel, (px, py), mask)
                    # Accent border around image panel
                    ImageDraw.Draw(img).rounded_rectangle(
                        [(px-3, py-3), (px+pw+3, py+ph+3)],
                        radius=25, outline=accent, width=3)
                    draw = ImageDraw.Draw(img)
                else:
                    # Fallback: horizontal bar chart
                    bar_labels = ["Q1", "Q2", "Q3", "Q4"]
                    bar_vals   = [0.80, 0.55, 0.72, 0.90]
                    bh         = 30
                    bgap       = 52
                    max_bw     = pw
                    bt         = tuple(min(255, c + 18) for c in bg)
                    by0        = py + (ph - (4 * bh + 3 * bgap)) // 2
                    for bi, bv in enumerate(bar_vals):
                        by  = by0 + bi * (bh + bgap)
                        bw  = int(max_bw * bv)
                        draw.rounded_rectangle([(px, by), (px+max_bw, by+bh)],
                                                radius=7, fill=bt)
                        draw.rounded_rectangle([(px, by), (px+bw, by+bh)],
                                                radius=7, fill=accent)
                        draw.text((px-10, by+bh//2), bar_labels[bi],
                                   font=self._font(20), fill=PALE, anchor="rm")
                        draw.ellipse([(px+bw-7, by+bh//2-7),
                                       (px+bw+7, by+bh//2+7)], fill=WHITE)

                # Progress bar
                total_c = max(n - 2, 1)
                prog    = max(0.05, (i - 1) / total_c)
                bar_y2  = H - 52
                pw2     = W - PAD * 2 - 20
                pt      = tuple(min(255, c + 22) for c in bg)
                draw.rounded_rectangle([(PAD+8, bar_y2), (PAD+8+pw2, bar_y2+7)],
                                        radius=4, fill=pt)
                draw.rounded_rectangle([(PAD+8, bar_y2), (PAD+8+int(pw2*prog), bar_y2+7)],
                                        radius=4, fill=accent)

            # ── Save slide PNG ────────────────────────────────────────────────
            out_slide = self._out / f"_infographic_slide_{i}.png"
            img.save(str(out_slide), "PNG")
            slide_paths.append(str(out_slide))

        # ── Combine all slides into PDF ───────────────────────────────────────
        pdf = FPDF(unit="mm", format=(210, 210))
        for sp in slide_paths:
            pdf.add_page()
            pdf.image(sp, x=0, y=0, w=210, h=210)

        out_path = self._out / filename
        pdf.output(str(out_path))

        for sp in slide_paths:
            Path(sp).unlink(missing_ok=True)

        print(f"  Hybrid carousel saved: {out_path}")
        return str(out_path)

    def _build_infographic_prompt(self, slide: dict, topic: str, accent_name: str) -> str:
        """
        Build a detailed 300-400 word DALL-E / image generation prompt that describes
        a professional infographic illustration tailored to the slide's content.
        """
        title      = slide.get("title", topic)
        body       = slide.get("body", "")
        key_stat   = slide.get("key_stat", "")
        icon       = slide.get("icon_concept", "data flow network")
        layout     = slide.get("layout", "split-left")
        base_hint  = slide.get("slide_image_prompt", "")

        # Layout-specific visual language
        if layout == "cover":
            layout_desc  = "panoramic hero layout — large central focal element surrounded by radiating geometric patterns filling the full frame"
            visual_focus = "bold central abstract icon or emblem with radiating connection lines, surrounded by layered hexagonal or circular frames"
        elif layout == "stat-callout":
            layout_desc  = "centered data-visualization layout dominated by a large circular gauge or radial progress chart"
            visual_focus = "circular arc progress gauge, radial segments, concentric rings, abstract metric dial with tick marks"
        elif layout == "cta":
            layout_desc  = "dynamic forward-motion layout with directional flow elements converging toward center"
            visual_focus = "upward-trending arrow clusters, converging network paths, forward-motion abstract shapes"
        else:
            layout_desc  = "structured information hierarchy layout — upper area has a bold visual anchor element, lower two-thirds is clear dark gradient for text"
            visual_focus = "abstract process flow diagram with connected rounded-rectangle nodes, arrows between steps, layered depth"

        stat_clause = (
            f"Abstract visual representation of the key metric '{key_stat}' "
            f"— suggest scale and achievement through proportional shapes and gauge elements. "
            if key_stat else ""
        )
        hint_clause = f"{base_hint}. " if base_hint else ""

        prompt = (
            f"Professional LinkedIn business infographic illustration. "
            f"Topic: {title}. Context: {body[:140] if body else topic}. "
            f"{hint_clause}"
            f"Visual composition: {layout_desc}. "
            f"Primary visual element: {visual_focus}, representing '{icon}' concept. "
            f"{stat_clause}"
            f"Detailed design specifications: "
            f"Background — deep dark navy blue (#0B1640) fading to dark charcoal in corners, "
            f"creating a premium dark corporate gradient. "
            f"Accent color — {accent_name} used for primary shapes, icon outlines, connecting arrows, "
            f"and highlight points; renders as bright vivid color against the dark background. "
            f"Supporting elements — soft blue-grey (#3A5A8A) for secondary geometric shapes; "
            f"pure white dots and fine lines for detail accents. "
            f"Texture — subtle diagonal dot-grid pattern at 5% opacity overlaid on background for professional depth; "
            f"faint diagonal parallel lines in background suggesting motion and dynamism. "
            f"Specific visual components to include: "
            f"(1) Large abstract {icon} symbol rendered in clean flat {accent_name} vector lines, center-weighted; "
            f"(2) Three to five connected rounded-rectangle or hexagonal information nodes arranged in logical flow; "
            f"(3) Smooth curved arrows or dotted connector lines linking the nodes in {accent_name}; "
            f"(4) Subtle radial glow or light-bloom behind the central element in {accent_name} at low opacity; "
            f"(5) Small decorative micro-icons (circuit nodes, data points, small arrows) scattered in background; "
            f"(6) Bottom third of image significantly darker (near black overlay) to ensure text overlay readability; "
            f"(7) Thin {accent_name} accent line along top edge and bottom edge of the image frame. "
            f"Composition rule: main visual weight concentrated in upper 60% of image; "
            f"lower 40% transitions to very dark gradient. "
            f"Quality targets: photorealistic render quality, 4K-level detail, clean vector aesthetic, "
            f"suitable for premium LinkedIn professional content, no noise or artifacts. "
            f"ABSOLUTE CONSTRAINT: Zero text, zero words, zero letters, zero numbers, zero characters "
            f"anywhere in the image. Purely abstract visual and graphical elements only."
        )
        return prompt


    # ── Classic carousel (shared background) ─────────────────────────────────

    def _generate_classic_carousel(
        self,
        slides: List[dict],
        topic: str,
        post_hook: str = "",
        filename: str = "carousel.pdf",
    ) -> str:
        """
        Original shared-background carousel:
          1. One FLUX call -> abstract background image
          2. Each slide reuses that background with a unique color tint overlay
          3. Pillow draws slide number, title, divider, body text on top
          4. All slides merged into a PDF
        """
        from fpdf import FPDF

        W = H = 1080

        # Colour tints per slide index (RGBA overlays)
        TINTS = [
            (8,   22,  70,  210),   # Cover     — deep navy
            (0,   80,  100, 195),   # Slide 2   — dark teal
            (55,  20,  120, 195),   # Slide 3   — deep purple
            (0,   60,  80,  195),   # Slide 4   — ocean
            (80,  40,  0,   195),   # Slide 5   — warm brown
            (120, 40,  0,   205),   # CTA       — amber/gold
        ]
        ACCENT_COLORS = [
            (0,   119, 181),   # LinkedIn blue
            (0,   200, 160),   # teal
            (140, 80,  220),   # purple
            (0,   160, 200),   # sky
            (220, 130,  20),   # amber
            (255, 180,   0),   # gold
        ]

        # ── Step 1: Generate ONE abstract background ─────────────────────────
        print("  Generating AI background for carousel (1 API call)...")
        bg_prompt = (
            "Abstract professional background, smooth dark gradient, "
            "soft bokeh light orbs, subtle diagonal geometric lines, "
            "cinematic depth of field, clean minimal, no text, no letters, "
            "no words, no typography, no logos, no people, 4K quality."
        )
        bg_path = self._generate_image_hf(bg_prompt, "_carousel_bg.png")

        try:
            bg_master = Image.open(bg_path).convert("RGBA").resize((W, H), Image.LANCZOS)
        except Exception:
            bg_master = None   # will use solid colour fallback per slide

        slide_paths: List[str] = []

        for i, slide in enumerate(slides):
            tint   = TINTS[min(i, len(TINTS) - 1)]
            accent = ACCENT_COLORS[min(i, len(ACCENT_COLORS) - 1)]
            is_cover = (i == 0)
            is_cta   = (i == len(slides) - 1)

            # ── Base: AI background or solid colour ──────────────────────────
            if bg_master:
                base = bg_master.copy()
            else:
                base = Image.new("RGBA", (W, H), (tint[0], tint[1], tint[2], 255))

            # ── Tint overlay ─────────────────────────────────────────────────
            tint_layer = Image.new("RGBA", (W, H), tint)
            base = Image.alpha_composite(base, tint_layer)

            # ── Accent bar at bottom ──────────────────────────────────────────
            bar = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            bar_d = ImageDraw.Draw(bar)
            bar_d.rectangle([(0, H - 8), (W, H)], fill=(*accent, 255))
            bar_d.rectangle([(0, 0), (W, 8)],     fill=(*accent, 100))
            base = Image.alpha_composite(base, bar)

            draw = ImageDraw.Draw(base)
            WHITE = (255, 255, 255)
            DIM   = (200, 220, 240)

            PAD = 72

            if is_cover:
                # ── COVER SLIDE ───────────────────────────────────────────────
                # Topic tag
                tag_font = self._font(22)
                tag = "#" + self._ascii_tag(topic)
                tb = draw.textbbox((0, 0), tag, font=tag_font)
                tag_w = tb[2] - tb[0] + 32
                draw.rounded_rectangle([(PAD, PAD), (PAD + tag_w, PAD + 38)],
                                        radius=19, fill=(*accent, 210))
                draw.text((PAD + 16, PAD + 6), tag, font=tag_font, fill=WHITE)

                # Big title
                t_font = self._font(72)
                y = self._draw_wrapped(draw, slide.get("title", topic),
                                       t_font, WHITE, W, PAD, H // 3,
                                       max_width=W - PAD * 2, align="left", line_gap=14)

                # Accent divider
                draw.rectangle([(PAD, y + 16), (PAD + 80, y + 22)],
                                fill=(*accent, 255))

                # Sub-headline — prefer the actual post hook so readers get context
                sub = post_hook or slide.get("body", "")
                if sub:
                    s_font = self._font(32)
                    self._draw_wrapped(draw, sub, s_font, DIM, W, PAD, y + 40,
                                       max_width=W - PAD * 2, align="left", line_gap=10)

                # "Swipe →" prompt
                sw_font = self._font(24)
                draw.text((W - PAD, H - PAD), "Swipe →",
                          font=sw_font, fill=(*accent, 230), anchor="rs")

            elif is_cta:
                # ── CTA SLIDE ─────────────────────────────────────────────────
                cta_font = self._font(60)
                y = self._draw_wrapped(draw, slide.get("title", "Follow for more"),
                                       cta_font, WHITE, W, PAD, H // 4,
                                       max_width=W - PAD * 2, align="left", line_gap=12)
                draw.rectangle([(PAD, y + 18), (PAD + 80, y + 24)],
                                fill=(*accent, 255))
                body = slide.get("body", "")
                if body:
                    b_font = self._font(30)
                    self._draw_wrapped(draw, body, b_font, DIM, W, PAD, y + 44,
                                       max_width=W - PAD * 2, align="left", line_gap=10)

                # Follow button pill
                btn_font = self._font(26)
                btn_text = "  Follow for more →  "
                bb = draw.textbbox((0, 0), btn_text, font=btn_font)
                btn_w = bb[2] - bb[0] + 20
                bx = (W - btn_w) // 2
                draw.rounded_rectangle([(bx, H - 160), (bx + btn_w, H - 108)],
                                        radius=30, fill=(*accent, 240))
                draw.text((bx + 10, H - 154), btn_text.strip(),
                          font=btn_font, fill=WHITE)

            else:
                # ── CONTENT SLIDE ─────────────────────────────────────────────
                # Step badge circle
                badge_font = self._font(38)
                cx, cy, cr = PAD + 30, PAD + 30, 38
                draw.ellipse([(cx - cr, cy - cr), (cx + cr, cy + cr)],
                              fill=(*accent, 230))
                num = str(i)
                nb = draw.textbbox((0, 0), num, font=badge_font)
                draw.text((cx - (nb[2] - nb[0]) // 2,
                            cy - (nb[3] - nb[1]) // 2 - 2),
                           num, font=badge_font, fill=WHITE)

                # Title
                t_font = self._font(54)
                y = self._draw_wrapped(draw, slide.get("title", ""),
                                       t_font, WHITE, W, PAD, PAD * 2 + 30,
                                       max_width=W - PAD * 2, align="left", line_gap=10)

                # Accent divider
                draw.rectangle([(PAD, y + 14), (PAD + 60, y + 19)],
                                fill=(*accent, 255))

                # Body text
                body = slide.get("body", "")
                if body:
                    b_font = self._font(30)
                    self._draw_wrapped(draw, body, b_font, DIM, W, PAD, y + 36,
                                       max_width=W - PAD * 2, align="left", line_gap=10)

                # Slide counter bottom-right
                cnt_font = self._font(22)
                draw.text((W - PAD, H - PAD),
                          f"{i + 1} / {len(slides)}",
                          font=cnt_font, fill=(*accent, 200), anchor="rs")

            out = self._out / f"_slide_{i}.png"
            base.convert("RGB").save(str(out), "PNG")
            slide_paths.append(str(out))

        # ── Combine into PDF ──────────────────────────────────────────────────
        pdf = FPDF(unit="mm", format=(210, 210))
        for sp in slide_paths:
            pdf.add_page()
            pdf.image(sp, x=0, y=0, w=210, h=210)

        out_path = self._out / filename
        pdf.output(str(out_path))

        for sp in slide_paths:
            Path(sp).unlink(missing_ok=True)
        if bg_path:
            Path(bg_path).unlink(missing_ok=True)

        return str(out_path)

    # ── HTML → Image renderer (Playwright) ───────────────────────────────────

    def _render_html_to_image(
        self,
        template_name: str,
        context: dict,
        width: int,
        height: int,
        filename: str,
    ) -> str:
        """Render an HTML/Jinja2 template to a PNG using Playwright headless Chromium."""
        try:
            from jinja2 import Environment, FileSystemLoader
            from playwright.sync_api import sync_playwright
        except ImportError:
            print("  [WARN] Playwright/Jinja2 not installed. Falling back to Pillow.")
            return self._flyer_pillow_fallback(
                context.get("headline_html", context.get("topic", "")),
                context.get("subtitle", ""),
                context.get("topic", ""),
                filename,
            )

        templates_dir = Path(__file__).parent.parent / "templates"
        env = Environment(loader=FileSystemLoader(str(templates_dir)))
        template = env.get_template(template_name)
        html_content = template.render(**context)

        out_path = self._out / filename
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page(viewport={"width": width, "height": height})
                page.set_content(html_content, wait_until="networkidle")
                page.screenshot(path=str(out_path), clip={"x": 0, "y": 0, "width": width, "height": height})
                browser.close()
        except Exception as exc:
            print(f"  [WARN] Playwright render failed ({exc}). Falling back to Pillow.")
            return self._flyer_pillow_fallback(
                context.get("headline_html", context.get("topic", "")),
                context.get("subtitle", ""),
                context.get("topic", ""),
                filename,
            )

        return str(out_path)

    def _flyer_pillow_fallback(
        self, headline: str, subtitle: str, topic: str, filename: str,
        body_text: str = "",
    ) -> str:
        """Pillow fallback when HF image generation is unavailable."""
        pal_idx = abs(hash(headline + body_text + subtitle)) % len(self._FLYER_PALETTES)
        pal    = self._FLYER_PALETTES[pal_idx]
        ACCENT = pal["top"]
        BOT    = pal["bot"]
        W, H = 1200, 627
        img = Image.new("RGB", (W, H), pal["panel"])
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, W, 8], fill=ACCENT)
        draw.rectangle([0, H - 8, W, H], fill=BOT)
        font_h = self._font(64)
        font_s = self._font(32)
        y = self._draw_wrapped(draw, headline, font_h, (255, 255, 255), W, 60, H // 4)
        if subtitle:
            y = self._draw_wrapped(draw, subtitle, font_s, pal["sub"], W, 60, y + 20)
        if body_text:
            draw.rectangle([60, y + 20, 64, y + 100], fill=ACCENT)
            self._draw_wrapped(draw, f'“{body_text}”', self._font(22), (220, 235, 255),
                               W, 80, y + 22, max_width=W // 2 - 80)
        out_path = self._out / filename
        img.save(str(out_path), "PNG")
        return str(out_path)

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _ascii_tag(text: str, max_chars: int = 28) -> str:
        """
        Convert topic text to a safe ASCII uppercase tag label.
        Normalises Unicode (e.g. é→e, em-dash→stripped) so Pillow never
        tries to render a glyph the font doesn't have (which shows as □).
        Truncates to max_chars words so the tag fits inside the panel.
        """
        # NFKD decomposition turns accented letters into base+combining;
        # encoding to ASCII then drops the combining marks and any other
        # non-ASCII codepoints (curly quotes, zero-width spaces, etc.).
        normalised = unicodedata.normalize("NFKD", text)
        ascii_only = normalised.encode("ascii", "ignore").decode("ascii")
        # Keep only printable characters, collapse whitespace
        clean = " ".join(ascii_only.split()).upper()
        # Hard-truncate at max_chars to prevent overflow
        if len(clean) > max_chars:
            clean = clean[:max_chars].rsplit(" ", 1)[0]  # break on word boundary
        return clean or "TOPIC"

    @staticmethod
    def _font(size: int) -> ImageFont.FreeTypeFont:
        """Load the best available system font at the given size."""
        candidates = {
            "Windows": [
                "C:\\Windows\\Fonts\\Arial.ttf",
                "C:\\Windows\\Fonts\\arial.ttf",
                "C:\\Windows\\Fonts\\Calibri.ttf",
                "C:\\Windows\\Fonts\\calibri.ttf",
                "C:\\Windows\\Fonts\\segoeui.ttf",
                "C:\\Windows\\Fonts\\Tahoma.ttf",
                "C:\\Windows\\Fonts\\Verdana.ttf",
            ],
            "Darwin": [
                "/System/Library/Fonts/Helvetica.ttc",
                "/Library/Fonts/Arial.ttf",
            ],
            "Linux": [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            ],
        }
        for path in candidates.get(platform.system(), []):
            try:
                return ImageFont.truetype(path, size)
            except (IOError, OSError):
                continue
        # Pillow >= 10 supports size on load_default
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()

    @staticmethod
    def _draw_wrapped(
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont,
        color: tuple,
        canvas_w: int,
        padding: int,
        start_y: int,
        line_gap: int = 12,
        max_width: int = None,
        align: str = "center",
    ) -> int:
        """
        Draw word-wrapped text.  align='center' centres across canvas_w;
        align='left' left-aligns at x=padding within max_width (or canvas_w-padding*2).
        Returns the y-coordinate immediately after the last line.
        """
        wrap_w = max_width if max_width else canvas_w - padding * 2
        words = text.split()
        lines: List[str] = []
        current: List[str] = []

        for word in words:
            test = " ".join(current + [word])
            bbox = draw.textbbox((0, 0), test, font=font)
            if bbox[2] - bbox[0] <= wrap_w:
                current.append(word)
            else:
                if current:
                    lines.append(" ".join(current))
                current = [word]
        if current:
            lines.append(" ".join(current))

        y = start_y
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_w = bbox[2] - bbox[0]
            line_h = bbox[3] - bbox[1]
            if align == "left":
                x = padding
            else:
                x = (canvas_w - line_w) // 2
            draw.text((x, y), line, font=font, fill=color)
            y += line_h + line_gap

        return y
