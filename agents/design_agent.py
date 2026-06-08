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
        self._image_model = settings.image_model
        self._image_size = settings.image_size
        self._out = Path("./data/outputs")
        self._out.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_image(
        self, image_prompt: str, filename: str = "post_image.png"
    ) -> str:
        """Generate a LinkedIn-ready image and save it to disk."""
        if self._image_provider == "huggingface":
            return self._generate_image_hf(image_prompt, filename)
        if self._image_provider == "pollinations":
            return self._generate_image_pollinations(image_prompt, filename)
        return self._generate_image_openai(image_prompt, filename)

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
        encoded = urllib.parse.quote(
            f"{image_prompt}. Professional LinkedIn graphic, clean modern design.",
            safe="",
        )
        # Use minimal params — enhanced/nologo features now require payment
        url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024"
        print("  Requesting image from Pollinations.ai...")
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

    def generate_carousel(
        self,
        slides: List[dict],
        topic: str,
        post_hook: str = "",
        filename: str = "carousel.pdf",
    ) -> str:
        """
        AI-enhanced carousel (Option B):
          1. One FLUX call → abstract background image
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
