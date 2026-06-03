"""
Design Agent — generates visuals for LinkedIn posts.
  • IMAGE   → DALL-E 3 (OpenAI) OR Pollinations.ai (free, no key)
  • FLYER   → Pillow-rendered branded graphic card (1200×627)
  • CAROUSEL → Pillow-rendered slides exported as a PDF (1080×1080 per slide)
"""

import platform
import urllib.parse
from pathlib import Path
from typing import List

import requests
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

from config.settings import get_settings


class DesignAgent:
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
        if self._image_provider == "pollinations":
            return self._generate_image_pollinations(image_prompt, filename)
        return self._generate_image_openai(image_prompt, filename)

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
        filename: str = "flyer.png",
    ) -> str:
        """Render a branded flyer card (1200×627 px) using Pillow."""
        W, H = 1200, 627
        BG = (15, 76, 129)        # deep LinkedIn blue
        ACCENT = (0, 119, 181)    # LinkedIn blue
        TEAL = (0, 200, 150)
        WHITE = (255, 255, 255)
        LIGHT = (200, 220, 240)

        img = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(img)

        # Decorative circles
        draw.ellipse((-120, -120, 280, 280), fill=ACCENT)
        draw.ellipse((980, 380, 1380, 780), fill=ACCENT)

        # Top and bottom accent bars
        draw.rectangle([0, 0, W, 8], fill=TEAL)
        draw.rectangle([0, H - 8, W, H], fill=TEAL)

        # Topic tag (top-right)
        tag_font = self._font(22)
        draw.text((W - 40, 24), f"#{topic.upper()}", font=tag_font, fill=TEAL, anchor="ra")

        # Headline — centred, word-wrapped
        h_font = self._font(68)
        y = self._draw_wrapped(draw, headline, h_font, WHITE, W, 60, H // 4)

        # Subtitle
        if subtitle:
            s_font = self._font(34)
            self._draw_wrapped(draw, subtitle, s_font, LIGHT, W, 60, y + 24)

        out_path = self._out / filename
        img.save(str(out_path), "PNG")
        return str(out_path)

    def generate_carousel(
        self,
        slides: List[dict],
        topic: str,
        filename: str = "carousel.pdf",
    ) -> str:
        """
        Render each slide as a 1080×1080 image and combine into a PDF.
        Each slide dict must have keys 'title' and 'body'.
        """
        from fpdf import FPDF

        PALETTE = [
            (15, 76, 129),    # deep blue
            (0, 119, 181),    # linkedin blue
            (0, 150, 136),    # teal
            (103, 58, 183),   # purple
            (183, 28, 28),    # red (CTA slide)
        ]
        W = H = 1080
        slide_paths: List[str] = []

        for i, slide in enumerate(slides[:5]):
            title = slide.get("title", f"Point {i + 1}")
            body = slide.get("body", "")
            bg = PALETTE[i % len(PALETTE)]

            img = Image.new("RGB", (W, H), bg)
            draw = ImageDraw.Draw(img)

            # Decorative bars
            draw.rectangle([0, 0, W, 10], fill=(255, 255, 255, 30))
            draw.rectangle([0, H - 10, W, H], fill=(255, 255, 255, 30))

            # Slide counter
            counter_font = self._font(36)
            draw.text(
                (48, 44),
                f"{i + 1} / {len(slides[:5])}",
                font=counter_font,
                fill=(255, 255, 255),
            )

            # Title
            t_font = self._font(58)
            y = self._draw_wrapped(draw, title, t_font, (255, 255, 255), W, 60, H // 4)

            # Body
            if body:
                b_font = self._font(34)
                self._draw_wrapped(draw, body, b_font, (200, 230, 255), W, 60, y + 30)

            slide_path = self._out / f"_slide_{i}.png"
            img.save(str(slide_path))
            slide_paths.append(str(slide_path))

        # Combine slides into a square PDF
        pdf = FPDF(unit="mm", format=(210, 210))
        for sp in slide_paths:
            pdf.add_page()
            pdf.image(sp, x=0, y=0, w=210, h=210)

        out_path = self._out / filename
        pdf.output(str(out_path))

        # Clean up temp slide images
        for sp in slide_paths:
            Path(sp).unlink(missing_ok=True)

        return str(out_path)

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _font(size: int) -> ImageFont.FreeTypeFont:
        """Load the best available system font at the given size."""
        candidates = {
            "Windows": [
                "C:\\Windows\\Fonts\\Arial.ttf",
                "C:\\Windows\\Fonts\\Calibri.ttf",
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
    ) -> int:
        """
        Draw word-wrapped text centred horizontally.
        Returns the y-coordinate immediately after the last line.
        """
        max_w = canvas_w - padding * 2
        words = text.split()
        lines: List[str] = []
        current: List[str] = []

        for word in words:
            test = " ".join(current + [word])
            bbox = draw.textbbox((0, 0), test, font=font)
            if bbox[2] - bbox[0] <= max_w:
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
            x = (canvas_w - line_w) // 2
            draw.text((x, y), line, font=font, fill=color)
            y += line_h + line_gap

        return y
