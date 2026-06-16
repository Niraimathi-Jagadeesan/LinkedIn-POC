from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class PostFormat(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    FLYER = "flyer"
    CAROUSEL = "carousel"
    PPTX = "pptx"


@dataclass
class LinkedInPost:
    urn: str
    author: str
    text: str
    created: int  # epoch ms
    media_type: Optional[str] = None


@dataclass
class GeneratedContent:
    text: str
    hashtags: List[str]
    image_prompt: str
    topic: str
    format: PostFormat
    flyer_headline: Optional[str] = None
    flyer_subtitle: Optional[str] = None
    slides: Optional[List[dict]] = None       # carousel slides
    image_path: Optional[str] = None
    flyer_path: Optional[str] = None
    carousel_path: Optional[str] = None
    asset_urn: Optional[str] = None
