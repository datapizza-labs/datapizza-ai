from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel
import yaml


class TargetProfile(BaseModel):
    industries: List[str] = []
    locations: List[str] = []
    size: str | None = None
    roles: List[str] = []


class TonePreferences(BaseModel):
    email: str = "professionale"
    linkedin: str = "professionale"


class FeaturedProduct(BaseModel):
    name: str
    tagline: str | None = None
    description: str | None = None
    key_features: List[str] = []
    target_audience: List[str] = []
    benefits: List[str] = []
    use_cases: List[str] = []
    tech_stack: str | None = None
    github_url: str | None = None
    cta_primary: str | None = None
    cta_secondary: str | None = None


class PersonalProfile(BaseModel):
    name: str
    title: str | None = None
    about: str | None = None
    services: List[str] = []
    offer_default: str | None = None
    tone: TonePreferences = TonePreferences()
    cta_link: Optional[str] = None
    targets: TargetProfile = TargetProfile()
    email_from: Optional[str] = None
    featured_product: Optional[FeaturedProduct] = None

    @classmethod
    def load(cls, path: str | Path) -> "PersonalProfile":
        p = Path(path)
        with p.open() as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
