"""Extract plain-text job posting content from HTML (incl. schema.org JSON-LD)."""

from __future__ import annotations

import html as html_lib
import json
import re
from typing import Any, List, Optional

_JSON_LD_SCRIPT_RE = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.DOTALL | re.IGNORECASE,
)
_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|noscript)[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _flatten_json_ld(node: Any) -> List[dict]:
    """Collect object nodes from JSON-LD (@graph, list, single object)."""
    out: List[dict] = []
    if isinstance(node, dict):
        graph = node.get("@graph")
        if isinstance(graph, list):
            for item in graph:
                out.extend(_flatten_json_ld(item))
        elif graph is not None:
            out.extend(_flatten_json_ld(graph))
        else:
            out.append(node)
    elif isinstance(node, list):
        for item in node:
            out.extend(_flatten_json_ld(item))
    return out


def _is_job_posting(obj: dict) -> bool:
    t = obj.get("@type")
    if isinstance(t, str):
        return "jobposting" in t.lower()
    if isinstance(t, list):
        return any(isinstance(x, str) and "jobposting" in x.lower() for x in t)
    return False


def _html_to_plain(fragment: str) -> str:
    if not fragment:
        return ""
    t = html_lib.unescape(fragment)
    t = _TAG_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _job_posting_to_text(obj: dict) -> str:
    parts: List[str] = []
    title = obj.get("title") or obj.get("name")
    if isinstance(title, str) and title.strip():
        parts.append(f"Посада: {title.strip()}")

    org = obj.get("hiringOrganization") or obj.get("employer")
    if isinstance(org, dict):
        org_name = org.get("name")
        if isinstance(org_name, str) and org_name.strip():
            parts.append(f"Роботодавець: {org_name.strip()}")

    desc = obj.get("description")
    if isinstance(desc, str) and desc.strip():
        plain = _html_to_plain(desc)
        if plain:
            parts.append(plain)

    skills = obj.get("skills")
    if isinstance(skills, str) and skills.strip():
        parts.append(f"Навички: {skills.strip()}")
    elif isinstance(skills, list):
        skill_lines = [str(s).strip() for s in skills if str(s).strip()]
        if skill_lines:
            parts.append("Навички: " + ", ".join(skill_lines))

    qual = obj.get("qualifications") or obj.get("experienceRequirements") or obj.get("educationRequirements")
    if isinstance(qual, str) and qual.strip():
        parts.append(_html_to_plain(qual))
    elif isinstance(qual, dict):
        qual_text = qual.get("description") or qual.get("name")
        if isinstance(qual_text, str) and qual_text.strip():
            parts.append(_html_to_plain(qual_text))

    emp = obj.get("employmentType")
    if isinstance(emp, str) and emp.strip():
        parts.append(f"Тип зайнятості: {emp.strip()}")

    loc = obj.get("jobLocation")
    if isinstance(loc, dict):
        addr = loc.get("address")
        if isinstance(addr, dict):
            locality = addr.get("addressLocality") or addr.get("addressRegion") or addr.get("addressCountry")
            if isinstance(locality, str) and locality.strip():
                parts.append(f"Локація: {locality.strip()}")

    return "\n\n".join(parts).strip()


def extract_job_text_from_html(html: str) -> Optional[str]:
    """
    Prefer schema.org JobPosting JSON-LD; fallback to visible HTML text.
    Returns None if nothing usable was found.
    """
    postings: List[dict] = []
    for block in _JSON_LD_SCRIPT_RE.findall(html or ""):
        block = block.strip()
        if not block:
            continue
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        for node in _flatten_json_ld(data):
            if _is_job_posting(node):
                postings.append(node)

    if postings:
        # Use the posting with the longest description
        texts = [_job_posting_to_text(p) for p in postings]
        texts = [t for t in texts if t]
        if texts:
            return max(texts, key=len)

    body = _SCRIPT_STYLE_RE.sub(" ", html or "")
    body = _TAG_RE.sub(" ", body)
    body = html_lib.unescape(body)
    body = re.sub(r"\s+", " ", body).strip()
    # Reject JSON-LD-shaped noise if it slipped through
    if body.startswith("{") and '"@context"' in body[:500]:
        return None
    return body if len(body) >= 80 else None
