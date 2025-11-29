"""LLM committee agents for book analysis."""

from ingest.agents.base import BaseAgent, LLMClient
from ingest.agents.bibliography import BibliographyAgent
from ingest.agents.chapter_summary import ChapterSummaryAgent
from ingest.agents.committee import LLMCommittee
from ingest.agents.genre_tags import GenreTagsAgent
from ingest.agents.key_quotes import KeyQuotesAgent
from ingest.agents.overview import BookOverviewAgent
from ingest.agents.technical_concepts import TechnicalConceptsAgent
from ingest.agents.themes import ThemesAgent
from ingest.agents.vlm import VLMAgent

__all__ = [
    "BaseAgent",
    "LLMClient",
    "LLMCommittee",
    "BibliographyAgent",
    "GenreTagsAgent",
    "KeyQuotesAgent",
    "BookOverviewAgent",
    "ChapterSummaryAgent",
    "ThemesAgent",
    "TechnicalConceptsAgent",
    "VLMAgent",
]
