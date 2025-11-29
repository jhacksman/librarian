"""LLM Committee orchestrator for coordinating all agents."""

import asyncio
from typing import Any

from ingest.agents.base import LLMClient
from ingest.agents.bibliography import BibliographyAgent
from ingest.agents.chapter_summary import ChapterSummaryAgent
from ingest.agents.genre_tags import GenreTagsAgent
from ingest.agents.key_quotes import KeyQuotesAgent
from ingest.agents.overview import BookOverviewAgent
from ingest.agents.technical_concepts import TechnicalConceptsAgent
from ingest.agents.themes import ThemesAgent
from ingest.agents.vlm import VLMAgent
from ingest.config import IngestConfig
from ingest.models import BookAnalysis, ExtractedContent
from ingest.utils.logging import get_logger

logger = get_logger("agents.committee")


class LLMCommittee:
    """Orchestrates the LLM committee for comprehensive book analysis."""

    def __init__(self, config: IngestConfig) -> None:
        """Initialize the LLM committee.

        Args:
            config: Ingest configuration
        """
        self.config = config
        self.llm_client = LLMClient(config.llm)

        self.bibliography_agent = BibliographyAgent(self.llm_client)
        self.genre_tags_agent = GenreTagsAgent(
            self.llm_client,
            predefined_tags=config.agents.genre_tags if config.agents else None,
        )
        self.overview_agent = BookOverviewAgent(self.llm_client)
        self.chapter_summary_agent = ChapterSummaryAgent(self.llm_client)
        self.key_quotes_agent = KeyQuotesAgent(self.llm_client)
        self.themes_agent = ThemesAgent(self.llm_client)
        self.technical_concepts_agent = TechnicalConceptsAgent(self.llm_client)

        self.vlm_agent = None
        if config.vlm and config.vlm.enabled:
            self.vlm_agent = VLMAgent(config.vlm)

        logger.info("Initialized LLM committee with 7 agents")
        if self.vlm_agent:
            logger.info("VLM agent enabled for image processing")

    async def analyze(self, content: ExtractedContent) -> BookAnalysis:
        """Perform comprehensive analysis of a book.

        Args:
            content: Extracted content from a document

        Returns:
            Complete book analysis
        """
        logger.info(f"Starting committee analysis for: {content.title}")

        image_results = []
        if self.vlm_agent and content.images:
            logger.info(f"Processing {len(content.images)} images with VLM")
            image_results = await self.vlm_agent.process_images(content.images)

        logger.info("Running text analysis agents in parallel")
        results = await asyncio.gather(
            self.bibliography_agent.analyze(content),
            self.genre_tags_agent.analyze(content),
            self.overview_agent.analyze(content),
            self.themes_agent.analyze(content),
            self.technical_concepts_agent.analyze(content),
            self.key_quotes_agent.analyze(content),
            return_exceptions=True,
        )

        bibliography_data = self._handle_result(results[0], "bibliography", {})
        genre_tags_data = self._handle_result(results[1], "genre_tags", {})
        overview_data = self._handle_result(results[2], "overview", {})
        themes_data = self._handle_result(results[3], "themes", {})
        concepts_data = self._handle_result(results[4], "technical_concepts", {})
        quotes_data = self._handle_result(results[5], "key_quotes", {})

        logger.info("Running chapter summary agent")
        chapter_summaries_data = await self.chapter_summary_agent.analyze(content)

        analysis = self._build_analysis(
            content=content,
            bibliography_data=bibliography_data,
            genre_tags_data=genre_tags_data,
            overview_data=overview_data,
            themes_data=themes_data,
            concepts_data=concepts_data,
            quotes_data=quotes_data,
            chapter_summaries_data=chapter_summaries_data,
            image_results=image_results,
        )

        logger.info(f"Completed committee analysis for: {content.title}")
        return analysis

    def _handle_result(self, result: Any, agent_name: str, default: Any) -> Any:
        """Handle a result from an agent, logging errors.

        Args:
            result: Result from asyncio.gather
            agent_name: Name of the agent for logging
            default: Default value if result is an exception

        Returns:
            Result or default value
        """
        if isinstance(result, Exception):
            logger.error(f"{agent_name} agent failed: {result}")
            return default
        return result

    def _build_analysis(
        self,
        content: ExtractedContent,
        bibliography_data: dict,
        genre_tags_data: dict,
        overview_data: dict,
        themes_data: dict,
        concepts_data: dict,
        quotes_data: dict,
        chapter_summaries_data: dict,
        image_results: list[dict],
    ) -> BookAnalysis:
        """Build a BookAnalysis from agent results.

        Args:
            content: Original extracted content
            bibliography_data: Bibliography agent results
            genre_tags_data: Genre/tags agent results
            overview_data: Overview agent results
            themes_data: Themes agent results
            concepts_data: Technical concepts agent results
            quotes_data: Key quotes agent results
            chapter_summaries_data: Chapter summary agent results
            image_results: VLM image processing results

        Returns:
            Complete BookAnalysis
        """
        bibliography = self.bibliography_agent.to_bibliography(bibliography_data)
        quotes = self.key_quotes_agent.to_quote_models(quotes_data)
        concepts = self.technical_concepts_agent.to_concept_models(concepts_data)

        return BookAnalysis(
            bibliography=bibliography,
            genres=genre_tags_data.get("genres", []),
            tags=genre_tags_data.get("tags", []),
            target_audience=genre_tags_data.get("target_audience", ""),
            difficulty_level=genre_tags_data.get("difficulty_level", ""),
            prerequisites=genre_tags_data.get("prerequisites", []),
            summary=overview_data.get("summary", ""),
            main_topics=overview_data.get("main_topics", []),
            key_takeaways=overview_data.get("key_takeaways", []),
            unique_value=overview_data.get("unique_value", ""),
            writing_style=overview_data.get("writing_style", ""),
            practical_applications=overview_data.get("practical_applications", ""),
            primary_themes=themes_data.get("primary_themes", []),
            secondary_themes=themes_data.get("secondary_themes", []),
            recurring_ideas=themes_data.get("recurring_ideas", []),
            chapter_summaries=chapter_summaries_data.get("chapters", []),
            key_quotes=quotes,
            technical_concepts=concepts,
            image_descriptions=image_results,
        )

    async def analyze_single_agent(
        self,
        content: ExtractedContent,
        agent_name: str,
    ) -> dict:
        """Run a single agent for targeted analysis.

        Args:
            content: Extracted content from a document
            agent_name: Name of the agent to run

        Returns:
            Agent analysis results
        """
        agents = {
            "bibliography": self.bibliography_agent,
            "genre_tags": self.genre_tags_agent,
            "overview": self.overview_agent,
            "chapter_summary": self.chapter_summary_agent,
            "key_quotes": self.key_quotes_agent,
            "themes": self.themes_agent,
            "technical_concepts": self.technical_concepts_agent,
        }

        if agent_name not in agents:
            raise ValueError(f"Unknown agent: {agent_name}. Available: {list(agents.keys())}")

        agent = agents[agent_name]
        logger.info(f"Running single agent: {agent.name}")
        return await agent.analyze(content)
