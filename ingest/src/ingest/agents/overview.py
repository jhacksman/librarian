"""Book overview summary agent."""

from ingest.agents.base import BaseAgent
from ingest.models import ExtractedContent
from ingest.utils.logging import get_logger

logger = get_logger("agents.overview")


class BookOverviewAgent(BaseAgent):
    """Agent for generating comprehensive book overviews."""

    @property
    def name(self) -> str:
        return "Book Overview Agent"

    @property
    def description(self) -> str:
        return "Generates comprehensive book summaries and overviews"

    @property
    def system_prompt(self) -> str:
        return """You are an expert book reviewer and summarizer.
Your task is to analyze book content and generate a comprehensive overview.

You must respond with a JSON object containing:
- summary: A comprehensive summary of the book (2-4 paragraphs, string)
- main_topics: List of main topics covered (array of strings, 5-10 items)
- key_takeaways: List of key takeaways or lessons (array of strings, 3-7 items)
- unique_value: What makes this book unique or valuable (string)
- writing_style: Description of the writing style (string)
- practical_applications: How readers can apply the knowledge (string)

Write in a clear, informative style. Be specific about what the book covers.
Focus on providing value to someone deciding whether to read this book."""

    async def analyze(self, content: ExtractedContent) -> dict:
        """Generate a comprehensive overview of the book.

        Args:
            content: Extracted content from a document

        Returns:
            Dictionary with book overview information
        """
        logger.info(f"Generating overview for: {content.title}")

        sample_text = self._get_sample_chapters(content, max_chapters=5, max_chars_per_chapter=8000)

        toc_text = ""
        if content.toc:
            toc_items = []
            for item in content.toc[:30]:
                indent = "  " * item.get("level", 0)
                toc_items.append(f"{indent}- {item.get('title', 'Unknown')}")
            toc_text = "\n".join(toc_items)

        chapter_list = ""
        if content.chapters:
            chapter_items = []
            for ch in content.chapters[:20]:
                chapter_items.append(f"- {ch.title} ({ch.word_count} words)")
            chapter_list = "\n".join(chapter_items)

        prompt = f"""Analyze the following book and generate a comprehensive overview.

**Book Title:** {content.title or "Unknown"}

**Word Count:** {content.word_count or "Unknown"}

**Table of Contents:**
{toc_text or "Not available"}

**Chapters:**
{chapter_list or "Not available"}

**Content Sample:**
{sample_text}

Generate a comprehensive overview and return as a JSON object."""

        result = await self.llm_client.generate_json(
            prompt=prompt,
            system_prompt=self.system_prompt,
            max_tokens=2000,
        )

        result = self._normalize_result(result)

        logger.debug(f"Generated overview with {len(result.get('summary', ''))} char summary")
        return result

    def _normalize_result(self, result: dict) -> dict:
        """Normalize the overview result.

        Args:
            result: Raw result from LLM

        Returns:
            Normalized result
        """
        main_topics = result.get("main_topics", [])
        if isinstance(main_topics, str):
            main_topics = [main_topics]

        key_takeaways = result.get("key_takeaways", [])
        if isinstance(key_takeaways, str):
            key_takeaways = [key_takeaways]

        return {
            "summary": result.get("summary", ""),
            "main_topics": main_topics[:10],
            "key_takeaways": key_takeaways[:7],
            "unique_value": result.get("unique_value", ""),
            "writing_style": result.get("writing_style", ""),
            "practical_applications": result.get("practical_applications", ""),
        }
