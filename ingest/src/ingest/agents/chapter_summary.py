"""Chapter-by-chapter summary agent."""

from ingest.agents.base import BaseAgent
from ingest.models import ChapterInfo, ExtractedContent
from ingest.utils.logging import get_logger

logger = get_logger("agents.chapter_summary")


class ChapterSummaryAgent(BaseAgent):
    """Agent for generating chapter-by-chapter summaries."""

    @property
    def name(self) -> str:
        return "Chapter Summary Agent"

    @property
    def description(self) -> str:
        return "Generates summaries for each chapter of a book"

    @property
    def system_prompt(self) -> str:
        return """You are an expert at summarizing book chapters.
Your task is to analyze a chapter and generate a concise but comprehensive summary.

You must respond with a JSON object containing:
- summary: A concise summary of the chapter (1-2 paragraphs, string)
- key_points: List of key points covered (array of strings, 3-5 items)
- concepts_introduced: New concepts or terms introduced (array of strings)
- practical_examples: Any practical examples or exercises mentioned (array of strings)

Be specific and accurate. Focus on the main ideas and practical value."""

    async def analyze(self, content: ExtractedContent) -> dict:
        """Generate summaries for all chapters.

        Args:
            content: Extracted content from a document

        Returns:
            Dictionary with chapter summaries
        """
        logger.info(f"Generating chapter summaries for: {content.title}")

        if not content.chapters:
            logger.warning("No chapters found in content")
            return {"chapters": []}

        chapter_summaries = []
        for chapter in content.chapters:
            summary = await self._summarize_chapter(chapter)
            chapter_summaries.append({
                "number": chapter.number,
                "title": chapter.title,
                "word_count": chapter.word_count,
                **summary,
            })

        logger.info(f"Generated summaries for {len(chapter_summaries)} chapters")
        return {"chapters": chapter_summaries}

    async def _summarize_chapter(self, chapter: ChapterInfo) -> dict:
        """Summarize a single chapter.

        Args:
            chapter: Chapter information

        Returns:
            Dictionary with chapter summary
        """
        logger.debug(f"Summarizing chapter: {chapter.title}")

        chapter_text = self._truncate_text(chapter.content, max_chars=15000)

        prompt = f"""Summarize the following chapter.

**Chapter Title:** {chapter.title}
**Chapter Number:** {chapter.number}
**Word Count:** {chapter.word_count}

**Content:**
{chapter_text}

Generate a summary and return as a JSON object."""

        result = await self.llm_client.generate_json(
            prompt=prompt,
            system_prompt=self.system_prompt,
            max_tokens=800,
        )

        return self._normalize_result(result)

    def _normalize_result(self, result: dict) -> dict:
        """Normalize the chapter summary result.

        Args:
            result: Raw result from LLM

        Returns:
            Normalized result
        """
        key_points = result.get("key_points", [])
        if isinstance(key_points, str):
            key_points = [key_points]

        concepts = result.get("concepts_introduced", [])
        if isinstance(concepts, str):
            concepts = [concepts]

        examples = result.get("practical_examples", [])
        if isinstance(examples, str):
            examples = [examples]

        return {
            "summary": result.get("summary", ""),
            "key_points": key_points[:5],
            "concepts_introduced": concepts,
            "practical_examples": examples,
        }
