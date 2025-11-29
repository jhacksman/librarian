"""Key quotes extraction agent."""

from ingest.agents.base import BaseAgent
from ingest.models import ExtractedContent, Quote
from ingest.utils.logging import get_logger

logger = get_logger("agents.key_quotes")


class KeyQuotesAgent(BaseAgent):
    """Agent for extracting key quotes and notable passages."""

    @property
    def name(self) -> str:
        return "Key Quotes Agent"

    @property
    def description(self) -> str:
        return "Extracts notable quotes and important passages from books"

    @property
    def system_prompt(self) -> str:
        return """You are an expert at identifying important and memorable quotes from books.
Your task is to extract the most valuable, insightful, or memorable quotes.

You must respond with a JSON object containing:
- quotes: Array of quote objects, each with:
  - text: The exact quote text (string)
  - context: Brief context about where/why this quote appears (string)
  - significance: Why this quote is important or memorable (string)
  - chapter: Chapter name or number if known (string or null)
  - page: Page number if known (integer or null)

Extract 5-15 of the most valuable quotes. Prioritize:
1. Key insights or wisdom
2. Memorable phrases
3. Important definitions or explanations
4. Practical advice
5. Thought-provoking statements

Be accurate with the quote text - use the exact wording from the book."""

    async def analyze(self, content: ExtractedContent) -> dict:
        """Extract key quotes from the book.

        Args:
            content: Extracted content from a document

        Returns:
            Dictionary with extracted quotes
        """
        logger.info(f"Extracting key quotes from: {content.title}")

        all_quotes = []

        if content.chapters:
            for chapter in content.chapters[:10]:
                chapter_quotes = await self._extract_from_chapter(chapter.title, chapter.content)
                for quote in chapter_quotes:
                    quote["chapter"] = chapter.title
                all_quotes.extend(chapter_quotes)
        else:
            all_quotes = await self._extract_from_text(content.raw_text)

        all_quotes = all_quotes[:15]

        logger.info(f"Extracted {len(all_quotes)} key quotes")
        return {"quotes": all_quotes}

    async def _extract_from_chapter(self, chapter_title: str, chapter_text: str) -> list[dict]:
        """Extract quotes from a single chapter.

        Args:
            chapter_title: Title of the chapter
            chapter_text: Text content of the chapter

        Returns:
            List of quote dictionaries
        """
        truncated_text = self._truncate_text(chapter_text, max_chars=12000)

        prompt = f"""Extract key quotes from the following chapter.

**Chapter:** {chapter_title}

**Content:**
{truncated_text}

Extract 2-5 of the most valuable quotes and return as a JSON object."""

        result = await self.llm_client.generate_json(
            prompt=prompt,
            system_prompt=self.system_prompt,
            max_tokens=1500,
        )

        return self._normalize_quotes(result.get("quotes", []))

    async def _extract_from_text(self, text: str) -> list[dict]:
        """Extract quotes from raw text.

        Args:
            text: Raw text content

        Returns:
            List of quote dictionaries
        """
        truncated_text = self._truncate_text(text, max_chars=40000)

        prompt = f"""Extract key quotes from the following book content.

**Content:**
{truncated_text}

Extract 10-15 of the most valuable quotes and return as a JSON object."""

        result = await self.llm_client.generate_json(
            prompt=prompt,
            system_prompt=self.system_prompt,
            max_tokens=3000,
        )

        return self._normalize_quotes(result.get("quotes", []))

    def _normalize_quotes(self, quotes: list) -> list[dict]:
        """Normalize quote results.

        Args:
            quotes: Raw quotes from LLM

        Returns:
            Normalized list of quote dictionaries
        """
        normalized = []
        for quote in quotes:
            if isinstance(quote, dict) and quote.get("text"):
                normalized.append({
                    "text": quote.get("text", ""),
                    "context": quote.get("context", ""),
                    "significance": quote.get("significance", ""),
                    "chapter": quote.get("chapter"),
                    "page": quote.get("page"),
                })
        return normalized

    def to_quote_models(self, data: dict) -> list[Quote]:
        """Convert extracted data to Quote models.

        Args:
            data: Extracted quotes data

        Returns:
            List of Quote model instances
        """
        quotes = []
        for q in data.get("quotes", []):
            quotes.append(
                Quote(
                    text=q.get("text", ""),
                    context=q.get("context"),
                    significance=q.get("significance"),
                    chapter=q.get("chapter"),
                    page=q.get("page"),
                )
            )
        return quotes
