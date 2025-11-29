"""Bibliography extraction agent."""

from ingest.agents.base import BaseAgent
from ingest.models import Bibliography, ExtractedContent
from ingest.utils.logging import get_logger

logger = get_logger("agents.bibliography")


class BibliographyAgent(BaseAgent):
    """Agent for extracting bibliographic information from books."""

    @property
    def name(self) -> str:
        return "Bibliography Agent"

    @property
    def description(self) -> str:
        return "Extracts bibliographic information including author, year, publisher, ISBN, and subjects"

    @property
    def system_prompt(self) -> str:
        return """You are a librarian expert at extracting bibliographic information from books.
Your task is to analyze book content and extract accurate bibliographic metadata.

You must respond with a JSON object containing the following fields:
- title: The book's title (string)
- authors: List of author names (array of strings)
- publisher: Publisher name if mentioned (string or null)
- publication_year: Year of publication if mentioned (integer or null)
- isbn: ISBN-10 if found (string or null)
- isbn13: ISBN-13 if found (string or null)
- language: Primary language of the book (string, default "en")
- subjects: List of subject categories (array of strings)
- description: A brief description of the book (string)
- edition: Edition information if mentioned (string or null)
- series: Series name if part of a series (string or null)
- series_number: Number in series if applicable (integer or null)

Extract information from the content provided. If information is not available, use null.
Be accurate and only include information that is clearly stated or strongly implied."""

    async def analyze(self, content: ExtractedContent) -> dict:
        """Extract bibliographic information from the content.

        Args:
            content: Extracted content from a document

        Returns:
            Dictionary with bibliographic information
        """
        logger.info(f"Extracting bibliography for: {content.title}")

        existing_metadata = content.metadata
        sample_text = self._get_sample_chapters(content, max_chapters=2, max_chars_per_chapter=5000)

        prompt = f"""Analyze the following book content and extract bibliographic information.

**Existing Metadata (may be incomplete or inaccurate):**
{self._format_existing_metadata(existing_metadata)}

**Book Content Sample:**
{sample_text}

Extract and return the bibliographic information as a JSON object."""

        result = await self.llm_client.generate_json(
            prompt=prompt,
            system_prompt=self.system_prompt,
            max_tokens=1000,
        )

        result = self._merge_with_existing(result, existing_metadata)

        logger.debug(f"Extracted bibliography: {result}")
        return result

    def _format_existing_metadata(self, metadata: dict) -> str:
        """Format existing metadata for the prompt.

        Args:
            metadata: Existing metadata dictionary

        Returns:
            Formatted string
        """
        if not metadata:
            return "No existing metadata available."

        lines = []
        for key, value in metadata.items():
            if value:
                lines.append(f"- {key}: {value}")

        return "\n".join(lines) if lines else "No existing metadata available."

    def _merge_with_existing(self, extracted: dict, existing: dict) -> dict:
        """Merge extracted data with existing metadata, preferring extracted when available.

        Args:
            extracted: Extracted bibliographic data
            existing: Existing metadata

        Returns:
            Merged dictionary
        """
        result = {
            "title": extracted.get("title") or existing.get("title"),
            "authors": extracted.get("authors") or existing.get("authors", []),
            "publisher": extracted.get("publisher") or existing.get("publisher"),
            "publication_year": extracted.get("publication_year"),
            "isbn": extracted.get("isbn") or existing.get("isbn"),
            "isbn13": extracted.get("isbn13") or existing.get("isbn13"),
            "language": extracted.get("language") or existing.get("language", "en"),
            "subjects": extracted.get("subjects") or existing.get("subjects", []),
            "description": extracted.get("description") or existing.get("description"),
            "edition": extracted.get("edition"),
            "series": extracted.get("series"),
            "series_number": extracted.get("series_number"),
        }

        if result.get("publication_year") is None and existing.get("date"):
            try:
                year_str = str(existing["date"])[:4]
                result["publication_year"] = int(year_str)
            except (ValueError, TypeError):
                pass

        return result

    def to_bibliography(self, data: dict) -> Bibliography:
        """Convert extracted data to a Bibliography model.

        Args:
            data: Extracted bibliographic data

        Returns:
            Bibliography model instance
        """
        return Bibliography(
            title=data.get("title", "Unknown"),
            authors=data.get("authors", []),
            publisher=data.get("publisher"),
            publication_year=data.get("publication_year"),
            isbn=data.get("isbn"),
            isbn13=data.get("isbn13"),
            language=data.get("language"),
            subjects=data.get("subjects", []),
            description=data.get("description"),
            edition=data.get("edition"),
            series=data.get("series"),
            series_number=data.get("series_number"),
        )
