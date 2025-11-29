"""Technical concepts extraction agent."""

from ingest.agents.base import BaseAgent
from ingest.models import ExtractedContent, TechnicalConcept
from ingest.utils.logging import get_logger

logger = get_logger("agents.technical_concepts")


class TechnicalConceptsAgent(BaseAgent):
    """Agent for extracting technical concepts and definitions."""

    @property
    def name(self) -> str:
        return "Technical Concepts Agent"

    @property
    def description(self) -> str:
        return "Extracts technical terms, concepts, and their definitions"

    @property
    def system_prompt(self) -> str:
        return """You are a technical expert at identifying and explaining concepts.
Your task is to extract technical terms, concepts, and definitions from book content.

You must respond with a JSON object containing:
- concepts: Array of concept objects, each with:
  - term: The technical term or concept name (string)
  - definition: Clear definition as explained in the book (string)
  - category: Category of the concept (string, e.g., "programming", "security", "networking")
  - related_terms: Related terms or concepts (array of strings)
  - importance: How important this concept is to the book ("core", "supporting", "mentioned")

Extract 10-30 of the most important technical concepts.
Prioritize concepts that are:
1. Central to the book's subject matter
2. Clearly defined or explained
3. Likely to be searched for by readers
4. Technical or domain-specific (not general vocabulary)

Use the book's own definitions when available."""

    async def analyze(self, content: ExtractedContent) -> dict:
        """Extract technical concepts from the book.

        Args:
            content: Extracted content from a document

        Returns:
            Dictionary with extracted concepts
        """
        logger.info(f"Extracting technical concepts from: {content.title}")

        all_concepts = []

        if content.chapters:
            for chapter in content.chapters[:8]:
                chapter_concepts = await self._extract_from_chapter(
                    chapter.title, chapter.content
                )
                all_concepts.extend(chapter_concepts)
        else:
            all_concepts = await self._extract_from_text(content.raw_text)

        all_concepts = self._deduplicate_concepts(all_concepts)

        logger.info(f"Extracted {len(all_concepts)} unique technical concepts")
        return {"concepts": all_concepts}

    async def _extract_from_chapter(self, chapter_title: str, chapter_text: str) -> list[dict]:
        """Extract concepts from a single chapter.

        Args:
            chapter_title: Title of the chapter
            chapter_text: Text content of the chapter

        Returns:
            List of concept dictionaries
        """
        truncated_text = self._truncate_text(chapter_text, max_chars=12000)

        prompt = f"""Extract technical concepts from the following chapter.

**Chapter:** {chapter_title}

**Content:**
{truncated_text}

Extract 5-10 technical concepts and return as a JSON object."""

        result = await self.llm_client.generate_json(
            prompt=prompt,
            system_prompt=self.system_prompt,
            max_tokens=2000,
        )

        return self._normalize_concepts(result.get("concepts", []))

    async def _extract_from_text(self, text: str) -> list[dict]:
        """Extract concepts from raw text.

        Args:
            text: Raw text content

        Returns:
            List of concept dictionaries
        """
        truncated_text = self._truncate_text(text, max_chars=40000)

        prompt = f"""Extract technical concepts from the following book content.

**Content:**
{truncated_text}

Extract 20-30 technical concepts and return as a JSON object."""

        result = await self.llm_client.generate_json(
            prompt=prompt,
            system_prompt=self.system_prompt,
            max_tokens=4000,
        )

        return self._normalize_concepts(result.get("concepts", []))

    def _normalize_concepts(self, concepts: list) -> list[dict]:
        """Normalize concept results.

        Args:
            concepts: Raw concepts from LLM

        Returns:
            Normalized list of concept dictionaries
        """
        normalized = []
        for concept in concepts:
            if isinstance(concept, dict) and concept.get("term"):
                related = concept.get("related_terms", [])
                if isinstance(related, str):
                    related = [related]

                normalized.append({
                    "term": concept.get("term", ""),
                    "definition": concept.get("definition", ""),
                    "category": concept.get("category", "general"),
                    "related_terms": related,
                    "importance": concept.get("importance", "supporting"),
                })
        return normalized

    def _deduplicate_concepts(self, concepts: list[dict]) -> list[dict]:
        """Remove duplicate concepts, keeping the most detailed version.

        Args:
            concepts: List of concept dictionaries

        Returns:
            Deduplicated list
        """
        seen = {}
        for concept in concepts:
            term_lower = concept.get("term", "").lower().strip()
            if not term_lower:
                continue

            if term_lower not in seen:
                seen[term_lower] = concept
            else:
                existing = seen[term_lower]
                if len(concept.get("definition", "")) > len(existing.get("definition", "")):
                    seen[term_lower] = concept

        return list(seen.values())[:30]

    def to_concept_models(self, data: dict) -> list[TechnicalConcept]:
        """Convert extracted data to TechnicalConcept models.

        Args:
            data: Extracted concepts data

        Returns:
            List of TechnicalConcept model instances
        """
        concepts = []
        for c in data.get("concepts", []):
            concepts.append(
                TechnicalConcept(
                    term=c.get("term", ""),
                    definition=c.get("definition", ""),
                    category=c.get("category"),
                    related_terms=c.get("related_terms", []),
                )
            )
        return concepts
