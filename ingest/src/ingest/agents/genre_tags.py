"""Genre and tags classification agent."""

from ingest.agents.base import BaseAgent
from ingest.models import ExtractedContent
from ingest.utils.logging import get_logger

logger = get_logger("agents.genre_tags")

DEFAULT_TAGS = [
    "Information Security",
    "Python",
    "AI/Machine Learning",
    "Web Development",
    "Systems Programming",
    "Networking",
    "Databases",
    "DevOps",
    "Cryptography",
    "Reverse Engineering",
    "Game Development",
    "Mobile Development",
    "Cloud Computing",
    "Data Science",
    "Software Engineering",
    "Linux/Unix",
    "Hardware",
    "Electronics",
    "Robotics",
    "Mathematics",
    "Science",
    "Business",
    "Design",
    "Writing",
]


class GenreTagsAgent(BaseAgent):
    """Agent for classifying books by genre and tags."""

    def __init__(self, llm_client, predefined_tags: list[str] | None = None) -> None:
        """Initialize the genre/tags agent.

        Args:
            llm_client: LLM client for generating responses
            predefined_tags: Optional list of predefined tags to use
        """
        super().__init__(llm_client)
        self.predefined_tags = predefined_tags or DEFAULT_TAGS

    @property
    def name(self) -> str:
        return "Genre/Tags Agent"

    @property
    def description(self) -> str:
        return "Classifies books by genre and assigns relevant tags"

    @property
    def system_prompt(self) -> str:
        tags_list = ", ".join(self.predefined_tags)
        return f"""You are an expert librarian specializing in technical book classification.
Your task is to analyze book content and assign appropriate genres and tags.

Available predefined tags: {tags_list}

You must respond with a JSON object containing:
- genres: List of 1-3 primary genres (array of strings)
- tags: List of 3-10 relevant tags from the predefined list or custom tags (array of strings)
- target_audience: Who this book is for (string, e.g., "beginners", "intermediate developers", "security professionals")
- difficulty_level: Difficulty level (string: "beginner", "intermediate", "advanced", "expert")
- prerequisites: List of knowledge/skills needed to understand this book (array of strings)

Prefer tags from the predefined list when applicable, but you can add custom tags if needed.
Be specific and accurate in your classifications."""

    async def analyze(self, content: ExtractedContent) -> dict:
        """Classify the book by genre and tags.

        Args:
            content: Extracted content from a document

        Returns:
            Dictionary with genres, tags, and audience information
        """
        logger.info(f"Classifying genres/tags for: {content.title}")

        sample_text = self._get_sample_chapters(content, max_chapters=3, max_chars_per_chapter=8000)

        toc_text = ""
        if content.toc:
            toc_items = []
            for item in content.toc[:20]:
                toc_items.append(f"- {item.get('title', 'Unknown')}")
            toc_text = "\n".join(toc_items)

        prompt = f"""Analyze the following book and classify it by genre and tags.

**Book Title:** {content.title or "Unknown"}

**Table of Contents:**
{toc_text or "Not available"}

**Content Sample:**
{sample_text}

Classify this book and return the results as a JSON object."""

        result = await self.llm_client.generate_json(
            prompt=prompt,
            system_prompt=self.system_prompt,
            max_tokens=500,
        )

        result = self._normalize_result(result)

        logger.debug(f"Classification result: {result}")
        return result

    def _normalize_result(self, result: dict) -> dict:
        """Normalize the classification result.

        Args:
            result: Raw result from LLM

        Returns:
            Normalized result
        """
        genres = result.get("genres", [])
        if isinstance(genres, str):
            genres = [genres]
        genres = [g.strip() for g in genres if g and isinstance(g, str)]

        tags = result.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        tags = [t.strip() for t in tags if t and isinstance(t, str)]

        prerequisites = result.get("prerequisites", [])
        if isinstance(prerequisites, str):
            prerequisites = [prerequisites]
        prerequisites = [p.strip() for p in prerequisites if p and isinstance(p, str)]

        return {
            "genres": genres[:3],
            "tags": tags[:10],
            "target_audience": result.get("target_audience", "general"),
            "difficulty_level": result.get("difficulty_level", "intermediate"),
            "prerequisites": prerequisites,
        }
