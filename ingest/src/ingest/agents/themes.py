"""Themes identification agent."""

from ingest.agents.base import BaseAgent
from ingest.models import ExtractedContent
from ingest.utils.logging import get_logger

logger = get_logger("agents.themes")


class ThemesAgent(BaseAgent):
    """Agent for identifying key themes in books."""

    @property
    def name(self) -> str:
        return "Themes Agent"

    @property
    def description(self) -> str:
        return "Identifies key themes and recurring ideas in books"

    @property
    def system_prompt(self) -> str:
        return """You are an expert literary analyst specializing in identifying themes.
Your task is to analyze book content and identify the key themes and recurring ideas.

You must respond with a JSON object containing:
- primary_themes: List of 3-5 primary themes (array of objects with "name" and "description")
- secondary_themes: List of 2-4 secondary themes (array of objects with "name" and "description")
- recurring_ideas: Ideas that appear throughout the book (array of strings)
- philosophical_underpinnings: Any philosophical or theoretical foundations (array of strings)
- narrative_patterns: Patterns in how information is presented (array of strings)

For each theme, provide:
- name: Short name for the theme (string)
- description: 1-2 sentence description of how this theme manifests in the book (string)

Be specific to this book. Avoid generic themes unless they are truly central."""

    async def analyze(self, content: ExtractedContent) -> dict:
        """Identify themes in the book.

        Args:
            content: Extracted content from a document

        Returns:
            Dictionary with identified themes
        """
        logger.info(f"Identifying themes for: {content.title}")

        sample_text = self._get_sample_chapters(content, max_chapters=5, max_chars_per_chapter=8000)

        toc_text = ""
        if content.toc:
            toc_items = []
            for item in content.toc[:25]:
                toc_items.append(f"- {item.get('title', 'Unknown')}")
            toc_text = "\n".join(toc_items)

        prompt = f"""Analyze the following book and identify its key themes.

**Book Title:** {content.title or "Unknown"}

**Table of Contents:**
{toc_text or "Not available"}

**Content Sample:**
{sample_text}

Identify the themes and return as a JSON object."""

        result = await self.llm_client.generate_json(
            prompt=prompt,
            system_prompt=self.system_prompt,
            max_tokens=1500,
        )

        result = self._normalize_result(result)

        logger.debug(f"Identified {len(result.get('primary_themes', []))} primary themes")
        return result

    def _normalize_result(self, result: dict) -> dict:
        """Normalize the themes result.

        Args:
            result: Raw result from LLM

        Returns:
            Normalized result
        """
        def normalize_themes(themes: list) -> list[dict]:
            normalized = []
            for theme in themes:
                if isinstance(theme, dict):
                    normalized.append({
                        "name": theme.get("name", "Unknown"),
                        "description": theme.get("description", ""),
                    })
                elif isinstance(theme, str):
                    normalized.append({
                        "name": theme,
                        "description": "",
                    })
            return normalized

        def normalize_list(items: list | str) -> list[str]:
            if isinstance(items, str):
                return [items]
            return [str(item) for item in items if item]

        return {
            "primary_themes": normalize_themes(result.get("primary_themes", []))[:5],
            "secondary_themes": normalize_themes(result.get("secondary_themes", []))[:4],
            "recurring_ideas": normalize_list(result.get("recurring_ideas", [])),
            "philosophical_underpinnings": normalize_list(result.get("philosophical_underpinnings", [])),
            "narrative_patterns": normalize_list(result.get("narrative_patterns", [])),
        }
