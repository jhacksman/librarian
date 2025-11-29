"""Base agent interface and LLM client for the committee."""

import json
from abc import ABC, abstractmethod
from typing import Any

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from ingest.config import LLMConfig
from ingest.models import ExtractedContent
from ingest.utils.logging import get_logger

logger = get_logger("agents.base")


class LLMClient:
    """Client for interacting with vLLM-served models via OpenAI-compatible API."""

    def __init__(self, config: LLMConfig) -> None:
        """Initialize the LLM client.

        Args:
            config: LLM configuration
        """
        self.config = config
        self.client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
        )
        logger.info(f"Initialized LLM client for {config.model} at {config.base_url}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: dict | None = None,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate (uses config default if not specified)
            temperature: Temperature for generation (uses config default if not specified)
            response_format: Optional response format specification for JSON mode

        Returns:
            Generated text response
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
        }

        if response_format:
            kwargs["response_format"] = response_format

        logger.debug(f"Generating response with {len(prompt)} char prompt")

        response = await self.client.chat.completions.create(**kwargs)

        result = response.choices[0].message.content or ""
        logger.debug(f"Generated {len(result)} char response")

        return result

    async def generate_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Generate a JSON response from the LLM.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Parsed JSON response as a dictionary
        """
        response = await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            json_match = response.find("{")
            if json_match != -1:
                try:
                    return json.loads(response[json_match:])
                except json.JSONDecodeError:
                    pass
            return {"error": "Failed to parse response", "raw": response}


class BaseAgent(ABC):
    """Abstract base class for LLM committee agents."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize the agent.

        Args:
            llm_client: LLM client for generating responses
        """
        self.llm_client = llm_client

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the agent's name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of what this agent does."""
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        pass

    @abstractmethod
    async def analyze(self, content: ExtractedContent) -> dict:
        """Analyze the extracted content.

        Args:
            content: Extracted content from a document

        Returns:
            Analysis results as a dictionary
        """
        pass

    def _truncate_text(self, text: str, max_chars: int = 50000) -> str:
        """Truncate text to a maximum number of characters.

        Args:
            text: Text to truncate
            max_chars: Maximum characters to keep

        Returns:
            Truncated text
        """
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n\n[... content truncated ...]"

    def _get_sample_chapters(
        self,
        content: ExtractedContent,
        max_chapters: int = 3,
        max_chars_per_chapter: int = 10000,
    ) -> str:
        """Get a sample of chapters for analysis.

        Args:
            content: Extracted content
            max_chapters: Maximum number of chapters to include
            max_chars_per_chapter: Maximum characters per chapter

        Returns:
            Formatted string with chapter samples
        """
        if not content.chapters:
            return self._truncate_text(content.raw_text, max_chars_per_chapter * max_chapters)

        samples = []
        for chapter in content.chapters[:max_chapters]:
            chapter_text = self._truncate_text(chapter.content, max_chars_per_chapter)
            samples.append(f"## {chapter.title}\n\n{chapter_text}")

        return "\n\n---\n\n".join(samples)
