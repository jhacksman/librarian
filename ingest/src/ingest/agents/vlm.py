"""VLM agent for image processing and OCR."""


from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from ingest.config import VLMConfig
from ingest.models import ImageInfo
from ingest.utils.logging import get_logger

logger = get_logger("agents.vlm")


class VLMAgent:
    """Agent for processing images using Vision Language Model."""

    def __init__(self, config: VLMConfig) -> None:
        """Initialize the VLM agent.

        Args:
            config: VLM configuration
        """
        self.config = config
        self.client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
        )
        logger.info(f"Initialized VLM agent for {config.model} at {config.base_url}")

    @property
    def name(self) -> str:
        return "VLM Agent"

    @property
    def description(self) -> str:
        return "Processes images for OCR and visual content summarization"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def _generate(
        self,
        prompt: str,
        image_data: str,
        content_type: str,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a response from the VLM.

        Args:
            prompt: The user prompt
            image_data: Base64-encoded image data
            content_type: MIME type of the image
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{content_type};base64,{image_data}",
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=0.3,
        )

        return response.choices[0].message.content or ""

    async def process_image(self, image: ImageInfo) -> dict:
        """Process a single image.

        Args:
            image: Image information including base64 data

        Returns:
            Dictionary with image analysis results
        """
        logger.debug(f"Processing image: {image.filename}")

        if not image.base64_data:
            logger.warning(f"No image data for {image.filename}")
            return {
                "filename": image.filename,
                "description": "",
                "ocr_text": "",
                "image_type": "unknown",
            }

        description = await self._describe_image(image)
        ocr_text = await self._extract_text(image)
        image_type = await self._classify_image(image)

        return {
            "filename": image.filename,
            "description": description,
            "ocr_text": ocr_text,
            "image_type": image_type,
            "width": image.width,
            "height": image.height,
            "page_number": image.page_number,
        }

    async def _describe_image(self, image: ImageInfo) -> str:
        """Generate a description of the image.

        Args:
            image: Image information

        Returns:
            Text description of the image
        """
        prompt = """Describe this image in detail. Include:
1. What type of image it is (diagram, chart, screenshot, photo, etc.)
2. The main content and purpose
3. Any important details or labels
4. How it relates to technical content if applicable

Be concise but comprehensive."""

        try:
            return await self._generate(
                prompt=prompt,
                image_data=image.base64_data,
                content_type=image.content_type,
                max_tokens=500,
            )
        except Exception as e:
            logger.warning(f"Failed to describe image {image.filename}: {e}")
            return ""

    async def _extract_text(self, image: ImageInfo) -> str:
        """Extract text from the image using OCR.

        Args:
            image: Image information

        Returns:
            Extracted text from the image
        """
        prompt = """Extract all visible text from this image.
Include:
- Code snippets
- Labels and captions
- Any written text
- Numbers and data

Format the text to preserve structure where possible.
If there is no text, respond with "NO_TEXT"."""

        try:
            result = await self._generate(
                prompt=prompt,
                image_data=image.base64_data,
                content_type=image.content_type,
                max_tokens=1000,
            )
            if result.strip().upper() == "NO_TEXT":
                return ""
            return result
        except Exception as e:
            logger.warning(f"Failed to extract text from {image.filename}: {e}")
            return ""

    async def _classify_image(self, image: ImageInfo) -> str:
        """Classify the type of image.

        Args:
            image: Image information

        Returns:
            Image type classification
        """
        prompt = """Classify this image into one of these categories:
- diagram: Technical diagrams, flowcharts, architecture diagrams
- chart: Graphs, charts, data visualizations
- code: Screenshots of code or terminal output
- screenshot: UI screenshots, application interfaces
- photo: Photographs
- illustration: Drawings, illustrations, artwork
- table: Tables of data
- other: Anything else

Respond with just the category name."""

        try:
            result = await self._generate(
                prompt=prompt,
                image_data=image.base64_data,
                content_type=image.content_type,
                max_tokens=20,
            )
            return result.strip().lower()
        except Exception as e:
            logger.warning(f"Failed to classify image {image.filename}: {e}")
            return "unknown"

    async def process_images(self, images: list[ImageInfo]) -> list[dict]:
        """Process multiple images.

        Args:
            images: List of images to process

        Returns:
            List of image analysis results
        """
        logger.info(f"Processing {len(images)} images")

        results = []
        for i, image in enumerate(images):
            logger.debug(f"Processing image {i + 1}/{len(images)}: {image.filename}")
            result = await self.process_image(image)
            results.append(result)

        logger.info(f"Processed {len(results)} images")
        return results

    async def summarize_all_images(self, image_results: list[dict]) -> str:
        """Generate a summary of all images in a document.

        Args:
            image_results: List of processed image results

        Returns:
            Summary of all images
        """
        if not image_results:
            return ""

        image_descriptions = []
        for result in image_results:
            desc = f"- {result['filename']} ({result['image_type']}): {result['description']}"
            if result.get("ocr_text"):
                desc += f"\n  Text content: {result['ocr_text'][:200]}..."
            image_descriptions.append(desc)

        summary = f"The document contains {len(image_results)} images:\n\n"
        summary += "\n\n".join(image_descriptions)

        return summary
