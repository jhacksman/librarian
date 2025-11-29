"""Batch processing script for initial book ingestion."""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from ingest.config import IngestConfig
from ingest.models import BatchProgress
from ingest.pipeline import IngestPipeline
from ingest.utils.logging import get_logger, setup_logging

logger = get_logger("batch")


class BatchProcessor:
    """Batch processor for ingesting multiple books."""

    def __init__(
        self,
        config: IngestConfig,
        progress_file: str | None = None,
    ) -> None:
        """Initialize the batch processor.

        Args:
            config: Ingest configuration
            progress_file: Optional file to save/resume progress
        """
        self.config = config
        self.progress_file = Path(progress_file) if progress_file else None
        self.pipeline = IngestPipeline(config)
        self.progress = BatchProgress(
            total_files=0,
            processed_files=0,
            successful_files=0,
            failed_files=0,
            current_file=None,
            start_time=datetime.now(),
            errors={},
        )

    def _save_progress(self) -> None:
        """Save progress to file."""
        if not self.progress_file:
            return

        data = {
            "total_files": self.progress.total_files,
            "processed_files": self.progress.processed_files,
            "successful_files": self.progress.successful_files,
            "failed_files": self.progress.failed_files,
            "processed_paths": list(self.progress.processed_paths),
            "errors": self.progress.errors,
            "start_time": self.progress.start_time.isoformat(),
        }

        with open(self.progress_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved progress to {self.progress_file}")

    def _load_progress(self) -> set[str]:
        """Load progress from file.

        Returns:
            Set of already processed file paths
        """
        if not self.progress_file or not self.progress_file.exists():
            return set()

        try:
            with open(self.progress_file) as f:
                data = json.load(f)

            processed = set(data.get("processed_paths", []))
            logger.info(f"Loaded progress: {len(processed)} files already processed")
            return processed
        except Exception as e:
            logger.warning(f"Failed to load progress file: {e}")
            return set()

    def discover_files(
        self,
        directory: Path,
        recursive: bool = True,
    ) -> list[Path]:
        """Discover book files in a directory.

        Args:
            directory: Directory to search
            recursive: Whether to search subdirectories

        Returns:
            List of book file paths
        """
        pattern = "**/*" if recursive else "*"
        files = []

        for ext in [".epub", ".pdf"]:
            files.extend(directory.glob(f"{pattern}{ext}"))

        files.sort(key=lambda p: p.name.lower())

        logger.info(f"Discovered {len(files)} book files")
        return files

    async def process_file(self, file_path: Path) -> bool:
        """Process a single file.

        Args:
            file_path: Path to the book file

        Returns:
            True if successful, False otherwise
        """
        self.progress.current_file = str(file_path)
        logger.info(f"Processing: {file_path.name}")

        try:
            book_doc = await self.pipeline.ingest_book(file_path)
            logger.info(
                f"Successfully ingested: {book_doc.analysis.bibliography.title} "
                f"({book_doc.total_chunks} chunks)"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")
            self.progress.errors[str(file_path)] = str(e)
            return False

    async def run(
        self,
        directory: str | Path,
        recursive: bool = True,
        resume: bool = True,
        max_files: int | None = None,
    ) -> BatchProgress:
        """Run batch processing on a directory.

        Args:
            directory: Directory containing books
            recursive: Whether to search subdirectories
            resume: Whether to resume from previous progress
            max_files: Maximum number of files to process

        Returns:
            BatchProgress with results
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        files = self.discover_files(directory, recursive)

        if max_files:
            files = files[:max_files]

        already_processed = self._load_progress() if resume else set()

        files_to_process = [f for f in files if str(f) not in already_processed]

        self.progress.total_files = len(files)
        self.progress.processed_files = len(already_processed)
        self.progress.successful_files = len(already_processed)
        self.progress.processed_paths = already_processed

        logger.info(
            f"Batch processing: {len(files_to_process)} files to process "
            f"({len(already_processed)} already done)"
        )

        for i, file_path in enumerate(files_to_process):
            logger.info(
                f"[{i + 1}/{len(files_to_process)}] Processing: {file_path.name}"
            )

            success = await self.process_file(file_path)

            self.progress.processed_files += 1
            self.progress.processed_paths.add(str(file_path))

            if success:
                self.progress.successful_files += 1
            else:
                self.progress.failed_files += 1

            self._save_progress()

            self._log_progress()

        self.progress.end_time = datetime.now()
        self.progress.current_file = None

        self._log_summary()

        return self.progress

    def _log_progress(self) -> None:
        """Log current progress."""
        pct = (self.progress.processed_files / self.progress.total_files * 100) if self.progress.total_files > 0 else 0

        elapsed = datetime.now() - self.progress.start_time
        rate = self.progress.processed_files / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0

        remaining = self.progress.total_files - self.progress.processed_files
        eta_seconds = remaining / rate if rate > 0 else 0
        eta_minutes = eta_seconds / 60

        logger.info(
            f"Progress: {self.progress.processed_files}/{self.progress.total_files} "
            f"({pct:.1f}%) - {self.progress.successful_files} success, "
            f"{self.progress.failed_files} failed - ETA: {eta_minutes:.1f} min"
        )

    def _log_summary(self) -> None:
        """Log final summary."""
        elapsed = self.progress.end_time - self.progress.start_time if self.progress.end_time else datetime.now() - self.progress.start_time

        logger.info("=" * 60)
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total files: {self.progress.total_files}")
        logger.info(f"Processed: {self.progress.processed_files}")
        logger.info(f"Successful: {self.progress.successful_files}")
        logger.info(f"Failed: {self.progress.failed_files}")
        logger.info(f"Duration: {elapsed}")

        if self.progress.errors:
            logger.info("\nFailed files:")
            for path, error in self.progress.errors.items():
                logger.info(f"  - {Path(path).name}: {error}")


async def main() -> None:
    """Main entry point for batch processing."""
    import argparse

    parser = argparse.ArgumentParser(description="Batch process books for ingestion")
    parser.add_argument("directory", help="Directory containing books")
    parser.add_argument("--config", "-c", help="Path to config file")
    parser.add_argument("--progress", "-p", help="Progress file for resume support")
    parser.add_argument("--no-recursive", action="store_true", help="Don't search subdirectories")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from previous progress")
    parser.add_argument("--max-files", "-n", type=int, help="Maximum files to process")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--debug", "-d", action="store_true", help="Debug logging")

    args = parser.parse_args()

    log_level = "DEBUG" if args.debug else ("INFO" if args.verbose else "WARNING")
    setup_logging(log_level)

    config = IngestConfig.load(args.config) if args.config else IngestConfig()

    processor = BatchProcessor(
        config=config,
        progress_file=args.progress or "batch_progress.json",
    )

    try:
        progress = await processor.run(
            directory=args.directory,
            recursive=not args.no_recursive,
            resume=not args.no_resume,
            max_files=args.max_files,
        )

        if progress.failed_files > 0:
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nBatch processing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
