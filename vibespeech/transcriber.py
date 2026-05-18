"""Core transcription module for VibeVoice.

This module provides the main transcription functionality using
OpenAI's Whisper model for automatic speech recognition (ASR).
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union

import whisper
import torch

logger = logging.getLogger(__name__)

# Supported audio/video formats
SUPPORTED_FORMATS = {
    ".mp3", ".mp4", ".wav", ".flac", ".ogg",
    ".m4a", ".webm", ".mkv", ".avi", ".mov"
}

# Available Whisper model sizes
MODEL_SIZES = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]


class Transcriber:
    """Handles audio/video transcription using Whisper.

    Attributes:
        model_size (str): Whisper model size to use.
        device (str): Device to run inference on ('cpu' or 'cuda').
        model: Loaded Whisper model instance.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        language: Optional[str] = None,
    ):
        """Initialize the Transcriber.

        Args:
            model_size: Whisper model size. Defaults to 'base'.
            device: Compute device ('cpu' or 'cuda'). Auto-detects if None.
            language: Language code (e.g. 'en', 'zh'). Auto-detect if None.
        """
        if model_size not in MODEL_SIZES:
            raise ValueError(
                f"Invalid model size '{model_size}'. Choose from: {MODEL_SIZES}"
            )

        self.model_size = model_size
        self.language = language
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        logger.info(
            "Transcriber initialized | model=%s | device=%s | language=%s",
            self.model_size,
            self.device,
            self.language or "auto",
        )

    def load_model(self) -> None:
        """Load the Whisper model into memory."""
        if self.model is not None:
            logger.debug("Model already loaded, skipping.")
            return

        logger.info("Loading Whisper model '%s'...", self.model_size)
        self.model = whisper.load_model(self.model_size, device=self.device)
        logger.info("Model loaded successfully.")

    def transcribe(
        self,
        audio_path: Union[str, Path],
        task: str = "transcribe",
        verbose: bool = False,
    ) -> dict:
        """Transcribe an audio or video file.

        Args:
            audio_path: Path to the audio/video file.
            task: 'transcribe' for transcription or 'translate' for English translation.
            verbose: If True, print progress to stdout.

        Returns:
            A dict containing:
                - 'text': Full transcribed text.
                - 'segments': List of timed segment dicts.
                - 'language': Detected or specified language code.

        Raises:
            FileNotFoundError: If the audio file does not exist.
            ValueError: If the file format is unsupported.
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if audio_path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{audio_path.suffix}'. "
                f"Supported: {SUPPORTED_FORMATS}"
            )

        # Lazy-load model on first transcription
        if self.model is None:
            self.load_model()

        logger.info("Transcribing: %s", audio_path.name)

        options = {
            "task": task,
            "verbose": verbose,
        }
        if self.language:
            options["language"] = self.language

        result = self.model.transcribe(str(audio_path), **options)

        logger.info(
            "Transcription complete | language=%s | segments=%d",
            result.get("language", "unknown"),
            len(result.get("segments", [])),
        )

        return result

    def transcribe_to_text(self, audio_path: Union[str, Path], **kwargs) -> str:
        """Convenience method that returns only the transcribed text string.

        Args:
            audio_path: Path to the audio/video file.
            **kwargs: Additional arguments forwarded to :meth:`transcribe`.

        Returns:
            The full transcribed text as a plain string.
        """
        result = self.transcribe(audio_path, **kwargs)
        return result["text"].strip()
