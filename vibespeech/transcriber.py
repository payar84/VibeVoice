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
        model_size: str = "small",  # bumped from 'base' — noticeably better accuracy for minimal speed cost
        device: Optional[str] = None,
        language: Optional[str] = None,
    ):
        """Initialize the Transcriber.

        Args:
            model_size: Whisper model size. Defaults to 'small'.
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
            ValueError: If the file format is 
