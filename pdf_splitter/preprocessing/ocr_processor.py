"""
High-performance OCR processor module for the PDF Splitter application.

This module provides state-of-the-art OCR capabilities optimized for CPU-only systems:
- Multi-engine support with PaddleOCR as primary engine
- Intelligent preprocessing and quality assessment
- Parallel processing with optimal resource utilization
- Advanced caching and memory management
- Seamless integration with existing PDF processing pipeline
"""

import gc
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field

from pdf_splitter.core.config import PDFConfig
from pdf_splitter.preprocessing.advanced_cache import PDFProcessingCache
from pdf_splitter.preprocessing.pdf_handler import PageType

# Critical optimization for parallel processing
os.environ["OMP_THREAD_LIMIT"] = "1"

logger = logging.getLogger(__name__)


class OCREngine(str, Enum):
    """Available OCR engines."""

    PADDLEOCR = "paddleocr"
    EASYOCR = "easyocr"
    TESSERACT = "tesseract"


class OCRConfig(BaseModel):
    """Configuration for OCR processing."""

    # Engine settings
    primary_engine: OCREngine = OCREngine.PADDLEOCR
    fallback_engines: List[OCREngine] = Field(default_factory=list)

    # PaddleOCR specific settings
    paddle_use_angle_cls: bool = True
    paddle_lang: str = "en"
    paddle_det_model_dir: Optional[str] = None
    paddle_rec_model_dir: Optional[str] = None
    paddle_cls_model_dir: Optional[str] = None
    paddle_use_gpu: bool = False
    paddle_cpu_threads: int = 4
    paddle_enable_mkldnn: bool = False
    paddle_rec_batch_num: int = 6

    # Processing settings
    max_workers: Optional[int] = None  # None = CPU count
    batch_size: int = 5
    preprocessing_enabled: bool = True
    min_confidence_threshold: float = 0.5

    # Quality thresholds
    low_quality_threshold: float = 0.3
    high_quality_threshold: float = 0.8

    # Preprocessing parameters
    denoise_enabled: bool = True
    deskew_enabled: bool = True
    deskew_angle_threshold: float = 0.5
    adaptive_threshold_block_size: int = 11
    adaptive_threshold_c: int = 2

    # Performance settings
    cache_enabled: bool = True
    warmup_enabled: bool = True
    memory_limit_mb: int = 2000
    timeout_per_page: float = 30.0


class BoundingBox(BaseModel):
    """Bounding box coordinates for text region."""

    x1: float
    y1: float
    x2: float
    y2: float


class TextLine(BaseModel):
    """Single line of OCR text with metadata."""

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    angle: float = 0.0


class OCRResult(BaseModel):
    """Complete OCR result for a page."""

    page_num: int
    text_lines: List[TextLine]
    full_text: str
    avg_confidence: float = Field(ge=0.0, le=1.0)
    processing_time: float
    engine_used: OCREngine
    preprocessing_applied: List[str] = Field(default_factory=list)
    word_count: int
    char_count: int
    quality_score: float = Field(ge=0.0, le=1.0)
    warnings: List[str] = Field(default_factory=list)


class OCRQualityMetrics(BaseModel):
    """Metrics for assessing OCR quality."""

    avg_word_length: float
    special_char_ratio: float
    numeric_ratio: float
    uppercase_ratio: float
    whitespace_ratio: float
    avg_line_confidence: float
    empty_line_ratio: float
    suspicious_patterns: int


class PreprocessingResult(BaseModel):
    """Result of image preprocessing."""

    image: Any  # numpy array
    operations_applied: List[str]
    improvement_score: float
    processing_time: float


class OCRProcessor:
    """
    High-performance OCR processor optimized for CPU-only systems.

    This class provides intelligent OCR processing with multi-engine support,
    adaptive preprocessing, and advanced caching for maximum throughput.
    """

    def __init__(
        self,
        config: Optional[OCRConfig] = None,
        pdf_config: Optional[PDFConfig] = None,
        cache_manager: Optional[PDFProcessingCache] = None,
    ):
        """
        Initialize the OCR processor.

        Args:
            config: OCR-specific configuration
            pdf_config: PDF processing configuration
            cache_manager: Optional cache manager instance
        """
        self.config = config or OCRConfig()
        self.pdf_config = pdf_config or PDFConfig()
        self.cache_manager = cache_manager

        # Lazy-loaded engines
        self._engines: Dict[OCREngine, Any] = {}
        self._primary_engine_initialized = False

        # Performance tracking
        self._total_pages_processed = 0
        self._total_processing_time = 0.0
        self._engine_usage_stats: Dict[OCREngine, int] = {}

        logger.info(
            f"OCRProcessor initialized with primary engine: "
            f"{self.config.primary_engine}"
        )

    def _initialize_engine(self, engine_type: OCREngine) -> Any:
        """
        Initialize an OCR engine with optimal settings.

        Args:
            engine_type: Type of engine to initialize

        Returns:
            Initialized engine instance
        """
        logger.info(f"Initializing {engine_type} engine...")

        if engine_type == OCREngine.PADDLEOCR:
            try:
                from paddleocr import PaddleOCR

                engine = PaddleOCR(
                    use_angle_cls=self.config.paddle_use_angle_cls,
                    lang=self.config.paddle_lang,
                    use_gpu=self.config.paddle_use_gpu,
                    cpu_threads=self.config.paddle_cpu_threads,
                    enable_mkldnn=self.config.paddle_enable_mkldnn,
                    rec_batch_num=self.config.paddle_rec_batch_num,
                    det_model_dir=self.config.paddle_det_model_dir,
                    rec_model_dir=self.config.paddle_rec_model_dir,
                    cls_model_dir=self.config.paddle_cls_model_dir,
                    show_log=False,
                )
                logger.info("PaddleOCR engine initialized successfully")
                return engine

            except Exception as e:
                logger.error(f"Failed to initialize PaddleOCR: {e}")
                raise

        elif engine_type == OCREngine.EASYOCR:
            try:
                import easyocr

                engine = easyocr.Reader(
                    ["en"], gpu=self.config.paddle_use_gpu, verbose=False
                )
                logger.info("EasyOCR engine initialized successfully")
                return engine

            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {e}")
                raise

        elif engine_type == OCREngine.TESSERACT:
            try:
                import pytesseract

                # Verify Tesseract is installed
                pytesseract.get_tesseract_version()
                logger.info("Tesseract engine verified")
                return pytesseract

            except Exception as e:
                logger.error(f"Failed to initialize Tesseract: {e}")
                raise

        else:
            raise ValueError(f"Unknown OCR engine type: {engine_type}")

    def _get_engine(self, engine_type: OCREngine) -> Any:
        """
        Get or initialize an OCR engine.

        Args:
            engine_type: Type of engine to get

        Returns:
            Engine instance
        """
        if engine_type not in self._engines:
            self._engines[engine_type] = self._initialize_engine(engine_type)
        return self._engines[engine_type]

    def preprocess_image(
        self, image: np.ndarray, page_quality_hints: Optional[Dict[str, Any]] = None
    ) -> PreprocessingResult:
        """
        Apply intelligent preprocessing to improve OCR accuracy.

        Args:
            image: Input image as numpy array
            page_quality_hints: Optional hints about page quality

        Returns:
            PreprocessingResult with processed image and metrics
        """
        start_time = time.time()
        operations = []
        original_quality = self._assess_image_quality(image)

        # Work on a copy
        processed = image.copy()

        # Convert to grayscale if needed
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            operations.append("grayscale_conversion")

        # Denoise if enabled and needed
        if self.config.denoise_enabled and original_quality < 0.7:
            processed = cv2.medianBlur(processed, 3)
            operations.append("noise_reduction")

        # Deskew if enabled
        if self.config.deskew_enabled:
            angle = self._detect_skew_angle(processed)
            if abs(angle) > self.config.deskew_angle_threshold:
                processed = self._deskew_image(processed, angle)
                operations.append(f"deskew_{angle:.1f}deg")

        # Adaptive thresholding for binarization
        if original_quality < 0.6:
            processed = cv2.adaptiveThreshold(
                processed,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                self.config.adaptive_threshold_block_size,
                self.config.adaptive_threshold_c,
            )
            operations.append("adaptive_threshold")

        # Calculate improvement score
        final_quality = self._assess_image_quality(processed)
        improvement = final_quality - original_quality

        return PreprocessingResult(
            image=processed,
            operations_applied=operations,
            improvement_score=improvement,
            processing_time=time.time() - start_time,
        )

    def _assess_image_quality(self, image: np.ndarray) -> float:
        """
        Assess image quality for OCR suitability.

        Args:
            image: Input image

        Returns:
            Quality score between 0 and 1
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Calculate various quality metrics
        # 1. Contrast (standard deviation)
        contrast = np.std(gray) / 255.0

        # 2. Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian) / 10000.0
        sharpness = min(sharpness, 1.0)

        # 3. Noise level (high-frequency content)
        # Use difference between original and blurred
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = np.mean(np.abs(gray.astype(float) - blurred.astype(float))) / 255.0
        noise_score = 1.0 - min(noise * 10, 1.0)

        # Combine metrics
        quality = contrast * 0.4 + sharpness * 0.4 + noise_score * 0.2

        return min(quality, 1.0)

    def _detect_skew_angle(self, image: np.ndarray) -> float:
        """
        Detect text skew angle in image.

        Args:
            image: Grayscale image

        Returns:
            Skew angle in degrees
        """
        # Apply edge detection
        edges = cv2.Canny(image, 50, 150, apertureSize=3)

        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

        if lines is None:
            return 0.0

        # Calculate angles
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            if -45 <= angle <= 45:
                angles.append(angle)

        if not angles:
            return 0.0

        # Return median angle
        return np.median(angles)

    def _deskew_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image to correct skew.

        Args:
            image: Input image
            angle: Rotation angle in degrees

        Returns:
            Deskewed image
        """
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Perform rotation
        rotated = cv2.warpAffine(
            image,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return rotated

    def process_image(
        self,
        image: np.ndarray,
        page_num: int,
        page_type: PageType,
        skip_preprocessing: bool = False,
        pdf_path: Optional[str] = None,
    ) -> OCRResult:
        """
        Process a single image with OCR.

        Args:
            image: Input image as numpy array
            page_num: Page number for reference
            page_type: Type of page content
            skip_preprocessing: Whether to skip preprocessing
            pdf_path: Optional PDF path for caching

        Returns:
            OCRResult with extracted text and metadata
        """
        # Check cache first if available
        if self.cache_manager and pdf_path and self.config.cache_enabled:
            cache_key = (pdf_path, page_num, "ocr_result")
            cached_result = self.cache_manager.analysis_cache.get(cache_key)
            if cached_result:
                logger.debug(f"OCR cache hit for page {page_num}")
                # Reconstruct OCRResult from cached dict
                return self._reconstruct_ocr_result(cached_result)

        start_time = time.time()
        warnings = []
        preprocessing_applied = []

        # Skip if page is already searchable
        if page_type == PageType.SEARCHABLE:
            warnings.append("Page is already searchable, OCR may not be needed")

        # Preprocess if enabled and not skipped
        if self.config.preprocessing_enabled and not skip_preprocessing:
            # Check if preprocessing is really needed for this image
            image_quality = self._assess_image_quality(image)

            # Skip preprocessing for high-quality computer-generated images
            if image_quality > 0.8 and page_type != PageType.MIXED:
                processed_image = image
                warnings.append(
                    f"Skipped preprocessing for high-quality image "
                    f"(quality={image_quality:.2f})"
                )
            else:
                preprocess_result = self.preprocess_image(image)
                processed_image = preprocess_result.image
                preprocessing_applied = preprocess_result.operations_applied
        else:
            processed_image = image

        # Perform OCR with primary engine
        try:
            result = self._perform_ocr(
                processed_image, self.config.primary_engine, page_num
            )

            # Check quality and potentially retry with fallback
            if (
                result.avg_confidence < self.config.low_quality_threshold
                and self.config.fallback_engines
            ):
                warnings.append(
                    f"Low confidence ({result.avg_confidence:.2f}) from primary engine"
                )

                for fallback_engine in self.config.fallback_engines:
                    try:
                        fallback_result = self._perform_ocr(
                            processed_image, fallback_engine, page_num
                        )

                        if fallback_result.avg_confidence > result.avg_confidence:
                            result = fallback_result
                            warnings.append(f"Used fallback engine: {fallback_engine}")
                            break

                    except Exception as e:
                        logger.warning(f"Fallback engine {fallback_engine} failed: {e}")

        except Exception as e:
            logger.error(f"OCR failed for page {page_num}: {e}")
            raise

        # Update result with additional metadata
        result.preprocessing_applied = preprocessing_applied
        result.warnings = warnings
        result.processing_time = time.time() - start_time

        # Cache the result if cache is available
        if self.cache_manager and pdf_path and self.config.cache_enabled:
            cache_key = (pdf_path, page_num, "ocr_result")
            self.cache_manager.analysis_cache.put(
                cache_key, self._serialize_ocr_result(result)
            )
            logger.debug(f"Cached OCR result for page {page_num}")

        # Update statistics
        self._total_pages_processed += 1
        self._total_processing_time += result.processing_time
        self._engine_usage_stats[result.engine_used] = (
            self._engine_usage_stats.get(result.engine_used, 0) + 1
        )

        return result

    def _perform_ocr(
        self, image: np.ndarray, engine_type: OCREngine, page_num: int
    ) -> OCRResult:
        """
        Perform OCR using specified engine.

        Args:
            image: Preprocessed image
            engine_type: OCR engine to use
            page_num: Page number for reference

        Returns:
            OCRResult with extracted text
        """
        engine = self._get_engine(engine_type)
        text_lines = []

        if engine_type == OCREngine.PADDLEOCR:
            result = engine.ocr(image, cls=True)

            if result and result[0]:
                for line in result[0]:
                    bbox_points = line[0]
                    text = line[1][0]
                    confidence = line[1][1]

                    # Convert bbox points to BoundingBox
                    x_coords = [p[0] for p in bbox_points]
                    y_coords = [p[1] for p in bbox_points]

                    bbox = BoundingBox(
                        x1=min(x_coords),
                        y1=min(y_coords),
                        x2=max(x_coords),
                        y2=max(y_coords),
                    )

                    text_lines.append(
                        TextLine(text=text, confidence=confidence, bbox=bbox)
                    )

        elif engine_type == OCREngine.EASYOCR:
            result = engine.readtext(image)

            for bbox_points, text, confidence in result:
                # Convert bbox format
                if len(bbox_points) == 4:
                    x_coords = [p[0] for p in bbox_points]
                    y_coords = [p[1] for p in bbox_points]
                else:
                    x_coords = [bbox_points[0], bbox_points[2]]
                    y_coords = [bbox_points[1], bbox_points[3]]

                bbox = BoundingBox(
                    x1=min(x_coords),
                    y1=min(y_coords),
                    x2=max(x_coords),
                    y2=max(y_coords),
                )

                text_lines.append(TextLine(text=text, confidence=confidence, bbox=bbox))

        elif engine_type == OCREngine.TESSERACT:
            import pytesseract

            # Get detailed OCR data
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

            n_boxes = len(data["text"])
            for i in range(n_boxes):
                if int(data["conf"][i]) > 0:  # Filter out empty results
                    text = data["text"][i].strip()
                    if text:  # Only include non-empty text
                        bbox = BoundingBox(
                            x1=data["left"][i],
                            y1=data["top"][i],
                            x2=data["left"][i] + data["width"][i],
                            y2=data["top"][i] + data["height"][i],
                        )

                        confidence = float(data["conf"][i]) / 100.0
                        text_lines.append(
                            TextLine(text=text, confidence=confidence, bbox=bbox)
                        )

        # Sort text lines by vertical position for proper reading order
        text_lines.sort(key=lambda x: (x.bbox.y1, x.bbox.x1))

        # Combine into full text
        full_text = "\n".join([line.text for line in text_lines])

        # Calculate metrics
        avg_confidence = (
            sum(line.confidence for line in text_lines) / len(text_lines)
            if text_lines
            else 0.0
        )

        word_count = len(full_text.split())
        char_count = len(full_text)

        # Calculate quality score
        quality_metrics = self._calculate_quality_metrics(text_lines, full_text)
        quality_score = self._calculate_quality_score(quality_metrics)

        return OCRResult(
            page_num=page_num,
            text_lines=text_lines,
            full_text=full_text,
            avg_confidence=avg_confidence,
            processing_time=0.0,  # Will be set by caller
            engine_used=engine_type,
            word_count=word_count,
            char_count=char_count,
            quality_score=quality_score,
        )

    def _calculate_quality_metrics(
        self, text_lines: List[TextLine], full_text: str
    ) -> OCRQualityMetrics:
        """
        Calculate detailed quality metrics for OCR result.

        Args:
            text_lines: List of text lines with confidence
            full_text: Combined full text

        Returns:
            OCRQualityMetrics instance
        """
        if not full_text:
            return OCRQualityMetrics(
                avg_word_length=0.0,
                special_char_ratio=0.0,
                numeric_ratio=0.0,
                uppercase_ratio=0.0,
                whitespace_ratio=0.0,
                avg_line_confidence=0.0,
                empty_line_ratio=1.0,
                suspicious_patterns=0,
            )

        # Word metrics
        words = full_text.split()
        avg_word_length = (
            sum(len(word) for word in words) / len(words) if words else 0.0
        )

        # Character ratios
        total_chars = len(full_text)
        special_chars = sum(1 for c in full_text if not c.isalnum() and not c.isspace())
        numeric_chars = sum(1 for c in full_text if c.isdigit())
        uppercase_chars = sum(1 for c in full_text if c.isupper())
        whitespace_chars = sum(1 for c in full_text if c.isspace())

        # Line metrics
        avg_line_confidence = (
            sum(line.confidence for line in text_lines) / len(text_lines)
            if text_lines
            else 0.0
        )
        empty_lines = sum(1 for line in text_lines if not line.text.strip())
        empty_line_ratio = empty_lines / len(text_lines) if text_lines else 0.0

        # Suspicious patterns (common OCR errors)
        suspicious_patterns = 0
        suspicious_indicators = [
            "|||",  # Multiple pipes often indicate table parsing issues
            "```",  # Multiple backticks
            "...",  # Excessive dots (more than 3)
            "___",  # Multiple underscores
            "!!!",  # Multiple exclamation marks
        ]

        for pattern in suspicious_indicators:
            suspicious_patterns += full_text.count(pattern)

        return OCRQualityMetrics(
            avg_word_length=avg_word_length,
            special_char_ratio=special_chars / total_chars if total_chars > 0 else 0.0,
            numeric_ratio=numeric_chars / total_chars if total_chars > 0 else 0.0,
            uppercase_ratio=uppercase_chars / total_chars if total_chars > 0 else 0.0,
            whitespace_ratio=whitespace_chars / total_chars if total_chars > 0 else 0.0,
            avg_line_confidence=avg_line_confidence,
            empty_line_ratio=empty_line_ratio,
            suspicious_patterns=suspicious_patterns,
        )

    def _serialize_ocr_result(self, result: OCRResult) -> Dict[str, Any]:
        """
        Serialize OCRResult for caching.

        Args:
            result: OCR result to serialize

        Returns:
            Dictionary representation
        """
        return {
            "page_num": result.page_num,
            "text_lines": [
                {
                    "text": line.text,
                    "confidence": line.confidence,
                    "bbox": {
                        "x1": line.bbox.x1,
                        "y1": line.bbox.y1,
                        "x2": line.bbox.x2,
                        "y2": line.bbox.y2,
                    },
                    "angle": line.angle,
                }
                for line in result.text_lines
            ],
            "full_text": result.full_text,
            "avg_confidence": result.avg_confidence,
            "processing_time": result.processing_time,
            "engine_used": result.engine_used.value,
            "preprocessing_applied": result.preprocessing_applied,
            "word_count": result.word_count,
            "char_count": result.char_count,
            "quality_score": result.quality_score,
            "warnings": result.warnings,
        }

    def _reconstruct_ocr_result(self, data: Dict[str, Any]) -> OCRResult:
        """
        Reconstruct OCRResult from cached data.

        Args:
            data: Cached dictionary data

        Returns:
            Reconstructed OCRResult
        """
        text_lines = [
            TextLine(
                text=line["text"],
                confidence=line["confidence"],
                bbox=BoundingBox(**line["bbox"]),
                angle=line.get("angle", 0.0),
            )
            for line in data["text_lines"]
        ]

        return OCRResult(
            page_num=data["page_num"],
            text_lines=text_lines,
            full_text=data["full_text"],
            avg_confidence=data["avg_confidence"],
            processing_time=data["processing_time"],
            engine_used=OCREngine(data["engine_used"]),
            preprocessing_applied=data.get("preprocessing_applied", []),
            word_count=data["word_count"],
            char_count=data["char_count"],
            quality_score=data["quality_score"],
            warnings=data.get("warnings", []),
        )

    def _calculate_quality_score(self, metrics: OCRQualityMetrics) -> float:
        """
        Calculate overall quality score from metrics.

        Args:
            metrics: Quality metrics

        Returns:
            Quality score between 0 and 1
        """
        score = 1.0

        # Penalize very short average word length
        if metrics.avg_word_length < 2:
            score *= 0.7
        elif metrics.avg_word_length > 15:
            score *= 0.8

        # Penalize high special character ratio
        if metrics.special_char_ratio > 0.3:
            score *= 0.8
        elif metrics.special_char_ratio > 0.5:
            score *= 0.6

        # Penalize very high uppercase ratio (might indicate poor recognition)
        if metrics.uppercase_ratio > 0.8:
            score *= 0.9

        # Confidence-based adjustment
        score *= metrics.avg_line_confidence

        # Penalize empty lines
        if metrics.empty_line_ratio > 0.3:
            score *= 0.9

        # Penalize suspicious patterns
        if metrics.suspicious_patterns > 5:
            score *= 0.8
        elif metrics.suspicious_patterns > 10:
            score *= 0.6

        return max(0.0, min(1.0, score))

    def process_batch(
        self,
        images: List[Tuple[np.ndarray, int, PageType]],
        max_workers: Optional[int] = None,
        pdf_path: Optional[str] = None,
    ) -> List[OCRResult]:
        """
        Process multiple images in parallel.

        Args:
            images: List of (image, page_num, page_type) tuples
            max_workers: Number of parallel workers
            pdf_path: Optional PDF path for caching

        Returns:
            List of OCR results
        """
        if not images:
            return []

        max_workers = max_workers or self.config.max_workers

        # For single image, process directly
        if len(images) == 1:
            image, page_num, page_type = images[0]
            return [self.process_image(image, page_num, page_type, pdf_path=pdf_path)]

        # Use multiprocessing for batch
        results = []

        # Create worker pool with proper initialization
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(self.config,),
        ) as executor:
            # Submit all tasks
            futures = []
            for image, page_num, page_type in images:
                future = executor.submit(
                    _process_image_worker, image, page_num, page_type
                )
                futures.append(future)

            # Collect results in order
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=self.config.timeout_per_page)

                    # Cache result if cache manager available
                    if self.cache_manager and pdf_path and self.config.cache_enabled:
                        cache_key = (pdf_path, result.page_num, "ocr_result")
                        self.cache_manager.analysis_cache.put(
                            cache_key, self._serialize_ocr_result(result)
                        )

                    results.append(result)
                except Exception as e:
                    logger.error(f"OCR processing failed: {e}")
                    # Create error result
                    page_num = images[i][1] if i < len(images) else -1
                    results.append(
                        OCRResult(
                            page_num=page_num,
                            text_lines=[],
                            full_text="",
                            avg_confidence=0.0,
                            processing_time=0.0,
                            engine_used=self.config.primary_engine,
                            word_count=0,
                            char_count=0,
                            quality_score=0.0,
                            warnings=[f"Processing failed: {str(e)}"],
                        )
                    )

        return results

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        avg_time_per_page = (
            self._total_processing_time / self._total_pages_processed
            if self._total_pages_processed > 0
            else 0.0
        )

        return {
            "total_pages_processed": self._total_pages_processed,
            "total_processing_time": round(self._total_processing_time, 2),
            "avg_time_per_page": round(avg_time_per_page, 2),
            "engine_usage": self._engine_usage_stats,
            "primary_engine": self.config.primary_engine.value,
            "preprocessing_enabled": self.config.preprocessing_enabled,
            "parallel_workers": self.config.max_workers,
        }

    def cleanup(self):
        """Clean up resources and temporary files."""
        # Clear engine instances
        self._engines.clear()

        # Force garbage collection
        gc.collect()

        logger.info("OCR processor cleanup complete")


# Global variable for multiprocessing workers
_worker_processor: Optional[OCRProcessor] = None


def _init_worker(config: OCRConfig):
    """
    Initialize worker process with OCR processor.

    Args:
        config: OCR configuration
    """
    global _worker_processor
    _worker_processor = OCRProcessor(config)
    logger.info(f"Worker process initialized with PID: {os.getpid()}")


def _process_image_worker(
    image: np.ndarray, page_num: int, page_type: PageType
) -> OCRResult:
    """
    Worker function for processing single image.

    Args:
        image: Input image
        page_num: Page number
        page_type: Type of page

    Returns:
        OCR result
    """
    if _worker_processor is None:
        raise RuntimeError("Worker not properly initialized")

    return _worker_processor.process_image(image, page_num, page_type)
