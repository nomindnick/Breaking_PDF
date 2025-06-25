"""Configuration for text extraction with centralized constants."""

from pydantic import BaseModel, Field


class TextExtractionConfig(BaseModel):
    """Configuration for text extraction operations."""

    # Layout detection thresholds
    table_alignment_tolerance: float = Field(
        default=5.0, description="Pixel tolerance for table column alignment detection"
    )
    table_min_avg_columns: float = Field(
        default=1.5, description="Minimum average columns per row to detect as table"
    )
    header_footer_page_ratio: float = Field(
        default=0.1,
        description="Ratio of page height for header/footer detection (0.1 = 10%)",
    )

    # Font analysis
    min_font_size_difference: float = Field(
        default=2.0,
        description="Minimum font size difference (points) to consider significant",
    )

    # Reading order detection
    reading_order_tolerance: float = Field(
        default=10.0, description="Pixel tolerance for reading order detection"
    )
    line_height_multiplier: float = Field(
        default=1.5, description="Multiplier for line height in reading order detection"
    )

    # Text quality thresholds
    min_text_length_quality: int = Field(
        default=100, description="Minimum text length for quality assessment"
    )
    word_confidence_threshold: float = Field(
        default=0.8, description="Confidence threshold for word extraction"
    )

    # Table detection
    min_table_rows: int = Field(
        default=2, description="Minimum rows to consider as table"
    )
    min_table_columns: int = Field(
        default=2, description="Minimum columns to consider as table"
    )
    table_cell_padding: float = Field(
        default=2.0, description="Padding around table cells for text extraction"
    )

    # Performance settings
    max_blocks_per_page: int = Field(
        default=1000, description="Maximum text blocks to process per page"
    )
    enable_layout_analysis: bool = Field(
        default=True, description="Enable advanced layout analysis"
    )
    enable_table_detection: bool = Field(
        default=True, description="Enable table structure detection"
    )

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        extra = "forbid"
