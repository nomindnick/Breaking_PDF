"""
Example test module demonstrating usage of shared fixtures and utilities.

This shows how to use the global conftest.py fixtures and test_utils functions.
"""

import pytest

from pdf_splitter.test_utils import (
    assert_pdf_valid,
    assert_text_quality,
    compare_images,
    create_noisy_image,
    create_test_pdf,
    measure_performance,
)


class TestExampleWithFixtures:
    """Example test class using shared fixtures."""

    def test_with_pdf_config(self, pdf_config):
        """Example using pdf_config fixture."""
        assert pdf_config.default_dpi == 150
        assert pdf_config.enable_cache_metrics is True

    def test_with_test_paths(self, test_pdf_paths):
        """Example using test_pdf_paths fixture."""
        assert "non_ocr" in test_pdf_paths
        assert "ocr" in test_pdf_paths

        # Check if files exist
        for pdf_type, path in test_pdf_paths.items():
            if path.exists():
                assert_pdf_valid(path)

    def test_with_loaded_handler(self, loaded_pdf_handler):
        """Example using loaded_pdf_handler fixture."""
        # Handler comes pre-loaded with a PDF
        assert loaded_pdf_handler.is_loaded
        assert loaded_pdf_handler.page_count > 0

        # Extract text from first page
        text = loaded_pdf_handler.extract_text(0)
        assert_text_quality(text.text, min_length=10, min_words=5)

    def test_with_mock_fixtures(self, mock_pdf_page, mock_pdf_document):
        """Example using mock fixtures."""
        # Mock page has predefined properties
        assert mock_pdf_page.rect.width == 612
        assert mock_pdf_page.get_text() == "Sample text content"

        # Mock document has 5 pages
        assert len(mock_pdf_document) == 5
        assert mock_pdf_document.metadata["title"] == "Test PDF"

    def test_with_image_fixtures(self, test_image_rgb, noisy_test_image):
        """Example using image fixtures."""
        # Compare clean and noisy images
        similarity = compare_images(test_image_rgb, noisy_test_image)
        assert 0.7 <= similarity <= 0.95  # Should be similar but not identical

        # Check image properties
        assert test_image_rgb.shape == (200, 400, 3)
        assert noisy_test_image.shape == (200, 400, 3)

    def test_with_sample_data(self, sample_text_blocks, sample_ocr_result):
        """Example using sample data fixtures."""
        # Text blocks have expected structure
        assert len(sample_text_blocks) == 3
        assert sample_text_blocks[0].is_bold  # Header is bold
        assert not sample_text_blocks[1].is_bold  # Body is not bold

        # OCR result has expected fields
        assert sample_ocr_result.avg_confidence > 0.9
        assert sample_ocr_result.word_count == 7

    def test_with_temp_dir(self, temp_dir):
        """Example using temporary directory."""
        # Create a test PDF in temp directory
        pdf_path = temp_dir / "test.pdf"
        create_test_pdf(num_pages=3, output_path=pdf_path)

        # Verify it was created
        assert pdf_path.exists()
        assert_pdf_valid(pdf_path)

        # Temp dir will be cleaned up automatically

    @pytest.mark.slow
    def test_performance_measurement(self, loaded_pdf_handler, performance_timer):
        """Example of performance testing."""

        def extract_all_pages():
            for i in range(loaded_pdf_handler.page_count):
                loaded_pdf_handler.extract_text(i)

        # Measure performance
        perf_stats = measure_performance(extract_all_pages, iterations=3)

        # Check performance meets requirements
        assert perf_stats["mean"] < 5.0  # Should average < 5 seconds

        # Also use timer context manager
        with performance_timer() as timer:
            loaded_pdf_handler.render_page(0)

        assert timer.elapsed < 1.0  # Single page render < 1 second


class TestUtilityFunctions:
    """Test the utility functions themselves."""

    def test_create_test_pdf(self, temp_dir):
        """Test PDF creation utility."""
        pdf_path = create_test_pdf(
            num_pages=10,
            include_text=True,
            include_images=True,
            output_path=temp_dir / "created.pdf",
        )

        assert pdf_path.exists()

        # Verify the created PDF
        import fitz

        doc = fitz.open(str(pdf_path))
        assert doc.page_count == 10

        # Check first page has text
        page = doc[0]
        text = page.get_text()
        assert "Test Document - Page 1" in text
        doc.close()

    def test_create_noisy_image(self):
        """Test noisy image creation."""
        clean_img = create_noisy_image(noise_level=0.0)
        noisy_img = create_noisy_image(noise_level=0.2)

        # Noisy image should be different from clean
        similarity = compare_images(clean_img, noisy_img)
        assert similarity < 0.95

        # But still somewhat similar
        assert similarity > 0.5

    def test_parametrization_helpers(self):
        """Test parametrization helper functions."""
        from conftest import (
            get_test_dpi_values,
            get_test_pdf_pages,
            get_test_quality_thresholds,
        )

        pages = get_test_pdf_pages()
        assert len(pages) > 0
        assert all(isinstance(p, int) for p in pages)

        dpis = get_test_dpi_values()
        assert len(dpis) > 0
        assert all(72 <= dpi <= 300 for dpi in dpis)

        thresholds = get_test_quality_thresholds()
        assert len(thresholds) > 0
        assert all(0 <= t <= 1 for t in thresholds)


# Example of using parametrization
@pytest.mark.parametrize("dpi", [72, 150, 300])
def test_with_different_dpis(pdf_handler, test_pdf_paths, dpi):
    """Example of parametrized test."""
    pdf_path = test_pdf_paths["non_ocr"]
    if not pdf_path.exists():
        pytest.skip("Test PDF not found")

    with pdf_handler.load_pdf(pdf_path):
        # Render at different DPIs
        img = pdf_handler.render_page(0, dpi=dpi)

        # Higher DPI should produce larger images
        expected_width = int(612 * dpi / 72)
        expected_height = int(792 * dpi / 72)

        assert abs(img.shape[1] - expected_width) < 10
        assert abs(img.shape[0] - expected_height) < 10


if __name__ == "__main__":
    # Run the example tests
    pytest.main([__file__, "-v"])
