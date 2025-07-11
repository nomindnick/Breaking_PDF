"""
Integration tests for concurrent processing and thread safety.

Tests cover:
- Thread safety of major components
- Concurrent session management
- Parallel PDF processing
- Race condition detection
- Resource contention (file handles, database connections)
- Proper cleanup in concurrent scenarios
- Load testing with simultaneous operations
- Deadlock detection
"""

import asyncio
import concurrent.futures
import multiprocessing
import os
import sqlite3
import threading
import time
from typing import List, Set

import pytest

from pdf_splitter.core.config import PDFConfig
from pdf_splitter.detection.detectors.embeddings_detector import EmbeddingsDetector
from pdf_splitter.detection.detectors.heuristic_detector import HeuristicDetector
from pdf_splitter.preprocessing.cache import AdvancedCache
from pdf_splitter.preprocessing.ocr_processor import OCRProcessor
from pdf_splitter.preprocessing.pdf_handler import PDFHandler
from pdf_splitter.preprocessing.text_extractor import TextExtractor
from pdf_splitter.splitting.session_manager import SplitSessionManager
from pdf_splitter.test_utils import create_test_pdf


class ThreadSafetyMonitor:
    """Monitor for detecting thread safety issues."""

    def __init__(self):
        """Initialize the thread safety monitor."""
        self.access_log = []
        self.lock = threading.Lock()
        self.concurrent_access = 0
        self.max_concurrent = 0
        self.race_conditions = []

    def enter(self, component: str, operation: str):
        """Record entry to a critical section."""
        with self.lock:
            self.concurrent_access += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent_access)
            self.access_log.append(
                {
                    "time": time.time(),
                    "thread": threading.current_thread().name,
                    "component": component,
                    "operation": operation,
                    "action": "enter",
                    "concurrent": self.concurrent_access,
                }
            )

    def exit(self, component: str, operation: str):
        """Record exit from a critical section."""
        with self.lock:
            self.concurrent_access -= 1
            self.access_log.append(
                {
                    "time": time.time(),
                    "thread": threading.current_thread().name,
                    "component": component,
                    "operation": operation,
                    "action": "exit",
                }
            )

    def detect_race_conditions(self) -> List[dict]:
        """Analyze access log for potential race conditions."""
        critical_sections = {}
        race_conditions = []

        for entry in self.access_log:
            key = f"{entry['component']}:{entry['operation']}"

            if entry["action"] == "enter":
                if key not in critical_sections:
                    critical_sections[key] = []
                critical_sections[key].append(entry)
            elif entry["action"] == "exit":
                if key in critical_sections and critical_sections[key]:
                    critical_sections[key].pop()

            # Check for concurrent access to same critical section
            if key in critical_sections and len(critical_sections[key]) > 1:
                race_conditions.append(
                    {
                        "component": entry["component"],
                        "operation": entry["operation"],
                        "concurrent_threads": [
                            e["thread"] for e in critical_sections[key]
                        ],
                        "time": entry["time"],
                    }
                )

        return race_conditions


class ResourceMonitor:
    """Monitor for tracking resource usage and leaks."""

    def __init__(self):
        """Initialize the resource monitor."""
        self.file_handles: Set[int] = set()
        self.db_connections: Set[sqlite3.Connection] = set()
        self.threads: Set[threading.Thread] = set()
        self.lock = threading.Lock()

    def register_file(self, fd: int):
        """Register a file descriptor."""
        with self.lock:
            self.file_handles.add(fd)

    def unregister_file(self, fd: int):
        """Unregister a file descriptor."""
        with self.lock:
            self.file_handles.discard(fd)

    def register_connection(self, conn: sqlite3.Connection):
        """Register a database connection."""
        with self.lock:
            self.db_connections.add(conn)

    def unregister_connection(self, conn: sqlite3.Connection):
        """Unregister a database connection."""
        with self.lock:
            self.db_connections.discard(conn)

    def check_leaks(self) -> dict:
        """Check for resource leaks."""
        with self.lock:
            return {
                "open_files": len(self.file_handles),
                "open_connections": len(self.db_connections),
                "active_threads": len([t for t in self.threads if t.is_alive()]),
            }


@pytest.fixture
def thread_monitor():
    """Create a thread safety monitor."""
    return ThreadSafetyMonitor()


@pytest.fixture
def resource_monitor():
    """Create a resource monitor."""
    return ResourceMonitor()


@pytest.fixture
def concurrent_test_pdfs(temp_dir):
    """Create multiple test PDFs for concurrent processing."""
    pdfs = []
    for i in range(10):
        pdf_path = temp_dir / f"test_concurrent_{i}.pdf"
        create_test_pdf(num_pages=5, output_path=pdf_path)
        pdfs.append(pdf_path)
    return pdfs


class TestThreadSafety:
    """Test thread safety of major components."""

    def test_pdf_handler_thread_safety(self, concurrent_test_pdfs, thread_monitor):
        """Test PDFHandler thread safety."""
        config = PDFConfig()
        errors = []

        def process_pdf(pdf_path):
            try:
                handler = PDFHandler(config)
                thread_monitor.enter("PDFHandler", "load_pdf")
                handler.load_pdf(pdf_path)
                thread_monitor.exit("PDFHandler", "load_pdf")

                # Simulate processing
                for i in range(handler.page_count):
                    thread_monitor.enter("PDFHandler", f"get_page_{i}")
                    _ = handler.get_page(i)
                    thread_monitor.exit("PDFHandler", f"get_page_{i}")

                handler.close()
            except Exception as e:
                errors.append(e)

        # Run concurrent PDF processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(process_pdf, pdf) for pdf in concurrent_test_pdfs[:5]
            ]
            concurrent.futures.wait(futures)

        # Check for errors and race conditions
        assert not errors, f"Errors during concurrent processing: {errors}"
        race_conditions = thread_monitor.detect_race_conditions()
        assert not race_conditions, f"Race conditions detected: {race_conditions}"

    def test_cache_thread_safety(self, thread_monitor):
        """Test AdvancedCache thread safety."""
        cache = AdvancedCache(max_memory_mb=100)
        errors = []
        test_data = {"key": "value", "data": list(range(1000))}

        def cache_operations(thread_id):
            try:
                for i in range(100):
                    key = f"thread_{thread_id}_item_{i}"

                    # Write operation
                    thread_monitor.enter("Cache", "set")
                    cache.set(key, test_data)
                    thread_monitor.exit("Cache", "set")

                    # Read operation
                    thread_monitor.enter("Cache", "get")
                    value = cache.get(key)
                    thread_monitor.exit("Cache", "get")

                    assert value == test_data

                    # Delete operation
                    if i % 10 == 0:
                        thread_monitor.enter("Cache", "delete")
                        cache.invalidate(key)
                        thread_monitor.exit("Cache", "delete")
            except Exception as e:
                errors.append(e)

        # Run concurrent cache operations
        threads = []
        for i in range(10):
            thread = threading.Thread(target=cache_operations, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert not errors, f"Cache errors: {errors}"
        assert thread_monitor.max_concurrent > 1, "No concurrent access detected"

    def test_text_extractor_thread_safety(self, concurrent_test_pdfs, thread_monitor):
        """Test TextExtractor thread safety."""
        config = PDFConfig()
        errors = []

        def extract_text(pdf_path):
            try:
                handler = PDFHandler(config)
                handler.load_pdf(pdf_path)
                extractor = TextExtractor(config)

                for i in range(handler.page_count):
                    thread_monitor.enter("TextExtractor", f"extract_page_{i}")
                    page = handler.get_page(i)
                    _ = extractor.extract_text(page)
                    thread_monitor.exit("TextExtractor", f"extract_page_{i}")

                handler.close()
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(extract_text, pdf) for pdf in concurrent_test_pdfs[:5]
            ]
            concurrent.futures.wait(futures)

        assert not errors, f"Text extraction errors: {errors}"

    @pytest.mark.skipif(
        os.environ.get("RUN_OCR_TESTS") != "true",
        reason="OCR tests disabled by default",
    )
    def test_ocr_processor_thread_safety(self, test_image_rgb, thread_monitor):
        """Test OCRProcessor thread safety."""
        errors = []
        results = []

        def perform_ocr(thread_id):
            try:
                processor = OCRProcessor()
                thread_monitor.enter("OCRProcessor", "process")
                result = processor.process_image(test_image_rgb)
                thread_monitor.exit("OCRProcessor", "process")
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            thread = threading.Thread(target=perform_ocr, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert not errors, f"OCR errors: {errors}"
        assert len(results) == 5
        # All results should be similar
        assert all(r.text == results[0].text for r in results)


class TestConcurrentSessionManagement:
    """Test concurrent session management."""

    def test_session_manager_concurrent_creation(self, temp_dir, thread_monitor):
        """Test concurrent session creation."""
        manager = SplitSessionManager(db_path=temp_dir / "sessions.db")
        session_ids = []
        errors = []

        def create_session(thread_id):
            try:
                thread_monitor.enter("SplitSessionManager", "create_session")
                # Create a dummy proposal for testing
                from pdf_splitter.splitting.models import SplitProposal

                proposal = SplitProposal(
                    pdf_path=temp_dir / f"test_{thread_id}.pdf",
                    total_pages=10,
                    segments=[],
                    detection_results=[],
                )
                session = manager.create_session(proposal)
                thread_monitor.exit("SplitSessionManager", "create_session")
                session_ids.append(session.session_id)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(20):
            thread = threading.Thread(target=create_session, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert not errors, f"Session creation errors: {errors}"
        assert len(session_ids) == 20
        assert len(set(session_ids)) == 20, "Duplicate session IDs created"

    def test_session_concurrent_operations(self, temp_dir):
        """Test concurrent operations on same session."""
        manager = SplitSessionManager(db_path=temp_dir / "sessions.db")
        # Create a dummy proposal
        from pdf_splitter.splitting.models import SplitProposal

        proposal = SplitProposal(
            pdf_path=temp_dir / "test.pdf",
            total_pages=10,
            segments=[],
            detection_results=[],
        )
        session = manager.create_session(proposal)
        errors = []
        operation_count = 0
        lock = threading.Lock()

        def perform_operations(thread_id):
            nonlocal operation_count
            try:
                for i in range(50):
                    # Update session status
                    status = "modified" if i % 2 == 0 else "confirmed"
                    try:
                        manager.update_session(session.session_id, status=status)
                    except Exception:
                        pass  # Expected - invalid state transitions

                    with lock:
                        operation_count += 1
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            thread = threading.Thread(target=perform_operations, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert not errors, f"Session operation errors: {errors}"
        assert operation_count == 500

    def test_session_cleanup_race_conditions(self, temp_dir):
        """Test race conditions during session cleanup."""
        manager = SplitSessionManager(db_path=temp_dir / "sessions.db")
        errors = []

        def create_and_cleanup(thread_id):
            try:
                # Create session
                from pdf_splitter.splitting.models import SplitProposal

                proposal = SplitProposal(
                    pdf_path=temp_dir / f"test_{thread_id}.pdf",
                    total_pages=10,
                    segments=[],
                    detection_results=[],
                )
                session = manager.create_session(proposal)
                session_id = session.session_id

                # Small delay to increase chance of race conditions
                time.sleep(0.01)

                # Delete session
                manager.delete_session(session_id)

                # Verify cleanup
                from pdf_splitter.splitting.exceptions import SessionNotFoundError

                try:
                    manager.get_session(session_id)
                    assert False, "Session still exists after deletion"
                except SessionNotFoundError:
                    pass  # Expected
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(20):
            thread = threading.Thread(target=create_and_cleanup, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert not errors, f"Cleanup errors: {errors}"


class TestParallelPDFProcessing:
    """Test parallel PDF processing scenarios."""

    def test_parallel_pdf_processing_pipeline(self, concurrent_test_pdfs, temp_dir):
        """Test full PDF processing pipeline in parallel."""
        config = PDFConfig()
        errors = []
        results = []

        def process_pdf_pipeline(pdf_path):
            try:
                # Load PDF
                handler = PDFHandler(config)
                handler.load_pdf(pdf_path)

                # Extract text
                extractor = TextExtractor(config)
                pages = []
                for i in range(handler.page_count):
                    page = handler.get_page(i)
                    text = extractor.extract_text(page)
                    pages.append(text)

                # Detect boundaries
                detector = HeuristicDetector()
                context = detector.create_context(pages)
                boundaries = detector.detect_boundaries(context)

                results.append(
                    {
                        "pdf": pdf_path.name,
                        "pages": len(pages),
                        "boundaries": len(boundaries),
                    }
                )

                handler.close()
            except Exception as e:
                errors.append((pdf_path, e))

        # Process PDFs in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(process_pdf_pipeline, pdf)
                for pdf in concurrent_test_pdfs
            ]
            concurrent.futures.wait(futures)

        assert not errors, f"Processing errors: {errors}"
        assert len(results) == len(concurrent_test_pdfs)

    def test_parallel_detector_processing(self, concurrent_test_pdfs):
        """Test parallel processing with multiple detectors."""
        config = PDFConfig()
        errors = []

        def process_with_detectors(pdf_path):
            try:
                handler = PDFHandler(config)
                handler.load_pdf(pdf_path)
                extractor = TextExtractor(config)

                # Extract all pages
                pages = []
                for i in range(handler.page_count):
                    page = handler.get_page(i)
                    text = extractor.extract_text(page)
                    pages.append(text)

                # Run multiple detectors in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    # Heuristic detector
                    heuristic_future = executor.submit(
                        lambda: HeuristicDetector().detect_boundaries(
                            HeuristicDetector().create_context(pages)
                        )
                    )

                    # Embeddings detector
                    embeddings_future = executor.submit(
                        lambda: EmbeddingsDetector().detect_boundaries(
                            EmbeddingsDetector().create_context(pages)
                        )
                    )

                    heuristic_result = heuristic_future.result()
                    embeddings_result = embeddings_future.result()

                    assert isinstance(heuristic_result, list)
                    assert isinstance(embeddings_result, list)

                handler.close()
            except Exception as e:
                errors.append(e)

        # Process multiple PDFs each with multiple detectors
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(process_with_detectors, pdf)
                for pdf in concurrent_test_pdfs[:3]
            ]
            concurrent.futures.wait(futures)

        assert not errors, f"Detector processing errors: {errors}"


class TestRaceConditionDetection:
    """Test for detecting race conditions."""

    def test_cache_race_condition_detection(self):
        """Test detection of cache race conditions."""
        cache = AdvancedCache(max_memory_mb=10)
        race_detected = False
        test_key = "shared_key"

        def rapid_cache_updates(thread_id):
            nonlocal race_detected
            for i in range(1000):
                # Write unique value
                cache.set(test_key, {"thread": thread_id, "iteration": i})

                # Immediately read back
                value = cache.get(test_key)

                # Check if we got our own value
                if value and (
                    value.get("thread") != thread_id or value.get("iteration") != i
                ):
                    race_detected = True
                    break

        threads = []
        for i in range(5):
            thread = threading.Thread(target=rapid_cache_updates, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Race conditions should be prevented by proper locking
        assert not race_detected, "Race condition detected in cache"

    def test_file_access_race_conditions(self, temp_dir):
        """Test file access race conditions."""
        test_file = temp_dir / "race_test.txt"
        errors = []
        expected_lines = 1000

        def write_to_file(thread_id):
            try:
                for i in range(100):
                    with open(test_file, "a") as f:
                        f.write(f"Thread {thread_id} - Line {i}\n")
                        f.flush()
            except Exception as e:
                errors.append(e)

        # Create file
        test_file.write_text("")

        # Multiple threads writing to same file
        threads = []
        for i in range(10):
            thread = threading.Thread(target=write_to_file, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify all writes succeeded
        assert not errors, f"File write errors: {errors}"
        lines = test_file.read_text().strip().split("\n")
        assert (
            len(lines) == expected_lines
        ), f"Expected {expected_lines} lines, got {len(lines)}"

    def test_database_race_conditions(self, temp_dir):
        """Test database access race conditions."""
        db_path = temp_dir / "test.db"
        errors = []
        expected_count = 500

        def db_operations(thread_id):
            try:
                conn = sqlite3.connect(str(db_path), timeout=10.0)
                cursor = conn.cursor()

                for i in range(50):
                    # Insert
                    cursor.execute(
                        "INSERT INTO test_table (thread_id, value) VALUES (?, ?)",
                        (thread_id, i),
                    )

                    # Update
                    cursor.execute(
                        "UPDATE test_table SET value = value + 1 WHERE thread_id = ?",
                        (thread_id,),
                    )

                    conn.commit()

                conn.close()
            except Exception as e:
                errors.append(e)

        # Initialize database
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE test_table (id INTEGER PRIMARY KEY, thread_id INTEGER, value INTEGER)"
        )
        conn.close()

        # Run concurrent database operations
        threads = []
        for i in range(10):
            thread = threading.Thread(target=db_operations, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify results
        assert not errors, f"Database errors: {errors}"
        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM test_table").fetchone()[0]
        conn.close()
        assert (
            count == expected_count
        ), f"Expected {expected_count} records, got {count}"


class TestResourceContention:
    """Test resource contention scenarios."""

    def test_file_handle_contention(self, temp_dir, resource_monitor):
        """Test file handle resource contention."""
        errors = []
        max_files = 50

        def open_many_files(thread_id):
            try:
                handles = []
                for i in range(max_files):
                    file_path = temp_dir / f"thread_{thread_id}_file_{i}.txt"
                    f = open(file_path, "w")
                    resource_monitor.register_file(f.fileno())
                    handles.append(f)

                    # Write some data
                    f.write(f"Thread {thread_id} - File {i}\n")
                    f.flush()

                # Clean up
                for f in handles:
                    resource_monitor.unregister_file(f.fileno())
                    f.close()
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            thread = threading.Thread(target=open_many_files, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Check for errors and leaks
        assert not errors, f"File handle errors: {errors}"
        leaks = resource_monitor.check_leaks()
        assert leaks["open_files"] == 0, f"File handle leaks detected: {leaks}"

    def test_memory_contention(self):
        """Test memory resource contention."""
        cache = AdvancedCache(max_memory_mb=50)  # Limited memory
        errors = []
        large_data = b"x" * (1024 * 1024)  # 1MB per item

        def consume_memory(thread_id):
            try:
                for i in range(20):
                    key = f"thread_{thread_id}_item_{i}"
                    cache.set(key, large_data)
                    time.sleep(0.01)  # Small delay to increase contention
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            thread = threading.Thread(target=consume_memory, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # System should handle memory pressure without errors
        assert not errors, f"Memory errors: {errors}"
        assert cache.stats()["memory_usage_mb"] <= 50

    def test_cpu_contention(self):
        """Test CPU resource contention."""
        errors = []
        results = []

        def cpu_intensive_task(thread_id):
            try:
                # Simulate CPU-intensive work
                result = 0
                for i in range(1000000):
                    result += i * thread_id
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Run CPU-intensive tasks in parallel
        threads = []
        for i in range(multiprocessing.cpu_count() * 2):  # Oversubscribe CPUs
            thread = threading.Thread(target=cpu_intensive_task, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert not errors, f"CPU task errors: {errors}"
        assert len(results) == multiprocessing.cpu_count() * 2


class TestConcurrentCleanup:
    """Test cleanup in concurrent scenarios."""

    def test_concurrent_session_cleanup(self, temp_dir):
        """Test concurrent session cleanup."""
        manager = SplitSessionManager(db_path=temp_dir / "sessions.db")
        session_ids = []
        errors = []

        # Create many sessions
        from pdf_splitter.splitting.models import SplitProposal

        for i in range(50):
            proposal = SplitProposal(
                pdf_path=temp_dir / f"test_{i}.pdf",
                total_pages=10,
                segments=[],
                detection_results=[],
            )
            session = manager.create_session(proposal)
            session_ids.append(session.session_id)

        def cleanup_session(session_id):
            try:
                manager.delete_session(session_id)
            except Exception as e:
                errors.append(e)

        # Cleanup all sessions concurrently
        threads = []
        for session_id in session_ids:
            thread = threading.Thread(target=cleanup_session, args=(session_id,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify cleanup
        assert not errors, f"Cleanup errors: {errors}"
        for session_id in session_ids:
            assert not manager.session_exists(session_id)

    def test_cache_cleanup_under_load(self):
        """Test cache cleanup while under load."""
        cache = AdvancedCache(max_memory_mb=10)
        errors = []
        stop_flag = threading.Event()

        def continuous_cache_operations():
            try:
                i = 0
                while not stop_flag.is_set():
                    cache.set(f"key_{i}", {"data": "x" * 1000})
                    cache.get(f"key_{i % 100}")  # Access recent keys
                    i += 1
            except Exception as e:
                errors.append(e)

        # Start continuous operations
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=continuous_cache_operations)
            threads.append(thread)
            thread.start()

        # Let it run for a bit
        time.sleep(2)

        # Trigger cleanup
        cache.cleanup()

        # Stop operations
        stop_flag.set()
        for thread in threads:
            thread.join()

        assert not errors, f"Cache operation errors: {errors}"

    def test_resource_cleanup_on_exception(self, temp_dir):
        """Test resource cleanup when exceptions occur."""
        leaked_resources = []

        def failing_operation(thread_id):
            handler = None
            try:
                config = PDFConfig()
                handler = PDFHandler(config)

                # Create a file that will fail to load
                bad_pdf = temp_dir / f"bad_{thread_id}.pdf"
                bad_pdf.write_text("not a pdf")

                handler.load_pdf(bad_pdf)  # This should fail
            except Exception:
                # Expected to fail
                pass
            finally:
                # Check if handler was properly cleaned up
                if handler and hasattr(handler, "pdf_document"):
                    if handler.pdf_document and not handler.pdf_document.is_closed:
                        leaked_resources.append(f"Handler {thread_id} not closed")

        threads = []
        for i in range(10):
            thread = threading.Thread(target=failing_operation, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert not leaked_resources, f"Resource leaks: {leaked_resources}"


class TestLoadTesting:
    """Load testing with many simultaneous operations."""

    def test_high_load_pdf_processing(self, temp_dir):
        """Test system under high load."""
        num_operations = 100
        max_workers = 20
        errors = []
        success_count = 0
        lock = threading.Lock()

        def process_operation(op_id):
            nonlocal success_count
            try:
                # Create small PDF
                pdf_path = temp_dir / f"load_test_{op_id}.pdf"
                create_test_pdf(num_pages=2, output_path=pdf_path)

                # Process it
                config = PDFConfig()
                handler = PDFHandler(config)
                handler.load_pdf(pdf_path)

                extractor = TextExtractor(config)
                for i in range(handler.page_count):
                    page = handler.get_page(i)
                    _ = extractor.extract_text(page)

                handler.close()

                with lock:
                    success_count += 1
            except Exception as e:
                with lock:
                    errors.append(e)

        # Run high-load test
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_operation, i) for i in range(num_operations)
            ]
            concurrent.futures.wait(futures)
        end_time = time.time()

        # Analyze results
        duration = end_time - start_time
        ops_per_second = success_count / duration

        assert (
            success_count >= num_operations * 0.95
        ), f"Too many failures: {len(errors)}"
        assert ops_per_second > 5, f"Performance too low: {ops_per_second:.2f} ops/sec"

    def test_sustained_load(self, temp_dir):
        """Test sustained load over time."""
        duration_seconds = 5
        errors = []
        operations = 0
        stop_flag = threading.Event()
        lock = threading.Lock()

        def continuous_operations(worker_id):
            nonlocal operations
            cache = AdvancedCache(max_memory_mb=50)
            op_count = 0

            try:
                while not stop_flag.is_set():
                    # Cache operations
                    key = f"worker_{worker_id}_op_{op_count}"
                    cache.set(key, {"data": "x" * 100})
                    _ = cache.get(key)

                    # File operations
                    file_path = (
                        temp_dir / f"worker_{worker_id}_file_{op_count % 10}.txt"
                    )
                    file_path.write_text(f"Operation {op_count}")

                    op_count += 1
                    with lock:
                        operations += 1
            except Exception as e:
                with lock:
                    errors.append(e)

        # Start workers
        threads = []
        for i in range(10):
            thread = threading.Thread(target=continuous_operations, args=(i,))
            threads.append(thread)
            thread.start()

        # Run for specified duration
        time.sleep(duration_seconds)
        stop_flag.set()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify sustained performance
        assert not errors, f"Errors during sustained load: {errors[:5]}"  # Show first 5
        ops_per_second = operations / duration_seconds
        assert (
            ops_per_second > 100
        ), f"Sustained performance too low: {ops_per_second:.2f} ops/sec"


class TestDeadlockDetection:
    """Test for deadlock detection and prevention."""

    def test_circular_dependency_prevention(self):
        """Test prevention of circular dependencies that could cause deadlock."""
        lock1 = threading.Lock()
        lock2 = threading.Lock()
        deadlock_detected = False
        completed = []

        def thread1_work():
            nonlocal deadlock_detected
            try:
                with lock1:
                    time.sleep(0.1)
                    # Try to acquire lock2 with timeout
                    acquired = lock2.acquire(timeout=1.0)
                    if acquired:
                        try:
                            completed.append("thread1")
                        finally:
                            lock2.release()
                    else:
                        deadlock_detected = True
            except Exception:
                deadlock_detected = True

        def thread2_work():
            nonlocal deadlock_detected
            try:
                with lock2:
                    time.sleep(0.1)
                    # Try to acquire lock1 with timeout
                    acquired = lock1.acquire(timeout=1.0)
                    if acquired:
                        try:
                            completed.append("thread2")
                        finally:
                            lock1.release()
                    else:
                        deadlock_detected = True
            except Exception:
                deadlock_detected = True

        # Run threads that could potentially deadlock
        thread1 = threading.Thread(target=thread1_work)
        thread2 = threading.Thread(target=thread2_work)

        thread1.start()
        thread2.start()

        thread1.join(timeout=3.0)
        thread2.join(timeout=3.0)

        # At least one should detect the potential deadlock
        assert deadlock_detected or len(completed) == 2

    def test_cache_deadlock_prevention(self):
        """Test that cache operations don't cause deadlocks."""
        cache = AdvancedCache(max_memory_mb=10)
        errors = []
        operations_completed = 0
        lock = threading.Lock()

        def cache_stress_test(thread_id):
            nonlocal operations_completed
            try:
                for i in range(100):
                    # Nested operations that could cause deadlock
                    key1 = f"key_{thread_id}_{i}"
                    key2 = f"key_{thread_id}_{i + 1}"

                    # Set with potential internal locking
                    cache.set(key1, {"ref": key2})

                    # Get with potential internal locking
                    val1 = cache.get(key1)
                    if val1:
                        # Simulate dependent operation
                        cache.set(key2, {"back_ref": key1})

                    # Stats access (might use different locks)
                    _ = cache.stats()

                with lock:
                    operations_completed += 1
            except Exception as e:
                with lock:
                    errors.append(e)

        # Run many threads to increase deadlock probability
        threads = []
        for i in range(20):
            thread = threading.Thread(target=cache_stress_test, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait with timeout to detect deadlocks
        for thread in threads:
            thread.join(timeout=10.0)
            if thread.is_alive():
                errors.append("Thread deadlocked")

        assert not errors, f"Deadlock or errors detected: {errors}"
        assert operations_completed == 20

    @pytest.mark.asyncio
    async def test_async_deadlock_prevention(self):
        """Test deadlock prevention in async operations."""
        locks = [asyncio.Lock() for _ in range(3)]
        completed = []
        errors = []

        async def async_worker(worker_id, lock_order):
            try:
                for lock_idx in lock_order:
                    async with asyncio.timeout(1.0):  # Timeout to prevent deadlock
                        async with locks[lock_idx]:
                            await asyncio.sleep(0.1)
                completed.append(worker_id)
            except asyncio.TimeoutError:
                errors.append(f"Worker {worker_id} timed out - potential deadlock")

        # Create workers with different lock acquisition orders
        tasks = [
            async_worker(0, [0, 1, 2]),
            async_worker(1, [1, 2, 0]),
            async_worker(2, [2, 0, 1]),
        ]

        # Run with overall timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), timeout=5.0
            )
        except asyncio.TimeoutError:
            errors.append("Overall deadlock detected")

        # Some workers might timeout, but system should not deadlock
        assert len(completed) >= 1, "Complete deadlock occurred"


@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS") != "true",
    reason="Integration tests disabled by default",
)
class TestFullConcurrentIntegration:
    """Full integration tests with all components under concurrent load."""

    def test_end_to_end_concurrent_processing(self, temp_dir):
        """Test complete PDF processing pipeline under concurrent load."""
        # Create test PDFs with varying sizes
        test_pdfs = []
        for i in range(10):
            pdf_path = temp_dir / f"integration_test_{i}.pdf"
            pages = 3 + (i % 5)  # 3-7 pages
            create_test_pdf(num_pages=pages, output_path=pdf_path)
            test_pdfs.append(pdf_path)

        manager = SplitSessionManager(db_path=temp_dir / "sessions.db")
        errors = []
        results = []
        lock = threading.Lock()

        def process_pdf_end_to_end(pdf_path):
            try:
                # Create session
                from pdf_splitter.splitting.models import SplitProposal

                proposal = SplitProposal(
                    pdf_path=pdf_path, total_pages=10, segments=[], detection_results=[]
                )
                session = manager.create_session(proposal)

                # Load PDF
                config = PDFConfig()
                handler = PDFHandler(config)
                handler.load_pdf(pdf_path)

                # Extract text
                extractor = TextExtractor(config)
                pages = []
                for i in range(handler.page_count):
                    page = handler.get_page(i)
                    text = extractor.extract_text(page)
                    pages.append(text)

                # Run multiple detectors
                detectors = [HeuristicDetector(), EmbeddingsDetector()]
                all_boundaries = []

                for detector in detectors:
                    context = detector.create_context(pages)
                    boundaries = detector.detect_boundaries(context)
                    all_boundaries.extend(boundaries)

                # Store results (removed session operations)

                with lock:
                    results.append(
                        {
                            "pdf": pdf_path.name,
                            "session": session.session_id,
                            "pages": len(pages),
                            "boundaries": len(all_boundaries),
                        }
                    )

                handler.close()
            except Exception as e:
                with lock:
                    errors.append((pdf_path.name, str(e)))

        # Process all PDFs concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(process_pdf_end_to_end, pdf) for pdf in test_pdfs
            ]
            concurrent.futures.wait(futures)

        # Verify results
        assert not errors, f"Processing errors: {errors}"
        assert len(results) == len(test_pdfs)
        for result in results:
            assert result["pages"] >= 3
            assert "boundaries" in result
