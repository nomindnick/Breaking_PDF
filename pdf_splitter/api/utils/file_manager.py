"""
File Management Utilities

Provides utilities for file operations, cleanup, and maintenance.
"""
import asyncio
import hashlib
import mimetypes
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiofiles

from pdf_splitter.api.config import config


class FileManager:
    """Utility class for file management operations."""

    def __init__(self):
        self.upload_dir = config.upload_dir
        self.output_dir = config.output_dir

    async def get_directory_stats(self, directory: Path) -> Dict[str, Any]:
        """
        Get statistics for a directory.

        Args:
            directory: Directory path

        Returns:
            Directory statistics
        """
        if not directory.exists():
            return {
                "exists": False,
                "total_files": 0,
                "total_size": 0,
                "error": "Directory does not exist",
            }

        total_files = 0
        total_size = 0
        file_types = {}

        for file_path in directory.rglob("*"):
            if file_path.is_file():
                total_files += 1
                file_size = file_path.stat().st_size
                total_size += file_size

                # Track file types
                suffix = file_path.suffix.lower()
                if suffix in file_types:
                    file_types[suffix]["count"] += 1
                    file_types[suffix]["size"] += file_size
                else:
                    file_types[suffix] = {"count": 1, "size": file_size}

        return {
            "exists": True,
            "path": str(directory),
            "total_files": total_files,
            "total_size": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "file_types": file_types,
            "subdirectories": len([d for d in directory.iterdir() if d.is_dir()]),
            "oldest_file": self._get_oldest_file(directory),
            "newest_file": self._get_newest_file(directory),
            "last_modified": datetime.fromtimestamp(
                directory.stat().st_mtime
            ).isoformat(),
        }

    async def cleanup_old_files(
        self,
        directory: Path,
        days_old: int,
        file_patterns: Optional[List[str]] = None,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """
        Clean up old files from a directory.

        Args:
            directory: Directory to clean
            days_old: Delete files older than this many days
            file_patterns: Optional file patterns to match (e.g., ["*.pdf", "*.tmp"])
            dry_run: If True, only report what would be deleted

        Returns:
            Cleanup results
        """
        if not directory.exists():
            return {"error": "Directory does not exist"}

        cutoff_date = datetime.now() - timedelta(days=days_old)
        files_to_delete = []
        total_size = 0

        # Find files to delete
        patterns = file_patterns or ["*"]
        for pattern in patterns:
            for file_path in directory.rglob(pattern):
                if file_path.is_file():
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_mtime < cutoff_date:
                        files_to_delete.append(file_path)
                        total_size += file_path.stat().st_size

        # Delete files if not dry run
        deleted_count = 0
        errors = []

        if not dry_run:
            for file_path in files_to_delete:
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    errors.append({"file": str(file_path), "error": str(e)})

        return {
            "dry_run": dry_run,
            "files_found": len(files_to_delete),
            "files_deleted": deleted_count,
            "total_size": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "errors": errors,
            "cutoff_date": cutoff_date.isoformat(),
        }

    async def cleanup_empty_directories(
        self, directory: Path, dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Remove empty directories.

        Args:
            directory: Root directory to clean
            dry_run: If True, only report what would be deleted

        Returns:
            Cleanup results
        """
        empty_dirs = []

        # Find empty directories (bottom-up)
        for dirpath in sorted(directory.rglob("*"), reverse=True):
            if dirpath.is_dir() and not any(dirpath.iterdir()):
                empty_dirs.append(dirpath)

        # Remove if not dry run
        removed_count = 0
        errors = []

        if not dry_run:
            for empty_dir in empty_dirs:
                try:
                    empty_dir.rmdir()
                    removed_count += 1
                except Exception as e:
                    errors.append({"directory": str(empty_dir), "error": str(e)})

        return {
            "dry_run": dry_run,
            "empty_dirs_found": len(empty_dirs),
            "dirs_removed": removed_count,
            "errors": errors,
        }

    async def validate_file_integrity(
        self,
        file_path: Path,
        expected_checksum: Optional[str] = None,
        algorithm: str = "md5",
    ) -> Dict[str, Any]:
        """
        Validate file integrity.

        Args:
            file_path: File to validate
            expected_checksum: Expected checksum value
            algorithm: Hash algorithm (md5, sha1, sha256)

        Returns:
            Validation results
        """
        if not file_path.exists():
            return {"valid": False, "error": "File does not exist"}

        try:
            # Calculate checksum
            actual_checksum = await self._calculate_file_checksum(file_path, algorithm)

            result = {
                "valid": True,
                "file": str(file_path),
                "size": file_path.stat().st_size,
                "algorithm": algorithm,
                "checksum": actual_checksum,
                "modified": datetime.fromtimestamp(
                    file_path.stat().st_mtime
                ).isoformat(),
            }

            # Verify against expected if provided
            if expected_checksum:
                result["expected_checksum"] = expected_checksum
                result["checksum_match"] = actual_checksum == expected_checksum
                result["valid"] = result["checksum_match"]

            return result

        except Exception as e:
            return {"valid": False, "error": str(e)}

    async def archive_files(
        self, files: List[Path], archive_path: Path, compression: str = "zip"
    ) -> Dict[str, Any]:
        """
        Archive multiple files.

        Args:
            files: List of files to archive
            archive_path: Output archive path
            compression: Archive format (zip, tar, tar.gz)

        Returns:
            Archive results
        """
        try:
            if compression == "zip":
                import zipfile

                with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for file_path in files:
                        if file_path.exists():
                            zf.write(file_path, file_path.name)

            elif compression in ["tar", "tar.gz"]:
                import tarfile

                mode = "w:gz" if compression == "tar.gz" else "w"
                with tarfile.open(archive_path, mode) as tf:
                    for file_path in files:
                        if file_path.exists():
                            tf.add(file_path, arcname=file_path.name)

            else:
                return {
                    "success": False,
                    "error": f"Unsupported compression: {compression}",
                }

            return {
                "success": True,
                "archive_path": str(archive_path),
                "files_archived": len([f for f in files if f.exists()]),
                "archive_size": archive_path.stat().st_size,
                "compression": compression,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def move_files(
        self,
        source_dir: Path,
        dest_dir: Path,
        file_pattern: str = "*",
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """
        Move files between directories.

        Args:
            source_dir: Source directory
            dest_dir: Destination directory
            file_pattern: File pattern to match
            overwrite: Whether to overwrite existing files

        Returns:
            Move results
        """
        if not source_dir.exists():
            return {"success": False, "error": "Source directory does not exist"}

        # Create destination if needed
        dest_dir.mkdir(parents=True, exist_ok=True)

        moved_files = []
        errors = []

        for file_path in source_dir.glob(file_pattern):
            if file_path.is_file():
                dest_path = dest_dir / file_path.name

                try:
                    if dest_path.exists() and not overwrite:
                        errors.append(
                            {
                                "file": str(file_path),
                                "error": "Destination exists and overwrite is False",
                            }
                        )
                        continue

                    shutil.move(str(file_path), str(dest_path))
                    moved_files.append(
                        {"source": str(file_path), "destination": str(dest_path)}
                    )

                except Exception as e:
                    errors.append({"file": str(file_path), "error": str(e)})

        return {
            "success": len(errors) == 0,
            "moved_count": len(moved_files),
            "moved_files": moved_files,
            "errors": errors,
        }

    def get_mime_type(self, file_path: Path) -> str:
        """Get MIME type for a file."""
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or "application/octet-stream"

    def generate_unique_filename(
        self, directory: Path, base_name: str, extension: str
    ) -> Path:
        """
        Generate a unique filename in a directory.

        Args:
            directory: Target directory
            base_name: Base filename
            extension: File extension

        Returns:
            Unique file path
        """
        # Clean base name
        base_name = "".join(c for c in base_name if c.isalnum() or c in "._- ")

        # Try original name first
        file_path = directory / f"{base_name}{extension}"
        if not file_path.exists():
            return file_path

        # Add counter
        counter = 1
        while True:
            file_path = directory / f"{base_name}_{counter}{extension}"
            if not file_path.exists():
                return file_path
            counter += 1

    async def _calculate_file_checksum(
        self, file_path: Path, algorithm: str = "md5"
    ) -> str:
        """Calculate file checksum asynchronously."""
        hash_func = hashlib.new(algorithm)

        async with aiofiles.open(file_path, "rb") as f:
            while chunk := await f.read(8192):
                hash_func.update(chunk)

        return hash_func.hexdigest()

    def _get_oldest_file(self, directory: Path) -> Optional[Dict[str, Any]]:
        """Get oldest file in directory."""
        oldest_file = None
        oldest_time = None

        for file_path in directory.rglob("*"):
            if file_path.is_file():
                mtime = file_path.stat().st_mtime
                if oldest_time is None or mtime < oldest_time:
                    oldest_time = mtime
                    oldest_file = file_path

        if oldest_file:
            return {
                "path": str(oldest_file),
                "name": oldest_file.name,
                "modified": datetime.fromtimestamp(oldest_time).isoformat(),
            }

        return None

    def _get_newest_file(self, directory: Path) -> Optional[Dict[str, Any]]:
        """Get newest file in directory."""
        newest_file = None
        newest_time = None

        for file_path in directory.rglob("*"):
            if file_path.is_file():
                mtime = file_path.stat().st_mtime
                if newest_time is None or mtime > newest_time:
                    newest_time = mtime
                    newest_file = file_path

        if newest_file:
            return {
                "path": str(newest_file),
                "name": newest_file.name,
                "modified": datetime.fromtimestamp(newest_time).isoformat(),
            }

        return None


# Global file manager instance
file_manager = FileManager()
