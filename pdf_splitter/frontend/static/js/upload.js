// File upload component for PDF Splitter
function fileUpload() {
    return {
        file: null,
        isDragging: false,
        uploading: false,
        uploadProgress: 0,
        uploadStatus: '',
        uploadComplete: false,
        error: null,
        uploadId: null,
        maxFileSizeMB: window.maxFileSizeMB || 500,

        handleDrop(e) {
            this.isDragging = false;
            this.error = null;

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.validateAndSetFile(files[0]);
            }
        },

        handleFileSelect(e) {
            this.error = null;
            const files = e.target.files;
            if (files.length > 0) {
                this.validateAndSetFile(files[0]);
            }
        },

        validateAndSetFile(file) {
            // Clear previous state
            this.uploadComplete = false;
            this.uploadProgress = 0;
            this.error = null;

            // Validate file type
            if (file.type !== 'application/pdf' && !file.name.toLowerCase().endsWith('.pdf')) {
                this.error = 'Please select a PDF file';
                this.file = null;
                return;
            }

            // Validate file size
            const maxSizeBytes = this.maxFileSizeMB * 1024 * 1024;
            if (file.size > maxSizeBytes) {
                this.error = `File size exceeds ${this.maxFileSizeMB}MB limit`;
                this.file = null;
                return;
            }

            this.file = file;
        },

        removeFile() {
            this.file = null;
            this.uploadProgress = 0;
            this.uploadStatus = '';
            this.error = null;
            this.uploadComplete = false;
            this.$refs.fileInput.value = '';
        },

        formatFileSize(bytes) {
            if (!bytes) return '';
            const mb = bytes / (1024 * 1024);
            if (mb > 1) {
                return mb.toFixed(2) + ' MB';
            } else {
                const kb = bytes / 1024;
                return kb.toFixed(0) + ' KB';
            }
        },

        async uploadFile() {
            if (!this.file || this.uploading) return;

            this.uploading = true;
            this.error = null;
            this.uploadStatus = 'Preparing upload...';

            const formData = new FormData();
            formData.append('file', this.file);

            try {
                // Create XMLHttpRequest for progress tracking
                const xhr = new XMLHttpRequest();

                // Track upload progress
                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable) {
                        this.uploadProgress = Math.round((e.loaded / e.total) * 100);
                        this.uploadStatus = 'Uploading...';
                    }
                });

                // Handle completion
                xhr.addEventListener('load', () => {
                    if (xhr.status === 200) {
                        try {
                            const response = JSON.parse(xhr.responseText);
                            this.uploadId = response.upload_id;
                            this.uploadComplete = true;
                            this.uploadStatus = 'Upload complete!';

                            // Show success notification
                            if (window.notify) {
                                window.notify.show('PDF uploaded successfully!', 'success');
                            }

                            // Redirect to progress page after a short delay
                            setTimeout(() => {
                                if (response.session_id) {
                                    // Redirect to progress page to track detection
                                    window.location.href = `/progress/${response.session_id}`;
                                } else {
                                    // If no session_id, create a new session
                                    this.createSessionAndRedirect();
                                }
                            }, 1500);
                        } catch (e) {
                            this.handleUploadError('Invalid response from server');
                        }
                    } else {
                        this.handleUploadError(`Upload failed: ${xhr.statusText}`);
                    }
                });

                // Handle errors
                xhr.addEventListener('error', () => {
                    this.handleUploadError('Network error occurred');
                });

                // Send request
                xhr.open('POST', '/api/upload/file');
                xhr.send(formData);

            } catch (error) {
                this.handleUploadError(error.message || 'Upload failed');
            }
        },

        handleUploadError(message) {
            this.uploading = false;
            this.uploadProgress = 0;
            this.uploadStatus = '';
            this.error = message;

            // Show error notification
            if (window.notify) {
                window.notify.show(message, 'error', 5000);
            }
        },

        async createSessionAndRedirect() {
            try {
                // Create a new session with the uploaded file
                const response = await fetch('/api/sessions', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        upload_id: this.uploadId,
                        filename: this.file.name
                    })
                });

                if (response.ok) {
                    const data = await response.json();
                    // Redirect to progress page to track detection
                    window.location.href = `/progress/${data.session_id}`;
                } else {
                    throw new Error('Failed to create session');
                }
            } catch (error) {
                this.handleUploadError('Failed to start processing session');
            }
        }
    }
}
