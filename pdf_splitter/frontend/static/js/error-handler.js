// Global Error Handling and Recovery System for PDF Splitter
class ErrorHandler {
    constructor() {
        this.retryAttempts = new Map();
        this.maxRetries = 3;
        this.retryDelay = 1000;
        this.errorLog = [];
        this.recoveryStrategies = new Map();

        // Initialize error recovery strategies
        this.initializeRecoveryStrategies();

        // Set up global error listeners
        this.setupGlobalListeners();

        // Connection status tracking
        this.connectionStatus = {
            api: true,
            websocket: true,
            lastCheck: Date.now()
        };

        // Start connection monitoring
        this.startConnectionMonitoring();
    }

    initializeRecoveryStrategies() {
        // Network errors
        this.recoveryStrategies.set('NetworkError', {
            canRetry: true,
            message: 'Connection issue detected. Retrying...',
            action: async (error, context) => {
                await this.waitForConnection();
                return true;
            }
        });

        // File upload errors
        this.recoveryStrategies.set('FileUploadError', {
            canRetry: true,
            message: 'Upload failed. Retrying...',
            action: async (error, context) => {
                // Resume upload from last checkpoint if possible
                if (context.resumable) {
                    return await this.resumeUpload(context);
                }
                return true;
            }
        });

        // Session timeout
        this.recoveryStrategies.set('SessionTimeout', {
            canRetry: false,
            message: 'Your session has expired.',
            action: async (error, context) => {
                // Offer to save work and start new session
                const saved = await this.saveWorkInProgress(context);
                if (saved) {
                    window.notify.show('Your work has been saved. Please start a new session.', 'info');
                    setTimeout(() => window.location.href = '/', 3000);
                }
                return false;
            }
        });

        // Rate limiting
        this.recoveryStrategies.set('RateLimitError', {
            canRetry: true,
            message: 'Too many requests. Please wait...',
            action: async (error, context) => {
                const waitTime = error.retryAfter || 60;
                window.notify.show(`Rate limited. Retrying in ${waitTime} seconds...`, 'warning');
                await this.delay(waitTime * 1000);
                return true;
            }
        });

        // WebSocket disconnection
        this.recoveryStrategies.set('WebSocketError', {
            canRetry: true,
            message: 'Real-time connection lost. Reconnecting...',
            action: async (error, context) => {
                // WebSocket manager handles its own reconnection
                // Just update UI status
                this.updateConnectionStatus('websocket', false);
                return false;
            }
        });
    }

    setupGlobalListeners() {
        // Catch unhandled promise rejections
        window.addEventListener('unhandledrejection', (event) => {
            console.error('Unhandled promise rejection:', event.reason);
            this.handleError(event.reason, {
                type: 'UnhandledRejection',
                critical: true
            });
            event.preventDefault();
        });

        // Catch global errors
        window.addEventListener('error', (event) => {
            console.error('Global error:', event.error);
            this.handleError(event.error, {
                type: 'GlobalError',
                critical: true
            });
        });

        // Monitor fetch requests
        this.interceptFetch();
    }

    interceptFetch() {
        const originalFetch = window.fetch;

        window.fetch = async (...args) => {
            const [url, options = {}] = args;
            const context = {
                url,
                method: options.method || 'GET',
                timestamp: Date.now()
            };

            try {
                const response = await originalFetch(...args);

                // Check for error responses
                if (!response.ok) {
                    const error = new Error(`HTTP ${response.status}: ${response.statusText}`);
                    error.response = response;
                    error.status = response.status;

                    // Try to parse error body
                    try {
                        const data = await response.clone().json();
                        error.detail = data.detail || data.message;
                    } catch (e) {
                        // Ignore JSON parse errors
                    }

                    // Determine error type
                    if (response.status === 429) {
                        error.type = 'RateLimitError';
                        error.retryAfter = response.headers.get('Retry-After');
                    } else if (response.status === 401 || response.status === 403) {
                        error.type = 'SessionTimeout';
                    } else if (response.status >= 500) {
                        error.type = 'ServerError';
                    } else {
                        error.type = 'RequestError';
                    }

                    // Handle with retry logic
                    return await this.handleRequestError(error, context, originalFetch, args);
                }

                // Update connection status
                this.updateConnectionStatus('api', true);
                return response;

            } catch (error) {
                // Network errors
                if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
                    error.type = 'NetworkError';
                }

                return await this.handleRequestError(error, context, originalFetch, args);
            }
        };
    }

    async handleRequestError(error, context, originalFetch, fetchArgs) {
        const errorType = error.type || 'UnknownError';
        const strategy = this.recoveryStrategies.get(errorType);

        if (strategy && strategy.canRetry) {
            const attemptKey = `${context.method}:${context.url}`;
            const attempts = this.retryAttempts.get(attemptKey) || 0;

            if (attempts < this.maxRetries) {
                this.retryAttempts.set(attemptKey, attempts + 1);

                // Show retry notification
                window.notify.show(
                    `${strategy.message} (Attempt ${attempts + 1}/${this.maxRetries})`,
                    'warning'
                );

                // Execute recovery action
                const shouldRetry = await strategy.action(error, context);

                if (shouldRetry) {
                    // Exponential backoff
                    const delay = this.retryDelay * Math.pow(2, attempts);
                    await this.delay(delay);

                    // Retry the request
                    try {
                        const response = await originalFetch(...fetchArgs);
                        this.retryAttempts.delete(attemptKey);
                        return response;
                    } catch (retryError) {
                        // Retry failed, continue to error handling
                        error = retryError;
                    }
                }
            }

            // Max retries reached
            this.retryAttempts.delete(attemptKey);
        }

        // Log error
        this.logError(error, context);

        // Show user-friendly error message
        this.showErrorMessage(error);

        // Throw error for calling code to handle
        throw error;
    }

    async handleError(error, context = {}) {
        // Normalize error
        const normalizedError = this.normalizeError(error);

        // Log error
        this.logError(normalizedError, context);

        // Determine recovery strategy
        const errorType = normalizedError.type || context.type || 'UnknownError';
        const strategy = this.recoveryStrategies.get(errorType);

        if (strategy) {
            // Execute recovery action
            try {
                await strategy.action(normalizedError, context);
            } catch (recoveryError) {
                console.error('Recovery failed:', recoveryError);
                this.showErrorMessage(normalizedError, true);
            }
        } else {
            // No recovery strategy, show error
            this.showErrorMessage(normalizedError, context.critical);
        }
    }

    normalizeError(error) {
        if (error instanceof Error) {
            return error;
        }

        if (typeof error === 'string') {
            return new Error(error);
        }

        if (error && typeof error === 'object') {
            const normalizedError = new Error(error.message || 'Unknown error');
            Object.assign(normalizedError, error);
            return normalizedError;
        }

        return new Error('Unknown error occurred');
    }

    logError(error, context) {
        const errorEntry = {
            timestamp: Date.now(),
            message: error.message,
            type: error.type || 'Unknown',
            stack: error.stack,
            context,
            userAgent: navigator.userAgent,
            url: window.location.href
        };

        this.errorLog.push(errorEntry);

        // Keep only last 50 errors
        if (this.errorLog.length > 50) {
            this.errorLog.shift();
        }

        // Send to monitoring service if configured
        if (window.errorReporting) {
            window.errorReporting.log(errorEntry);
        }
    }

    showErrorMessage(error, critical = false) {
        let message = error.detail || error.message || 'An unexpected error occurred';

        // Make message user-friendly
        const userFriendlyMessages = {
            'Failed to fetch': 'Unable to connect to the server. Please check your internet connection.',
            'NetworkError': 'Connection lost. Please check your internet connection.',
            'SessionTimeout': 'Your session has expired. Please refresh the page.',
            'RateLimitError': 'Too many requests. Please wait a moment and try again.',
            'ServerError': 'The server encountered an error. Please try again later.',
            'FileUploadError': 'File upload failed. Please try again.',
            'ValidationError': 'Please check your input and try again.'
        };

        for (const [key, friendly] of Object.entries(userFriendlyMessages)) {
            if (message.includes(key) || error.type === key) {
                message = friendly;
                break;
            }
        }

        // Add recovery suggestion
        const suggestions = {
            'NetworkError': 'Try refreshing the page or checking your connection.',
            'SessionTimeout': 'Your work has been saved. Please log in again.',
            'FileUploadError': 'Make sure the file is a valid PDF and under the size limit.'
        };

        const suggestion = suggestions[error.type];
        if (suggestion) {
            message += ` ${suggestion}`;
        }

        // Show notification
        window.notify.show(message, critical ? 'error' : 'warning', critical ? 8000 : 5000);
    }

    // Connection monitoring
    startConnectionMonitoring() {
        // Check connection every 30 seconds
        setInterval(() => this.checkConnections(), 30000);

        // Monitor online/offline events
        window.addEventListener('online', () => {
            this.updateConnectionStatus('api', true);
            window.notify.show('Connection restored', 'success');
        });

        window.addEventListener('offline', () => {
            this.updateConnectionStatus('api', false);
            window.notify.show('Connection lost. Some features may be unavailable.', 'warning');
        });
    }

    async checkConnections() {
        // Check API connection
        try {
            const response = await fetch('/health', {
                method: 'GET',
                cache: 'no-cache'
            });
            this.updateConnectionStatus('api', response.ok);
        } catch (error) {
            this.updateConnectionStatus('api', false);
        }
    }

    updateConnectionStatus(service, isConnected) {
        const previousStatus = this.connectionStatus[service];
        this.connectionStatus[service] = isConnected;
        this.connectionStatus.lastCheck = Date.now();

        // Update UI indicator
        const indicator = document.getElementById('connection-status');
        if (indicator) {
            const allConnected = this.connectionStatus.api && this.connectionStatus.websocket;
            indicator.classList.toggle('connected', allConnected);
            indicator.classList.toggle('disconnected', !allConnected);

            // Update tooltip
            const status = allConnected ? 'All systems operational' :
                         !this.connectionStatus.api ? 'API connection lost' :
                         'Real-time updates unavailable';
            indicator.setAttribute('title', status);
        }

        // Emit event for other components
        if (previousStatus !== isConnected) {
            window.dispatchEvent(new CustomEvent('connectionStatusChanged', {
                detail: { service, isConnected, status: this.connectionStatus }
            }));
        }
    }

    // Utility methods
    async waitForConnection(maxWait = 30000) {
        const startTime = Date.now();

        while (!navigator.onLine || !this.connectionStatus.api) {
            if (Date.now() - startTime > maxWait) {
                throw new Error('Connection timeout');
            }
            await this.delay(1000);
        }
    }

    async resumeUpload(context) {
        // Implement resumable upload logic
        // This would integrate with the upload service
        console.log('Resuming upload:', context);
        return true;
    }

    async saveWorkInProgress(context) {
        try {
            // Save current state to localStorage
            const state = {
                timestamp: Date.now(),
                url: window.location.href,
                data: context.data || {}
            };

            localStorage.setItem('pdf_splitter_recovery', JSON.stringify(state));
            return true;
        } catch (error) {
            console.error('Failed to save work:', error);
            return false;
        }
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // Public API
    getErrorLog() {
        return [...this.errorLog];
    }

    clearErrorLog() {
        this.errorLog = [];
    }

    getConnectionStatus() {
        return { ...this.connectionStatus };
    }

    // Manual error reporting
    report(message, context = {}) {
        const error = new Error(message);
        error.type = 'UserReported';
        this.handleError(error, context);
    }
}

// Initialize global error handler
window.errorHandler = new ErrorHandler();

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ErrorHandler;
}
