// Unit tests for ErrorHandler
import { ErrorHandler } from '../../static/js/error-handler.js';
import { setupTestEnvironment, teardownTestEnvironment, waitFor } from '../test_utils.js';

describe('ErrorHandler', () => {
    let errorHandler;
    let originalFetch;

    beforeEach(() => {
        setupTestEnvironment();
        originalFetch = window.fetch;
        errorHandler = new ErrorHandler();
    });

    afterEach(() => {
        teardownTestEnvironment();
        window.fetch = originalFetch;
        errorHandler.destroy?.();
    });

    describe('Network Error Handling', () => {
        test('should retry failed requests with exponential backoff', async () => {
            let callCount = 0;
            window.fetch = jest.fn(() => {
                callCount++;
                if (callCount < 3) {
                    return Promise.reject(new TypeError('Failed to fetch'));
                }
                return Promise.resolve({ ok: true });
            });

            const response = await fetch('/api/test');

            expect(window.fetch).toHaveBeenCalledTimes(3);
            expect(response.ok).toBe(true);
            expect(window.notify.show).toHaveBeenCalledWith(
                expect.stringContaining('Connection issue'),
                'warning'
            );
        });

        test('should stop retrying after max attempts', async () => {
            window.fetch = jest.fn(() =>
                Promise.reject(new TypeError('Failed to fetch'))
            );

            await expect(fetch('/api/test')).rejects.toThrow('Failed to fetch');

            expect(window.fetch).toHaveBeenCalledTimes(4); // 1 original + 3 retries
        });

        test('should handle rate limit errors with retry-after header', async () => {
            const retryAfter = '5';
            window.fetch = jest.fn()
                .mockResolvedValueOnce({
                    ok: false,
                    status: 429,
                    statusText: 'Too Many Requests',
                    headers: new Map([['retry-after', retryAfter]]),
                    json: async () => ({ detail: 'Rate limited' })
                })
                .mockResolvedValueOnce({ ok: true });

            const startTime = Date.now();
            await fetch('/api/test');

            // Should wait approximately the retry-after time
            // (allowing some margin for test execution)
            expect(Date.now() - startTime).toBeGreaterThanOrEqual(4000);
            expect(window.notify.show).toHaveBeenCalledWith(
                expect.stringContaining('Rate limited'),
                'warning'
            );
        });
    });

    describe('Session Timeout Handling', () => {
        test('should handle 401 errors as session timeout', async () => {
            window.fetch = jest.fn().mockResolvedValue({
                ok: false,
                status: 401,
                statusText: 'Unauthorized',
                json: async () => ({ detail: 'Session expired' })
            });

            await expect(fetch('/api/test')).rejects.toThrow();

            expect(window.notify.show).toHaveBeenCalledWith(
                expect.stringContaining('session has expired'),
                expect.any(String)
            );
        });

        test('should save work in progress on session timeout', async () => {
            window.fetch = jest.fn().mockResolvedValue({
                ok: false,
                status: 401,
                statusText: 'Unauthorized',
                json: async () => ({ detail: 'Session expired' })
            });

            await expect(fetch('/api/test')).rejects.toThrow();

            const savedData = localStorage.getItem('pdf_splitter_recovery');
            expect(savedData).toBeTruthy();
            const parsed = JSON.parse(savedData);
            expect(parsed.timestamp).toBeDefined();
            expect(parsed.url).toBe(window.location.href);
        });
    });

    describe('Connection Monitoring', () => {
        test('should update connection status on successful requests', async () => {
            window.fetch = jest.fn().mockResolvedValue({ ok: true });

            await fetch('/api/test');

            const status = errorHandler.getConnectionStatus();
            expect(status.api).toBe(true);
        });

        test('should update connection status on failed requests', async () => {
            window.fetch = jest.fn().mockRejectedValue(new Error('Network error'));

            try {
                await fetch('/api/test');
            } catch (e) {
                // Expected to fail
            }

            const status = errorHandler.getConnectionStatus();
            expect(status.api).toBe(false);
        });

        test('should trigger connection status event', async () => {
            const statusHandler = jest.fn();
            window.addEventListener('connectionStatusChanged', statusHandler);

            // Simulate going offline
            errorHandler.updateConnectionStatus('api', false);

            expect(statusHandler).toHaveBeenCalledWith(
                expect.objectContaining({
                    detail: expect.objectContaining({
                        service: 'api',
                        isConnected: false
                    })
                })
            );

            window.removeEventListener('connectionStatusChanged', statusHandler);
        });
    });

    describe('Error Logging', () => {
        test('should log errors with context', () => {
            const error = new Error('Test error');
            const context = { component: 'TestComponent', action: 'testAction' };

            errorHandler.handleError(error, context);

            const errorLog = errorHandler.getErrorLog();
            expect(errorLog).toHaveLength(1);
            expect(errorLog[0]).toMatchObject({
                message: 'Test error',
                context,
                timestamp: expect.any(Number)
            });
        });

        test('should limit error log size', () => {
            // Fill error log beyond limit
            for (let i = 0; i < 60; i++) {
                errorHandler.logError(new Error(`Error ${i}`), {});
            }

            const errorLog = errorHandler.getErrorLog();
            expect(errorLog).toHaveLength(50); // Max log size
            expect(errorLog[0].message).toBe('Error 10'); // First 10 should be removed
        });
    });

    describe('User-Friendly Error Messages', () => {
        test('should convert technical errors to user-friendly messages', () => {
            const testCases = [
                {
                    error: new TypeError('Failed to fetch'),
                    expected: 'Unable to connect to the server'
                },
                {
                    error: { type: 'SessionTimeout' },
                    expected: 'Your session has expired'
                },
                {
                    error: { type: 'RateLimitError' },
                    expected: 'Too many requests'
                }
            ];

            testCases.forEach(({ error, expected }) => {
                errorHandler.showErrorMessage(error);

                expect(window.notify.show).toHaveBeenCalledWith(
                    expect.stringContaining(expected),
                    expect.any(String),
                    expect.any(Number)
                );
            });
        });
    });

    describe('Recovery Strategies', () => {
        test('should wait for connection before retrying', async () => {
            // Mock offline state
            Object.defineProperty(navigator, 'onLine', {
                writable: true,
                value: false
            });
            errorHandler.updateConnectionStatus('api', false);

            const waitPromise = errorHandler.waitForConnection(1000);

            // Simulate coming back online after 500ms
            setTimeout(() => {
                navigator.onLine = true;
                errorHandler.updateConnectionStatus('api', true);
            }, 500);

            await expect(waitPromise).resolves.toBeUndefined();
        });

        test('should timeout if connection not restored', async () => {
            Object.defineProperty(navigator, 'onLine', {
                writable: true,
                value: false
            });
            errorHandler.updateConnectionStatus('api', false);

            await expect(
                errorHandler.waitForConnection(100)
            ).rejects.toThrow('Connection timeout');
        });
    });

    describe('Global Error Handling', () => {
        test('should catch unhandled promise rejections', () => {
            const error = new Error('Unhandled rejection');

            window.dispatchEvent(new PromiseRejectionEvent('unhandledrejection', {
                reason: error,
                promise: Promise.reject(error)
            }));

            expect(window.notify.show).toHaveBeenCalledWith(
                expect.stringContaining('Unhandled rejection'),
                'error'
            );
        });

        test('should catch global errors', () => {
            const error = new Error('Global error');

            window.dispatchEvent(new ErrorEvent('error', {
                error,
                message: error.message
            }));

            const errorLog = errorHandler.getErrorLog();
            expect(errorLog).toContainEqual(
                expect.objectContaining({
                    message: 'Global error',
                    type: 'Unknown'
                })
            );
        });
    });
});
