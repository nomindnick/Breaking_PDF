// Frontend Testing Utilities for PDF Splitter

// Mock DOM elements
export function createMockElement(tagName, attributes = {}) {
    const element = document.createElement(tagName);
    Object.entries(attributes).forEach(([key, value]) => {
        if (key === 'innerHTML' || key === 'textContent') {
            element[key] = value;
        } else {
            element.setAttribute(key, value);
        }
    });
    return element;
}

// Mock Alpine.js component
export function createAlpineComponent(data, template) {
    const container = document.createElement('div');
    container.innerHTML = template;

    // Simple Alpine mock
    const component = {
        $el: container.firstElementChild,
        $refs: {},
        $dispatch: jest.fn(),
        $nextTick: () => Promise.resolve(),
        ...data
    };

    // Collect refs
    container.querySelectorAll('[x-ref]').forEach(el => {
        const refName = el.getAttribute('x-ref');
        component.$refs[refName] = el;
    });

    return component;
}

// Mock API responses
export const mockApiResponses = {
    uploadSuccess: {
        ok: true,
        json: async () => ({
            session_id: 'test-session-123',
            upload_id: 'test-upload-456',
            message: 'Upload successful'
        })
    },

    uploadError: {
        ok: false,
        status: 413,
        statusText: 'Payload Too Large',
        json: async () => ({
            detail: 'File size exceeds maximum limit'
        })
    },

    sessionProposal: {
        ok: true,
        json: async () => ({
            segments: [
                {
                    id: 'seg-1',
                    start_page: 0,
                    end_page: 5,
                    type: 'letter',
                    title: 'Document 1',
                    confidence: 0.95
                },
                {
                    id: 'seg-2',
                    start_page: 6,
                    end_page: 10,
                    type: 'invoice',
                    title: 'Invoice #12345',
                    confidence: 0.87
                }
            ]
        })
    }
};

// Mock WebSocket
export class MockWebSocket {
    constructor(url) {
        this.url = url;
        this.readyState = WebSocket.CONNECTING;
        this.onopen = null;
        this.onmessage = null;
        this.onerror = null;
        this.onclose = null;

        // Simulate connection
        setTimeout(() => {
            this.readyState = WebSocket.OPEN;
            if (this.onopen) this.onopen(new Event('open'));
        }, 10);
    }

    send(data) {
        // Store sent messages for assertions
        this.sentMessages = this.sentMessages || [];
        this.sentMessages.push(data);
    }

    close() {
        this.readyState = WebSocket.CLOSED;
        if (this.onclose) this.onclose(new Event('close'));
    }

    // Test helper to simulate incoming messages
    simulateMessage(data) {
        if (this.onmessage) {
            this.onmessage(new MessageEvent('message', { data: JSON.stringify(data) }));
        }
    }

    // Test helper to simulate errors
    simulateError(error) {
        if (this.onerror) {
            this.onerror(new Event('error'));
        }
    }
}

// Mock file objects
export function createMockFile(name, size, type = 'application/pdf') {
    const file = new File(['a'.repeat(size)], name, { type });
    // Add any custom properties needed for testing
    Object.defineProperty(file, 'size', { value: size });
    return file;
}

// Wait utilities
export async function waitFor(condition, timeout = 1000) {
    const startTime = Date.now();

    while (!condition()) {
        if (Date.now() - startTime > timeout) {
            throw new Error('Timeout waiting for condition');
        }
        await new Promise(resolve => setTimeout(resolve, 10));
    }
}

export async function waitForElement(selector, container = document) {
    return waitFor(() => container.querySelector(selector));
}

// Event simulation
export function simulateEvent(element, eventType, eventData = {}) {
    const event = new Event(eventType, { bubbles: true, cancelable: true });
    Object.assign(event, eventData);
    element.dispatchEvent(event);
}

export function simulateDragAndDrop(element, files) {
    const dataTransfer = {
        files,
        items: files.map(file => ({
            kind: 'file',
            type: file.type,
            getAsFile: () => file
        })),
        types: ['Files']
    };

    simulateEvent(element, 'dragenter', { dataTransfer });
    simulateEvent(element, 'dragover', { dataTransfer });
    simulateEvent(element, 'drop', { dataTransfer });
}

// Performance testing helpers
export function measurePerformance(fn) {
    const start = performance.now();
    const result = fn();
    const duration = performance.now() - start;

    return {
        result: result instanceof Promise ? result : Promise.resolve(result),
        duration
    };
}

// Accessibility testing helpers
export function getAccessibleName(element) {
    // Simple implementation - real version would be more complex
    return element.getAttribute('aria-label') ||
           element.getAttribute('aria-labelledby') ||
           element.textContent?.trim() ||
           element.getAttribute('alt') ||
           element.getAttribute('title');
}

export function checkKeyboardAccessible(element) {
    const focusable = element.matches(`
        a[href],
        button:not([disabled]),
        input:not([disabled]),
        select:not([disabled]),
        textarea:not([disabled]),
        [tabindex]:not([tabindex="-1"])
    `);

    const hasRole = element.hasAttribute('role');
    const hasAriaLabel = element.hasAttribute('aria-label') ||
                        element.hasAttribute('aria-labelledby');

    return {
        focusable,
        hasRole,
        hasAriaLabel,
        isAccessible: focusable && (hasRole || hasAriaLabel || element.tagName !== 'DIV')
    };
}

// Local storage mock
export class LocalStorageMock {
    constructor() {
        this.store = {};
    }

    getItem(key) {
        return this.store[key] || null;
    }

    setItem(key, value) {
        this.store[key] = value.toString();
    }

    removeItem(key) {
        delete this.store[key];
    }

    clear() {
        this.store = {};
    }
}

// Setup and teardown helpers
export function setupTestEnvironment() {
    // Mock window.notify
    window.notify = {
        show: jest.fn(),
        hide: jest.fn(),
        clear: jest.fn()
    };

    // Mock localStorage
    const localStorageMock = new LocalStorageMock();
    Object.defineProperty(window, 'localStorage', {
        value: localStorageMock,
        writable: true
    });

    // Mock WebSocket
    global.WebSocket = MockWebSocket;

    // Clear DOM
    document.body.innerHTML = '<div id="notifications"></div>';
}

export function teardownTestEnvironment() {
    // Clear all mocks
    jest.clearAllMocks();

    // Clear DOM
    document.body.innerHTML = '';

    // Reset fetch if mocked
    if (window.fetch.mockRestore) {
        window.fetch.mockRestore();
    }
}

// Snapshot testing helpers
export function sanitizeSnapshot(html) {
    return html
        .replace(/id="[^"]+"/g, 'id="[ID]"')
        .replace(/class="[^"]*transition[^"]*"/g, 'class="[TRANSITION]"')
        .replace(/style="[^"]+"/g, 'style="[STYLE]"')
        .replace(/\s+/g, ' ')
        .trim();
}

// Error boundary testing
export function expectNoConsoleErrors(fn) {
    const originalError = console.error;
    const errors = [];

    console.error = (...args) => {
        errors.push(args);
    };

    try {
        fn();
        expect(errors).toHaveLength(0);
    } finally {
        console.error = originalError;
    }
}

// Export all utilities
export default {
    createMockElement,
    createAlpineComponent,
    mockApiResponses,
    MockWebSocket,
    createMockFile,
    waitFor,
    waitForElement,
    simulateEvent,
    simulateDragAndDrop,
    measurePerformance,
    getAccessibleName,
    checkKeyboardAccessible,
    LocalStorageMock,
    setupTestEnvironment,
    teardownTestEnvironment,
    sanitizeSnapshot,
    expectNoConsoleErrors
};
