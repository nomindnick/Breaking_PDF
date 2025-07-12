# Frontend Testing Guide for PDF Splitter

This directory contains the frontend testing infrastructure for the PDF Splitter application.

## Testing Strategy

### 1. Unit Tests
- Test individual JavaScript modules and utilities
- Mock dependencies and DOM elements
- Focus on pure logic and data transformations

### 2. Component Tests
- Test Alpine.js components in isolation
- Verify component lifecycle and state management
- Test user interactions and events

### 3. Integration Tests
- Test interactions between multiple components
- Verify API integration and WebSocket communication
- Test complete user workflows

### 4. Visual Regression Tests
- Capture screenshots of key UI states
- Compare against baseline images
- Detect unintended visual changes

### 5. End-to-End Tests
- Test complete user journeys
- Verify application works from upload to download
- Test error scenarios and edge cases

## Test Structure

```
frontend/tests/
├── unit/               # Unit tests for utilities and modules
├── components/         # Component-level tests
├── integration/        # Integration tests
├── visual/            # Visual regression tests
├── e2e/               # End-to-end tests
├── fixtures/          # Test data and mocks
├── utils/             # Testing utilities
└── README.md          # This file
```

## Running Tests

### Unit Tests
```bash
# Run all unit tests
npm test

# Run specific test file
npm test -- unit/error-handler.test.js

# Run with coverage
npm test -- --coverage
```

### Visual Tests
```bash
# Run visual regression tests
npm run test:visual

# Update baseline images
npm run test:visual:update
```

### E2E Tests
```bash
# Run end-to-end tests
npm run test:e2e

# Run in headed mode for debugging
npm run test:e2e:debug
```

## Testing Tools

- **Jest**: Main testing framework
- **Alpine Test Utils**: For testing Alpine.js components
- **Puppeteer**: For visual and E2E testing
- **MSW**: Mock Service Worker for API mocking
- **Testing Library**: DOM testing utilities

## Writing Tests

### Unit Test Example
```javascript
import { ErrorHandler } from '../../static/js/error-handler.js';

describe('ErrorHandler', () => {
    let errorHandler;

    beforeEach(() => {
        errorHandler = new ErrorHandler();
    });

    test('should retry failed requests', async () => {
        const mockFetch = jest.fn()
            .mockRejectedValueOnce(new Error('Network error'))
            .mockResolvedValueOnce({ ok: true });

        window.fetch = mockFetch;

        const response = await fetch('/api/test');

        expect(mockFetch).toHaveBeenCalledTimes(2);
        expect(response.ok).toBe(true);
    });
});
```

### Component Test Example
```javascript
import { mount } from '@alpinejs/test-utils';

describe('FileUpload Component', () => {
    test('should handle file selection', async () => {
        const component = await mount(`
            <div x-data="fileUpload()">
                <input x-ref="fileInput" type="file" @change="handleFileSelect">
            </div>
        `);

        const file = new File(['test'], 'test.pdf', { type: 'application/pdf' });
        const input = component.$refs.fileInput;

        await component.trigger(input, 'change', { target: { files: [file] } });

        expect(component.file).toEqual(file);
    });
});
```

### E2E Test Example
```javascript
describe('PDF Upload and Split', () => {
    test('should complete full workflow', async () => {
        await page.goto('http://localhost:8000');

        // Navigate to upload
        await page.click('a[href="/upload"]');

        // Upload file
        const fileInput = await page.$('input[type="file"]');
        await fileInput.uploadFile('./fixtures/test.pdf');

        // Wait for processing
        await page.waitForSelector('.progress-complete');

        // Verify results
        const documentCount = await page.$$eval('.document-card', els => els.length);
        expect(documentCount).toBeGreaterThan(0);
    });
});
```

## Best Practices

1. **Keep tests focused**: Each test should verify one specific behavior
2. **Use descriptive names**: Test names should clearly describe what they test
3. **Mock external dependencies**: Don't make real API calls in tests
4. **Clean up after tests**: Reset state and remove test artifacts
5. **Test edge cases**: Include tests for error conditions and boundary values
6. **Maintain test data**: Keep fixtures up-to-date and realistic

## Continuous Integration

Tests run automatically on:
- Every push to the repository
- Pull request creation and updates
- Scheduled nightly runs for E2E tests

## Debugging Tests

### Visual Debugging
```bash
# Run tests with browser visible
HEADLESS=false npm run test:e2e
```

### Step Debugging
```javascript
// Add debugger statement in test
debugger;

// Run with inspector
node --inspect-brk node_modules/.bin/jest
```

### Screenshot on Failure
Tests automatically capture screenshots when they fail, saved to:
```
frontend/tests/screenshots/failures/
```

## Performance Testing

Monitor frontend performance metrics:
- Page load time
- Time to interactive
- API response times
- Memory usage

```javascript
test('should load within performance budget', async () => {
    const metrics = await page.metrics();

    expect(metrics.LayoutDuration).toBeLessThan(100);
    expect(metrics.ScriptDuration).toBeLessThan(200);
    expect(metrics.JSHeapUsedSize).toBeLessThan(50 * 1024 * 1024); // 50MB
});
```

## Accessibility Testing

All components should pass accessibility tests:

```javascript
import { axe, toHaveNoViolations } from 'jest-axe';

expect.extend(toHaveNoViolations);

test('should have no accessibility violations', async () => {
    const results = await axe(container);
    expect(results).toHaveNoViolations();
});
```
