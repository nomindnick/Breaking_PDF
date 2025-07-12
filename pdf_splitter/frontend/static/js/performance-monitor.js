// Performance Monitoring and Optimization Utilities
class PerformanceMonitor {
    constructor() {
        this.metrics = {
            pageLoad: {},
            apiCalls: new Map(),
            userInteractions: [],
            resourceTimings: [],
            memoryUsage: []
        };

        this.observers = new Map();
        this.config = {
            enableLogging: true,
            sampleRate: 1.0, // 100% sampling
            slowThreshold: 1000, // 1 second
            memoryCheckInterval: 30000 // 30 seconds
        };

        // Initialize monitoring
        this.init();
    }

    init() {
        // Monitor page load performance
        this.monitorPageLoad();

        // Set up performance observer
        this.setupPerformanceObserver();

        // Monitor memory usage
        this.monitorMemory();

        // Monitor long tasks
        this.monitorLongTasks();

        // Set up API call monitoring
        this.monitorAPICalls();

        // Monitor user interactions
        this.monitorUserInteractions();

        // Set up reporting
        this.setupReporting();
    }

    monitorPageLoad() {
        if (window.performance && window.performance.timing) {
            window.addEventListener('load', () => {
                setTimeout(() => {
                    const timing = window.performance.timing;
                    const navigation = window.performance.navigation;

                    this.metrics.pageLoad = {
                        // Network timings
                        dns: timing.domainLookupEnd - timing.domainLookupStart,
                        tcp: timing.connectEnd - timing.connectStart,
                        request: timing.responseStart - timing.requestStart,
                        response: timing.responseEnd - timing.responseStart,

                        // Processing timings
                        domProcessing: timing.domComplete - timing.domLoading,
                        domContentLoaded: timing.domContentLoadedEventEnd - timing.navigationStart,
                        loadComplete: timing.loadEventEnd - timing.navigationStart,

                        // Core Web Vitals approximations
                        ttfb: timing.responseStart - timing.navigationStart, // Time to First Byte
                        fcp: this.getFirstContentfulPaint(),
                        lcp: this.getLargestContentfulPaint(),

                        // Navigation info
                        type: navigation.type,
                        redirectCount: navigation.redirectCount
                    };

                    this.logMetric('PageLoad', this.metrics.pageLoad);

                    // Check for slow page load
                    if (this.metrics.pageLoad.loadComplete > this.config.slowThreshold * 3) {
                        this.reportSlowLoad();
                    }
                }, 0);
            });
        }
    }

    setupPerformanceObserver() {
        if ('PerformanceObserver' in window) {
            // Observe resource timings
            const resourceObserver = new PerformanceObserver((list) => {
                for (const entry of list.getEntries()) {
                    this.handleResourceTiming(entry);
                }
            });

            try {
                resourceObserver.observe({ entryTypes: ['resource'] });
                this.observers.set('resource', resourceObserver);
            } catch (e) {
                console.warn('Resource timing not supported');
            }

            // Observe paint timings
            const paintObserver = new PerformanceObserver((list) => {
                for (const entry of list.getEntries()) {
                    this.metrics[entry.name] = Math.round(entry.startTime);
                }
            });

            try {
                paintObserver.observe({ entryTypes: ['paint'] });
                this.observers.set('paint', paintObserver);
            } catch (e) {
                console.warn('Paint timing not supported');
            }
        }
    }

    monitorLongTasks() {
        if ('PerformanceObserver' in window && PerformanceObserver.supportedEntryTypes.includes('longtask')) {
            const longTaskObserver = new PerformanceObserver((list) => {
                for (const entry of list.getEntries()) {
                    this.logMetric('LongTask', {
                        duration: entry.duration,
                        startTime: entry.startTime,
                        attribution: entry.attribution
                    });

                    // Warn about long tasks
                    if (entry.duration > 100) {
                        console.warn(`Long task detected: ${entry.duration}ms`);
                    }
                }
            });

            try {
                longTaskObserver.observe({ entryTypes: ['longtask'] });
                this.observers.set('longtask', longTaskObserver);
            } catch (e) {
                console.warn('Long task monitoring not supported');
            }
        }
    }

    monitorAPICalls() {
        const originalFetch = window.fetch;

        window.fetch = async (...args) => {
            const startTime = performance.now();
            const [url, options = {}] = args;
            const method = options.method || 'GET';
            const callId = `${method}:${url}:${Date.now()}`;

            // Track API call start
            this.metrics.apiCalls.set(callId, {
                url,
                method,
                startTime,
                status: 'pending'
            });

            try {
                const response = await originalFetch(...args);
                const endTime = performance.now();
                const duration = endTime - startTime;

                // Update metrics
                const callMetrics = this.metrics.apiCalls.get(callId);
                callMetrics.endTime = endTime;
                callMetrics.duration = duration;
                callMetrics.status = response.status;
                callMetrics.ok = response.ok;
                callMetrics.size = response.headers.get('content-length');

                this.logMetric('APICall', callMetrics);

                // Check for slow API calls
                if (duration > this.config.slowThreshold) {
                    this.reportSlowAPI(callMetrics);
                }

                // Clean up old entries
                if (this.metrics.apiCalls.size > 100) {
                    const oldestKey = this.metrics.apiCalls.keys().next().value;
                    this.metrics.apiCalls.delete(oldestKey);
                }

                return response;

            } catch (error) {
                const endTime = performance.now();
                const callMetrics = this.metrics.apiCalls.get(callId);
                callMetrics.endTime = endTime;
                callMetrics.duration = endTime - startTime;
                callMetrics.status = 'error';
                callMetrics.error = error.message;

                this.logMetric('APIError', callMetrics);
                throw error;
            }
        };
    }

    monitorUserInteractions() {
        const interactionEvents = ['click', 'input', 'change', 'submit'];

        interactionEvents.forEach(eventType => {
            document.addEventListener(eventType, (event) => {
                if (Math.random() > this.config.sampleRate) return;

                const target = event.target;
                const interaction = {
                    type: eventType,
                    timestamp: performance.now(),
                    target: {
                        tagName: target.tagName,
                        id: target.id,
                        className: target.className,
                        text: target.textContent?.substring(0, 50)
                    },
                    path: this.getElementPath(target)
                };

                this.metrics.userInteractions.push(interaction);

                // Keep only recent interactions
                if (this.metrics.userInteractions.length > 50) {
                    this.metrics.userInteractions.shift();
                }

                // Measure interaction response time
                this.measureInteractionResponse(interaction);
            }, { passive: true, capture: true });
        });
    }

    measureInteractionResponse(interaction) {
        const startTime = performance.now();

        // Use requestAnimationFrame to measure when browser is ready
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                const responseTime = performance.now() - startTime;

                if (responseTime > 100) {
                    this.logMetric('SlowInteraction', {
                        ...interaction,
                        responseTime
                    });
                }
            });
        });
    }

    monitorMemory() {
        if (performance.memory) {
            const checkMemory = () => {
                const memoryInfo = {
                    timestamp: Date.now(),
                    usedJSHeapSize: Math.round(performance.memory.usedJSHeapSize / 1048576), // MB
                    totalJSHeapSize: Math.round(performance.memory.totalJSHeapSize / 1048576), // MB
                    jsHeapSizeLimit: Math.round(performance.memory.jsHeapSizeLimit / 1048576) // MB
                };

                this.metrics.memoryUsage.push(memoryInfo);

                // Keep only last 10 measurements
                if (this.metrics.memoryUsage.length > 10) {
                    this.metrics.memoryUsage.shift();
                }

                // Check for memory issues
                const usagePercent = (memoryInfo.usedJSHeapSize / memoryInfo.jsHeapSizeLimit) * 100;
                if (usagePercent > 90) {
                    this.reportMemoryIssue(memoryInfo);
                }
            };

            setInterval(checkMemory, this.config.memoryCheckInterval);
            checkMemory(); // Initial check
        }
    }

    handleResourceTiming(entry) {
        const resource = {
            name: entry.name,
            type: entry.initiatorType,
            duration: Math.round(entry.duration),
            size: entry.transferSize || 0,
            cached: entry.transferSize === 0 && entry.decodedBodySize > 0
        };

        // Only track significant resources
        if (resource.duration > 50 || resource.size > 50000) {
            this.metrics.resourceTimings.push(resource);

            // Keep only recent resources
            if (this.metrics.resourceTimings.length > 50) {
                this.metrics.resourceTimings.shift();
            }

            // Check for slow resources
            if (resource.duration > this.config.slowThreshold) {
                this.reportSlowResource(resource);
            }
        }
    }

    // Utility methods
    getFirstContentfulPaint() {
        const entries = performance.getEntriesByType('paint');
        const fcp = entries.find(entry => entry.name === 'first-contentful-paint');
        return fcp ? Math.round(fcp.startTime) : null;
    }

    getLargestContentfulPaint() {
        // This would need the LCP API which might not be available
        return new Promise(resolve => {
            if ('PerformanceObserver' in window) {
                const observer = new PerformanceObserver((list) => {
                    const entries = list.getEntries();
                    const lastEntry = entries[entries.length - 1];
                    resolve(Math.round(lastEntry.startTime));
                    observer.disconnect();
                });

                try {
                    observer.observe({ entryTypes: ['largest-contentful-paint'] });
                } catch (e) {
                    resolve(null);
                }
            } else {
                resolve(null);
            }
        });
    }

    getElementPath(element) {
        const path = [];
        let current = element;

        while (current && current !== document.body) {
            let selector = current.tagName.toLowerCase();
            if (current.id) {
                selector += `#${current.id}`;
            } else if (current.className) {
                selector += `.${current.className.split(' ')[0]}`;
            }
            path.unshift(selector);
            current = current.parentElement;
        }

        return path.join(' > ');
    }

    // Optimization utilities
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    throttle(func, limit) {
        let inThrottle;
        return function(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }

    // Lazy loading utility
    lazyLoad(selector, options = {}) {
        const elements = document.querySelectorAll(selector);

        const imageObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;

                    if (img.dataset.src) {
                        img.src = img.dataset.src;
                        img.removeAttribute('data-src');
                    }

                    if (img.dataset.srcset) {
                        img.srcset = img.dataset.srcset;
                        img.removeAttribute('data-srcset');
                    }

                    img.classList.add('lazy-loaded');
                    observer.unobserve(img);
                }
            });
        }, {
            rootMargin: options.rootMargin || '50px 0px',
            threshold: options.threshold || 0.01
        });

        elements.forEach(img => imageObserver.observe(img));

        return imageObserver;
    }

    // Request idle callback polyfill
    requestIdleCallback(callback, options) {
        if ('requestIdleCallback' in window) {
            return window.requestIdleCallback(callback, options);
        }

        // Fallback for browsers that don't support it
        const start = Date.now();
        return setTimeout(() => {
            callback({
                didTimeout: false,
                timeRemaining: () => Math.max(0, 50 - (Date.now() - start))
            });
        }, 1);
    }

    // Reporting methods
    logMetric(type, data) {
        if (!this.config.enableLogging) return;

        const metric = {
            type,
            timestamp: Date.now(),
            data
        };

        // Console logging in development
        if (window.location.hostname === 'localhost') {
            console.log(`[Performance] ${type}:`, data);
        }

        // Send to analytics if configured
        if (window.analytics) {
            window.analytics.track('Performance Metric', metric);
        }
    }

    reportSlowLoad() {
        const report = {
            loadTime: this.metrics.pageLoad.loadComplete,
            metrics: this.metrics.pageLoad,
            url: window.location.href,
            userAgent: navigator.userAgent
        };

        console.warn('Slow page load detected:', report);

        // Show user notification if very slow
        if (this.metrics.pageLoad.loadComplete > 5000) {
            window.notify?.show('This page took longer than usual to load', 'warning');
        }
    }

    reportSlowAPI(metrics) {
        console.warn(`Slow API call: ${metrics.method} ${metrics.url} took ${metrics.duration}ms`);
    }

    reportSlowResource(resource) {
        console.warn(`Slow resource: ${resource.name} took ${resource.duration}ms`);
    }

    reportMemoryIssue(memoryInfo) {
        console.warn('High memory usage detected:', memoryInfo);
        window.notify?.show('High memory usage detected. Consider refreshing the page.', 'warning');
    }

    setupReporting() {
        // Send performance report before page unload
        window.addEventListener('beforeunload', () => {
            const report = this.generateReport();

            // Use sendBeacon for reliable delivery
            if (navigator.sendBeacon && window.location.hostname !== 'localhost') {
                const blob = new Blob([JSON.stringify(report)], { type: 'application/json' });
                navigator.sendBeacon('/api/analytics/performance', blob);
            }
        });

        // Periodic reporting for long sessions
        setInterval(() => {
            if (document.visibilityState === 'visible') {
                this.generateReport();
            }
        }, 300000); // Every 5 minutes
    }

    generateReport() {
        const report = {
            timestamp: Date.now(),
            sessionDuration: Date.now() - (window.performance.timing.navigationStart || Date.now()),
            url: window.location.href,
            metrics: {
                pageLoad: this.metrics.pageLoad,
                apiCallSummary: this.getAPICallSummary(),
                memoryUsage: this.metrics.memoryUsage[this.metrics.memoryUsage.length - 1],
                interactionCount: this.metrics.userInteractions.length,
                resourceCount: this.metrics.resourceTimings.length
            }
        };

        return report;
    }

    getAPICallSummary() {
        const calls = Array.from(this.metrics.apiCalls.values());

        if (calls.length === 0) return null;

        const durations = calls.map(c => c.duration || 0).filter(d => d > 0);
        const errors = calls.filter(c => c.status === 'error').length;

        return {
            total: calls.length,
            errors,
            avgDuration: durations.length ? Math.round(durations.reduce((a, b) => a + b, 0) / durations.length) : 0,
            maxDuration: durations.length ? Math.max(...durations) : 0,
            slowCalls: calls.filter(c => c.duration > this.config.slowThreshold).length
        };
    }

    // Public API
    getMetrics() {
        return {
            ...this.metrics,
            report: this.generateReport()
        };
    }

    reset() {
        this.metrics.apiCalls.clear();
        this.metrics.userInteractions = [];
        this.metrics.resourceTimings = [];
        this.metrics.memoryUsage = [];
    }

    destroy() {
        this.observers.forEach(observer => observer.disconnect());
        this.observers.clear();
    }
}

// Initialize performance monitor
window.performanceMonitor = new PerformanceMonitor();

// Export optimization utilities for easy use
window.perfUtils = {
    debounce: window.performanceMonitor.debounce,
    throttle: window.performanceMonitor.throttle,
    lazyLoad: window.performanceMonitor.lazyLoad.bind(window.performanceMonitor),
    requestIdleCallback: window.performanceMonitor.requestIdleCallback
};

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PerformanceMonitor;
}
