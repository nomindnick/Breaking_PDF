// Help System with Tooltips, Keyboard Shortcuts, and Onboarding
class HelpSystem {
    constructor() {
        this.tooltips = new Map();
        this.onboardingSteps = [];
        this.currentOnboardingStep = 0;
        this.onboardingActive = false;

        // Keyboard shortcuts registry
        this.shortcuts = [
            { keys: 'Ctrl+N', description: 'Add new document boundary', context: 'review' },
            { keys: 'Ctrl+M', description: 'Merge selected documents', context: 'review' },
            { keys: 'Delete', description: 'Delete selected documents', context: 'review' },
            { keys: 'Ctrl+S', description: 'Save changes', context: 'editor' },
            { keys: 'Ctrl+Enter', description: 'Execute split', context: 'review' },
            { keys: 'Escape', description: 'Close modal/dialog', context: 'global' },
            { keys: 'Ctrl+A', description: 'Select all documents', context: 'review' },
            { keys: '?', description: 'Show help', context: 'global' },
            { keys: 'Tab', description: 'Navigate forward', context: 'global' },
            { keys: 'Shift+Tab', description: 'Navigate backward', context: 'global' },
            { keys: 'Enter', description: 'Activate focused element', context: 'global' },
            { keys: 'Space', description: 'Toggle checkbox/button', context: 'global' }
        ];

        // Initialize
        this.init();
    }

    init() {
        // Set up global keyboard shortcuts
        this.setupKeyboardShortcuts();

        // Set up tooltip system
        this.setupTooltips();

        // Check if user needs onboarding
        this.checkOnboarding();

        // Create help modal
        this.createHelpModal();
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Show help on '?'
            if (e.key === '?' && !e.ctrlKey && !e.metaKey && !e.altKey) {
                if (!this.isInputFocused()) {
                    e.preventDefault();
                    this.showHelp();
                }
            }

            // Show context-sensitive help on F1
            if (e.key === 'F1') {
                e.preventDefault();
                this.showContextHelp();
            }
        });
    }

    setupTooltips() {
        // Create tooltip container
        const tooltipContainer = document.createElement('div');
        tooltipContainer.id = 'tooltip-container';
        tooltipContainer.className = 'fixed z-50 pointer-events-none';
        tooltipContainer.innerHTML = `
            <div id="tooltip"
                 class="hidden bg-gray-900 text-white text-sm px-3 py-2 rounded-lg shadow-lg max-w-xs"
                 role="tooltip">
                <div id="tooltip-content"></div>
                <div class="absolute w-2 h-2 bg-gray-900 transform rotate-45"
                     id="tooltip-arrow"></div>
            </div>
        `;
        document.body.appendChild(tooltipContainer);

        // Set up tooltip triggers
        this.initializeTooltips();

        // Monitor DOM changes for new elements
        const observer = new MutationObserver(() => {
            this.initializeTooltips();
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }

    initializeTooltips() {
        // Find all elements with data-tooltip attribute
        const elements = document.querySelectorAll('[data-tooltip]');

        elements.forEach(element => {
            if (!this.tooltips.has(element)) {
                this.addTooltip(element);
            }
        });
    }

    addTooltip(element) {
        const tooltip = document.getElementById('tooltip');
        const tooltipContent = document.getElementById('tooltip-content');
        const tooltipArrow = document.getElementById('tooltip-arrow');

        let showTimeout;
        let hideTimeout;

        const showTooltip = () => {
            clearTimeout(hideTimeout);

            showTimeout = setTimeout(() => {
                // Set content
                const content = element.getAttribute('data-tooltip');
                const html = element.getAttribute('data-tooltip-html');

                if (html) {
                    tooltipContent.innerHTML = html;
                } else {
                    tooltipContent.textContent = content;
                }

                // Show tooltip
                tooltip.classList.remove('hidden');

                // Position tooltip
                this.positionTooltip(element, tooltip, tooltipArrow);

                // Set ARIA
                element.setAttribute('aria-describedby', 'tooltip');
            }, 500); // Delay before showing
        };

        const hideTooltip = () => {
            clearTimeout(showTimeout);

            hideTimeout = setTimeout(() => {
                tooltip.classList.add('hidden');
                element.removeAttribute('aria-describedby');
            }, 100);
        };

        // Mouse events
        element.addEventListener('mouseenter', showTooltip);
        element.addEventListener('mouseleave', hideTooltip);

        // Focus events for keyboard users
        element.addEventListener('focus', showTooltip);
        element.addEventListener('blur', hideTooltip);

        // Touch events
        let touchTimeout;
        element.addEventListener('touchstart', () => {
            touchTimeout = setTimeout(showTooltip, 500);
        });

        element.addEventListener('touchend', () => {
            clearTimeout(touchTimeout);
            hideTooltip();
        });

        this.tooltips.set(element, { showTooltip, hideTooltip });
    }

    positionTooltip(element, tooltip, arrow) {
        const rect = element.getBoundingClientRect();
        const tooltipRect = tooltip.getBoundingClientRect();

        // Default position is above
        let top = rect.top - tooltipRect.height - 10;
        let left = rect.left + (rect.width - tooltipRect.width) / 2;

        // Adjust if tooltip would go off screen
        if (top < 10) {
            // Position below
            top = rect.bottom + 10;
            arrow.style.top = '-4px';
            arrow.style.bottom = 'auto';
        } else {
            arrow.style.bottom = '-4px';
            arrow.style.top = 'auto';
        }

        if (left < 10) {
            left = 10;
        } else if (left + tooltipRect.width > window.innerWidth - 10) {
            left = window.innerWidth - tooltipRect.width - 10;
        }

        // Center arrow
        const arrowLeft = rect.left + rect.width / 2 - left - 4;
        arrow.style.left = `${arrowLeft}px`;

        // Apply position
        tooltip.style.top = `${top}px`;
        tooltip.style.left = `${left}px`;
    }

    createHelpModal() {
        const modal = document.createElement('div');
        modal.id = 'help-modal';
        modal.innerHTML = `
            <div x-data="{ showHelp: false, activeTab: 'shortcuts' }"
                 x-show="showHelp"
                 @help-modal.window="showHelp = true"
                 class="fixed inset-0 bg-black bg-opacity-50 z-50 overflow-y-auto"
                 x-transition
                 @click.self="showHelp = false">

                <div class="min-h-screen px-4 text-center">
                    <span class="inline-block h-screen align-middle" aria-hidden="true">&#8203;</span>

                    <div class="inline-block w-full max-w-2xl my-8 overflow-hidden text-left align-middle bg-white shadow-xl rounded-2xl"
                         x-trap="showHelp"
                         @click.stop
                         role="dialog"
                         aria-modal="true"
                         aria-labelledby="help-modal-title">

                        <!-- Header -->
                        <div class="bg-gray-50 px-6 py-4 border-b">
                            <div class="flex items-center justify-between">
                                <h2 id="help-modal-title" class="text-xl font-semibold text-gray-900">Help & Keyboard Shortcuts</h2>
                                <button @click="showHelp = false"
                                        class="text-gray-400 hover:text-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500 rounded"
                                        aria-label="Close help">
                                    <svg class="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
                                    </svg>
                                </button>
                            </div>
                        </div>

                        <!-- Tabs -->
                        <div class="border-b">
                            <nav class="flex -mb-px" role="tablist">
                                <button @click="activeTab = 'shortcuts'"
                                        :class="activeTab === 'shortcuts' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'"
                                        class="px-6 py-3 border-b-2 font-medium text-sm focus:outline-none focus:ring-2 focus:ring-inset focus:ring-blue-500"
                                        role="tab"
                                        :aria-selected="activeTab === 'shortcuts'">
                                    Keyboard Shortcuts
                                </button>
                                <button @click="activeTab = 'features'"
                                        :class="activeTab === 'features' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'"
                                        class="px-6 py-3 border-b-2 font-medium text-sm focus:outline-none focus:ring-2 focus:ring-inset focus:ring-blue-500"
                                        role="tab"
                                        :aria-selected="activeTab === 'features'">
                                    Features Guide
                                </button>
                                <button @click="activeTab = 'tips'"
                                        :class="activeTab === 'tips' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'"
                                        class="px-6 py-3 border-b-2 font-medium text-sm focus:outline-none focus:ring-2 focus:ring-inset focus:ring-blue-500"
                                        role="tab"
                                        :aria-selected="activeTab === 'tips'">
                                    Tips & Tricks
                                </button>
                            </nav>
                        </div>

                        <!-- Content -->
                        <div class="px-6 py-4 max-h-96 overflow-y-auto">
                            <!-- Keyboard Shortcuts -->
                            <div x-show="activeTab === 'shortcuts'" role="tabpanel">
                                <div class="space-y-4">
                                    <div>
                                        <h3 class="font-semibold text-gray-900 mb-2">Global Shortcuts</h3>
                                        <div class="space-y-2" id="global-shortcuts"></div>
                                    </div>

                                    <div>
                                        <h3 class="font-semibold text-gray-900 mb-2">Document Review</h3>
                                        <div class="space-y-2" id="review-shortcuts"></div>
                                    </div>

                                    <div>
                                        <h3 class="font-semibold text-gray-900 mb-2">Editor</h3>
                                        <div class="space-y-2" id="editor-shortcuts"></div>
                                    </div>
                                </div>
                            </div>

                            <!-- Features Guide -->
                            <div x-show="activeTab === 'features'" role="tabpanel">
                                <div class="prose prose-sm max-w-none">
                                    <h3>Uploading PDFs</h3>
                                    <p>Drag and drop your PDF file or click to browse. The system automatically detects document boundaries within your PDF.</p>

                                    <h3>Reviewing Documents</h3>
                                    <p>After processing, review the detected document boundaries. You can:</p>
                                    <ul>
                                        <li>Add new boundaries where documents were missed</li>
                                        <li>Adjust existing boundaries by editing page ranges</li>
                                        <li>Merge multiple segments into one document</li>
                                        <li>Delete incorrect boundaries</li>
                                        <li>Change document types and metadata</li>
                                    </ul>

                                    <h3>Splitting PDFs</h3>
                                    <p>Once satisfied with the boundaries, click "Split PDF" to create individual files. Each document will be named intelligently based on its content.</p>
                                </div>
                            </div>

                            <!-- Tips & Tricks -->
                            <div x-show="activeTab === 'tips'" role="tabpanel">
                                <div class="space-y-4 text-sm">
                                    <div class="bg-blue-50 p-4 rounded-lg">
                                        <h4 class="font-semibold text-blue-900 mb-1">💡 Quick Selection</h4>
                                        <p class="text-blue-700">Use Ctrl+Click to select multiple documents for batch operations.</p>
                                    </div>

                                    <div class="bg-green-50 p-4 rounded-lg">
                                        <h4 class="font-semibold text-green-900 mb-1">🚀 Keyboard Navigation</h4>
                                        <p class="text-green-700">Navigate entirely with keyboard using Tab, Arrow keys, and shortcuts.</p>
                                    </div>

                                    <div class="bg-yellow-50 p-4 rounded-lg">
                                        <h4 class="font-semibold text-yellow-900 mb-1">⚡ Performance Tip</h4>
                                        <p class="text-yellow-700">For large PDFs, the system processes pages in parallel for faster results.</p>
                                    </div>

                                    <div class="bg-purple-50 p-4 rounded-lg">
                                        <h4 class="font-semibold text-purple-900 mb-1">🎯 Accuracy Tip</h4>
                                        <p class="text-purple-700">The AI detection works best with clear document transitions like title pages or headers.</p>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Footer -->
                        <div class="bg-gray-50 px-6 py-3 border-t">
                            <div class="flex items-center justify-between text-sm text-gray-600">
                                <span>Press <kbd class="px-2 py-1 bg-gray-200 rounded text-xs">?</kbd> anytime to show this help</span>
                                <button @click="$dispatch('start-onboarding')"
                                        class="text-blue-600 hover:text-blue-800 font-medium">
                                    Start Tutorial
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Populate shortcuts
        this.populateShortcuts();
    }

    populateShortcuts() {
        const contexts = {
            global: document.getElementById('global-shortcuts'),
            review: document.getElementById('review-shortcuts'),
            editor: document.getElementById('editor-shortcuts')
        };

        this.shortcuts.forEach(shortcut => {
            const container = contexts[shortcut.context];
            if (container) {
                const item = document.createElement('div');
                item.className = 'flex items-center justify-between';
                item.innerHTML = `
                    <span class="text-gray-600">${shortcut.description}</span>
                    <kbd class="px-2 py-1 bg-gray-100 border border-gray-300 rounded text-xs font-mono">
                        ${shortcut.keys}
                    </kbd>
                `;
                container.appendChild(item);
            }
        });
    }

    showHelp() {
        window.dispatchEvent(new CustomEvent('help-modal'));
    }

    showContextHelp() {
        // Determine current context and show relevant help
        const path = window.location.pathname;

        if (path.includes('/upload')) {
            window.notify.show('Drag and drop a PDF or click to browse. Max size: 500MB', 'info', 5000);
        } else if (path.includes('/review')) {
            window.notify.show('Review detected boundaries. Use Ctrl+N to add, Ctrl+M to merge, Del to delete.', 'info', 5000);
        } else if (path.includes('/results')) {
            window.notify.show('Download individual files or all as ZIP. Results expire in 24 hours.', 'info', 5000);
        } else {
            this.showHelp();
        }
    }

    checkOnboarding() {
        const hasSeenOnboarding = localStorage.getItem('pdf_splitter_onboarding_complete');
        const isFirstVisit = !localStorage.getItem('pdf_splitter_visited');

        if (isFirstVisit) {
            localStorage.setItem('pdf_splitter_visited', 'true');
        }

        // Listen for onboarding trigger
        window.addEventListener('start-onboarding', () => {
            this.startOnboarding();
        });

        // Auto-start for first-time users on home page
        if (!hasSeenOnboarding && window.location.pathname === '/') {
            setTimeout(() => {
                this.offerOnboarding();
            }, 1000);
        }
    }

    offerOnboarding() {
        const offer = document.createElement('div');
        offer.className = 'fixed bottom-20 right-4 bg-blue-600 text-white p-4 rounded-lg shadow-lg z-40 max-w-sm';
        offer.innerHTML = `
            <div class="flex items-start">
                <div class="flex-1">
                    <h4 class="font-semibold mb-1">Welcome to PDF Splitter! 👋</h4>
                    <p class="text-sm opacity-90">Would you like a quick tour of the features?</p>
                </div>
                <button onclick="this.parentElement.parentElement.remove()"
                        class="ml-2 text-white opacity-70 hover:opacity-100">
                    <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"/>
                    </svg>
                </button>
            </div>
            <div class="mt-3 flex space-x-2">
                <button onclick="window.helpSystem.startOnboarding(); this.parentElement.parentElement.remove();"
                        class="px-3 py-1 bg-white text-blue-600 rounded text-sm font-medium hover:bg-blue-50">
                    Start Tour
                </button>
                <button onclick="localStorage.setItem('pdf_splitter_onboarding_complete', 'true'); this.parentElement.parentElement.remove();"
                        class="px-3 py-1 text-white opacity-90 hover:opacity-100 text-sm">
                    Skip
                </button>
            </div>
        `;

        document.body.appendChild(offer);

        // Auto-remove after 10 seconds
        setTimeout(() => {
            if (offer.parentElement) {
                offer.remove();
            }
        }, 10000);
    }

    startOnboarding() {
        // Define onboarding steps based on current page
        const path = window.location.pathname;

        if (path === '/') {
            this.onboardingSteps = [
                {
                    element: 'a[href="/upload"]',
                    title: 'Start Here',
                    content: 'Click here to upload a PDF file for splitting.',
                    position: 'bottom'
                },
                {
                    element: 'a[href="/history"]',
                    title: 'Processing History',
                    content: 'View all your previous PDF splitting sessions here.',
                    position: 'bottom'
                },
                {
                    element: '.feature-card',
                    title: 'Smart Detection',
                    content: 'Our AI automatically detects document boundaries within your PDF.',
                    position: 'top'
                }
            ];
        } else if (path === '/upload') {
            this.onboardingSteps = [
                {
                    element: '[x-data="fileUpload()"]',
                    title: 'Upload Your PDF',
                    content: 'Drag and drop your file here or click to browse.',
                    position: 'center'
                }
            ];
        }

        if (this.onboardingSteps.length > 0) {
            this.currentOnboardingStep = 0;
            this.onboardingActive = true;
            this.showOnboardingStep();
        }
    }

    showOnboardingStep() {
        if (!this.onboardingActive || this.currentOnboardingStep >= this.onboardingSteps.length) {
            this.endOnboarding();
            return;
        }

        const step = this.onboardingSteps[this.currentOnboardingStep];
        const element = document.querySelector(step.element);

        if (!element) {
            this.nextOnboardingStep();
            return;
        }

        // Create overlay
        this.createOnboardingOverlay(element, step);
    }

    createOnboardingOverlay(element, step) {
        // Remove existing overlay
        const existing = document.getElementById('onboarding-overlay');
        if (existing) existing.remove();

        const overlay = document.createElement('div');
        overlay.id = 'onboarding-overlay';
        overlay.className = 'fixed inset-0 z-50';
        overlay.innerHTML = `
            <div class="absolute inset-0 bg-black bg-opacity-50"></div>
            <div class="spotlight"></div>
            <div class="tooltip-onboarding">
                <h4 class="font-semibold text-lg mb-1">${step.title}</h4>
                <p class="text-sm mb-3">${step.content}</p>
                <div class="flex items-center justify-between">
                    <span class="text-xs opacity-70">
                        Step ${this.currentOnboardingStep + 1} of ${this.onboardingSteps.length}
                    </span>
                    <div class="space-x-2">
                        <button onclick="window.helpSystem.endOnboarding()"
                                class="text-xs opacity-70 hover:opacity-100">
                            Skip Tour
                        </button>
                        <button onclick="window.helpSystem.nextOnboardingStep()"
                                class="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700">
                            ${this.currentOnboardingStep < this.onboardingSteps.length - 1 ? 'Next' : 'Finish'}
                        </button>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(overlay);

        // Position spotlight and tooltip
        this.positionOnboardingElements(element, step.position);
    }

    positionOnboardingElements(element, position) {
        const rect = element.getBoundingClientRect();
        const spotlight = document.querySelector('.spotlight');
        const tooltip = document.querySelector('.tooltip-onboarding');

        // Position spotlight
        spotlight.style.position = 'absolute';
        spotlight.style.top = `${rect.top - 10}px`;
        spotlight.style.left = `${rect.left - 10}px`;
        spotlight.style.width = `${rect.width + 20}px`;
        spotlight.style.height = `${rect.height + 20}px`;
        spotlight.style.border = '2px solid #3B82F6';
        spotlight.style.borderRadius = '8px';
        spotlight.style.boxShadow = '0 0 0 9999px rgba(0, 0, 0, 0.5)';
        spotlight.style.pointerEvents = 'none';

        // Position tooltip
        tooltip.style.position = 'absolute';
        tooltip.style.background = 'white';
        tooltip.style.padding = '1rem';
        tooltip.style.borderRadius = '8px';
        tooltip.style.boxShadow = '0 10px 25px rgba(0, 0, 0, 0.2)';
        tooltip.style.maxWidth = '300px';
        tooltip.style.zIndex = '51';

        if (position === 'bottom') {
            tooltip.style.top = `${rect.bottom + 20}px`;
            tooltip.style.left = `${rect.left}px`;
        } else if (position === 'top') {
            tooltip.style.bottom = `${window.innerHeight - rect.top + 20}px`;
            tooltip.style.left = `${rect.left}px`;
        } else {
            tooltip.style.top = '50%';
            tooltip.style.left = '50%';
            tooltip.style.transform = 'translate(-50%, -50%)';
        }
    }

    nextOnboardingStep() {
        this.currentOnboardingStep++;
        this.showOnboardingStep();
    }

    endOnboarding() {
        this.onboardingActive = false;
        const overlay = document.getElementById('onboarding-overlay');
        if (overlay) overlay.remove();

        localStorage.setItem('pdf_splitter_onboarding_complete', 'true');

        if (this.currentOnboardingStep >= this.onboardingSteps.length - 1) {
            window.notify.show('Tour complete! Press ? anytime for help.', 'success');
        }
    }

    isInputFocused() {
        const activeElement = document.activeElement;
        return activeElement && (
            activeElement.tagName === 'INPUT' ||
            activeElement.tagName === 'TEXTAREA' ||
            activeElement.contentEditable === 'true'
        );
    }
}

// Initialize help system
window.helpSystem = new HelpSystem();

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = HelpSystem;
}
