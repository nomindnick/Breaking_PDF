// Focus Trap Utility for Modal Dialogs
class FocusTrap {
    constructor(element, options = {}) {
        this.element = element;
        this.options = {
            initialFocus: options.initialFocus || null,
            fallbackFocus: options.fallbackFocus || element,
            escapeDeactivates: options.escapeDeactivates !== false,
            clickOutsideDeactivates: options.clickOutsideDeactivates !== false,
            returnFocusOnDeactivate: options.returnFocusOnDeactivate !== false,
            ...options
        };

        this.active = false;
        this.previouslyFocusedElement = null;
        this.firstFocusableElement = null;
        this.lastFocusableElement = null;

        // Bind methods
        this.handleKeyDown = this.handleKeyDown.bind(this);
        this.handleClickOutside = this.handleClickOutside.bind(this);
        this.updateFocusableElements = this.updateFocusableElements.bind(this);
    }

    activate() {
        if (this.active) return;

        this.active = true;
        this.previouslyFocusedElement = document.activeElement;

        // Update focusable elements
        this.updateFocusableElements();

        // Add event listeners
        document.addEventListener('keydown', this.handleKeyDown);
        if (this.options.clickOutsideDeactivates) {
            document.addEventListener('click', this.handleClickOutside, true);
        }

        // Set initial focus
        const initialFocusElement = this.options.initialFocus
            ? this.element.querySelector(this.options.initialFocus)
            : this.firstFocusableElement || this.options.fallbackFocus;

        if (initialFocusElement) {
            // Use requestAnimationFrame to ensure DOM is ready
            requestAnimationFrame(() => {
                initialFocusElement.focus();
            });
        }

        // Set up mutation observer to track DOM changes
        this.observer = new MutationObserver(this.updateFocusableElements);
        this.observer.observe(this.element, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['disabled', 'tabindex']
        });

        // Announce to screen readers
        this.announceModal();
    }

    deactivate() {
        if (!this.active) return;

        this.active = false;

        // Remove event listeners
        document.removeEventListener('keydown', this.handleKeyDown);
        document.removeEventListener('click', this.handleClickOutside, true);

        // Disconnect observer
        if (this.observer) {
            this.observer.disconnect();
        }

        // Return focus
        if (this.options.returnFocusOnDeactivate && this.previouslyFocusedElement) {
            this.previouslyFocusedElement.focus();
        }
    }

    updateFocusableElements() {
        const focusableSelectors = [
            'a[href]:not([disabled])',
            'button:not([disabled])',
            'textarea:not([disabled])',
            'input:not([disabled])',
            'select:not([disabled])',
            '[tabindex]:not([tabindex="-1"]):not([disabled])',
            'audio[controls]',
            'video[controls]',
            '[contenteditable]:not([contenteditable="false"])',
            'details>summary:first-of-type',
            'details'
        ];

        const focusableElements = this.element.querySelectorAll(focusableSelectors.join(','));
        const visibleFocusableElements = Array.from(focusableElements).filter(el => {
            return !this.isHidden(el) && !this.isInert(el);
        });

        this.firstFocusableElement = visibleFocusableElements[0];
        this.lastFocusableElement = visibleFocusableElements[visibleFocusableElements.length - 1];
    }

    isHidden(element) {
        if (element.offsetParent === null) return true;

        const style = window.getComputedStyle(element);
        return style.display === 'none' || style.visibility === 'hidden';
    }

    isInert(element) {
        let el = element;
        while (el) {
            if (el.hasAttribute('inert')) return true;
            el = el.parentElement;
        }
        return false;
    }

    handleKeyDown(event) {
        if (!this.active) return;

        // Handle Escape key
        if (this.options.escapeDeactivates && event.key === 'Escape') {
            event.preventDefault();
            if (this.options.onEscape) {
                this.options.onEscape();
            } else {
                this.deactivate();
            }
            return;
        }

        // Handle Tab key
        if (event.key === 'Tab') {
            if (!this.firstFocusableElement || !this.lastFocusableElement) {
                event.preventDefault();
                return;
            }

            if (event.shiftKey) {
                // Shift + Tab (backwards)
                if (document.activeElement === this.firstFocusableElement) {
                    event.preventDefault();
                    this.lastFocusableElement.focus();
                }
            } else {
                // Tab (forwards)
                if (document.activeElement === this.lastFocusableElement) {
                    event.preventDefault();
                    this.firstFocusableElement.focus();
                }
            }
        }
    }

    handleClickOutside(event) {
        if (!this.active) return;

        if (!this.element.contains(event.target)) {
            if (this.options.onClickOutside) {
                this.options.onClickOutside(event);
            } else {
                this.deactivate();
            }
        }
    }

    announceModal() {
        // Create a live region for screen reader announcements
        const announcement = document.createElement('div');
        announcement.setAttribute('role', 'status');
        announcement.setAttribute('aria-live', 'polite');
        announcement.className = 'sr-only';
        announcement.textContent = 'Dialog opened. Press Escape to close.';

        document.body.appendChild(announcement);

        // Remove after announcement
        setTimeout(() => {
            document.body.removeChild(announcement);
        }, 1000);
    }
}

// Alpine.js directive for easy focus trap usage
if (window.Alpine) {
    document.addEventListener('alpine:init', () => {
        Alpine.directive('trap', (el, { expression, modifiers }, { effect, cleanup }) => {
            let trap;

            effect(() => {
                const shouldTrap = expression ? Alpine.evaluate(el, expression) : true;

                if (shouldTrap && !trap) {
                    trap = new FocusTrap(el, {
                        escapeDeactivates: !modifiers.includes('noescape'),
                        clickOutsideDeactivates: !modifiers.includes('noclick'),
                        initialFocus: el.dataset.initialFocus,
                        onEscape: () => {
                            if (el._x_trapEscape) {
                                el._x_trapEscape();
                            }
                        },
                        onClickOutside: () => {
                            if (el._x_trapClickOutside) {
                                el._x_trapClickOutside();
                            }
                        }
                    });
                    trap.activate();
                } else if (!shouldTrap && trap) {
                    trap.deactivate();
                    trap = null;
                }
            });

            cleanup(() => {
                if (trap) {
                    trap.deactivate();
                }
            });
        });
    });
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FocusTrap;
}
