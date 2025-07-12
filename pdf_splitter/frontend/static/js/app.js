// Global notification system for PDF Splitter
window.notify = {
    show(message, type = 'info', duration = 4000) {
        const container = document.getElementById('notifications');
        if (!container) {
            console.warn('Notifications container not found');
            return;
        }

        const id = 'notification-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);

        const colors = {
            info: 'bg-blue-500 border-blue-600',
            success: 'bg-green-500 border-green-600',
            warning: 'bg-yellow-500 border-yellow-600',
            error: 'bg-red-500 border-red-600'
        };

        const icons = {
            info: '<svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"/></svg>',
            success: '<svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/></svg>',
            warning: '<svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/></svg>',
            error: '<svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"/></svg>'
        };

        const notification = document.createElement('div');
        notification.id = id;
        notification.className = `${colors[type]} text-white px-4 py-3 rounded-lg shadow-lg border-l-4 transform transition-all duration-500 ease-out translate-x-full opacity-0 max-w-sm`;
        notification.innerHTML = `
            <div class="flex items-start">
                <div class="flex-shrink-0 mr-3">
                    ${icons[type]}
                </div>
                <div class="flex-1 min-w-0">
                    <p class="text-sm font-medium leading-5">${message}</p>
                </div>
                <div class="ml-3 flex-shrink-0">
                    <button onclick="notify.hide('${id}')" class="inline-flex text-white hover:text-gray-200 focus:outline-none focus:text-gray-200 transition-colors duration-200">
                        <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                            <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"/>
                        </svg>
                    </button>
                </div>
            </div>
        `;

        container.appendChild(notification);

        // Trigger enter animation
        requestAnimationFrame(() => {
            notification.classList.remove('translate-x-full', 'opacity-0');
            notification.classList.add('translate-x-0', 'opacity-100');
        });

        // Auto-hide after duration
        if (duration > 0) {
            setTimeout(() => this.hide(id), duration);
        }

        return id;
    },

    hide(id) {
        const notification = document.getElementById(id);
        if (!notification) return;

        notification.classList.remove('translate-x-0', 'opacity-100');
        notification.classList.add('translate-x-full', 'opacity-0');

        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 500);
    },

    clear() {
        const container = document.getElementById('notifications');
        if (container) {
            container.innerHTML = '';
        }
    }
};

// Global keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + S to save (in boundary editor)
    if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        const saveButton = document.querySelector('[data-action="save"]');
        if (saveButton && !saveButton.disabled) {
            saveButton.click();
        }
    }

    // Escape to close modals
    if (e.key === 'Escape') {
        const closeButton = document.querySelector('[data-action="close-modal"]');
        if (closeButton) {
            closeButton.click();
        }
    }

    // Delete key to delete selected documents
    if (e.key === 'Delete' || e.key === 'Backspace') {
        if (e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
            e.preventDefault();
            const deleteButton = document.querySelector('[data-action="delete-selected"]');
            if (deleteButton && !deleteButton.disabled) {
                deleteButton.click();
            }
        }
    }

    // Ctrl/Cmd + A to select all documents
    if ((e.ctrlKey || e.metaKey) && e.key === 'a') {
        if (e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
            e.preventDefault();
            const selectAllButton = document.querySelector('[data-action="select-all"]');
            if (selectAllButton) {
                selectAllButton.click();
            }
        }
    }

    // Ctrl/Cmd + M to merge selected documents
    if ((e.ctrlKey || e.metaKey) && e.key === 'm') {
        e.preventDefault();
        const mergeButton = document.querySelector('[data-action="merge-selected"]');
        if (mergeButton && !mergeButton.disabled) {
            mergeButton.click();
        }
    }

    // Ctrl/Cmd + N to add new document
    if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
        e.preventDefault();
        const addButton = document.querySelector('[data-action="add-document"]');
        if (addButton) {
            addButton.click();
        }
    }

    // Ctrl/Cmd + Enter to execute split
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        const splitButton = document.querySelector('[data-action="execute-split"]');
        if (splitButton && !splitButton.disabled) {
            splitButton.click();
        }
    }
});

// Global error handling for fetch requests
window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    if (event.reason && event.reason.message) {
        notify.show(`Error: ${event.reason.message}`, 'error');
    } else {
        notify.show('An unexpected error occurred', 'error');
    }
});

// Loading states for buttons
function addLoadingState(button, text = 'Loading...') {
    if (!button) return;

    button.disabled = true;
    button.dataset.originalText = button.textContent;
    button.innerHTML = `
        <svg class="animate-spin h-4 w-4 mr-2 inline" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        ${text}
    `;
}

function removeLoadingState(button) {
    if (!button) return;

    button.disabled = false;
    if (button.dataset.originalText) {
        button.textContent = button.dataset.originalText;
        delete button.dataset.originalText;
    }
}

// Utility function for confirming destructive actions
function confirmAction(message, actionText = 'Continue') {
    return new Promise((resolve) => {
        const isConfirmed = confirm(message);
        resolve(isConfirmed);
    });
}

// Enhanced HTMX error handling
document.addEventListener('htmx:responseError', (event) => {
    const response = event.detail.xhr;
    let message = 'An error occurred';

    try {
        const data = JSON.parse(response.responseText);
        message = data.detail || data.message || message;
    } catch (e) {
        message = `Error ${response.status}: ${response.statusText}`;
    }

    notify.show(message, 'error', 6000);
});

// HTMX loading states
document.addEventListener('htmx:beforeRequest', (event) => {
    const target = event.detail.target;
    if (target) {
        target.classList.add('opacity-75', 'pointer-events-none');
    }
});

document.addEventListener('htmx:afterRequest', (event) => {
    const target = event.detail.target;
    if (target) {
        target.classList.remove('opacity-75', 'pointer-events-none');
    }
});

// Auto-save functionality (for future use)
function createAutoSaver(saveFunction, delay = 3000) {
    let timeoutId;

    return function() {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => {
            saveFunction();
        }, delay);
    };
}

// Mobile-specific enhancements
if ('ontouchstart' in window) {
    // Add touch-friendly class to body
    document.body.classList.add('touch-device');

    // Prevent zoom on double-tap for better UX
    let lastTouchEnd = 0;
    document.addEventListener('touchend', (event) => {
        const now = (new Date()).getTime();
        if (now - lastTouchEnd <= 300) {
            event.preventDefault();
        }
        lastTouchEnd = now;
    }, false);
}

// Page visibility API for better resource management
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Page is hidden, pause non-critical operations
        console.log('Page hidden, reducing activity');
    } else {
        // Page is visible, resume operations
        console.log('Page visible, resuming activity');
    }
});

// Print functionality
window.printPage = function() {
    // Prepare page for printing
    document.body.classList.add('printing');

    // Trigger print dialog
    window.print();

    // Clean up after printing
    window.addEventListener('afterprint', () => {
        document.body.classList.remove('printing');
    }, { once: true });
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('PDF Splitter frontend initialized');

    // Add data attributes to buttons for keyboard shortcuts
    const buttons = {
        '[onclick*="saveBoundaryChanges"]': 'save',
        '[onclick*="closeBoundaryEditor"]': 'close-modal',
        '[onclick*="deleteSelectedDocuments"]': 'delete-selected',
        '[onclick*="mergeSelectedDocuments"]': 'merge-selected',
        '[onclick*="addDocumentBoundary"]': 'add-document',
        '[onclick*="executeSplit"]': 'execute-split'
    };

    Object.entries(buttons).forEach(([selector, action]) => {
        const button = document.querySelector(selector);
        if (button) {
            button.setAttribute('data-action', action);
        }
    });

    // Show keyboard shortcuts help on first visit
    if (!localStorage.getItem('keyboard-shortcuts-shown')) {
        setTimeout(() => {
            notify.show('Tip: Use Ctrl+N to add, Ctrl+M to merge, Del to delete documents', 'info', 8000);
            localStorage.setItem('keyboard-shortcuts-shown', 'true');
        }, 2000);
    }
});
