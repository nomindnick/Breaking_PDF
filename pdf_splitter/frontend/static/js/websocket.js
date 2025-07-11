// WebSocket manager for real-time progress updates
class ProgressWebSocket {
    constructor(sessionId) {
        this.sessionId = sessionId;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.listeners = {};
        this.connectionState = 'disconnected';
        this.pingInterval = null;
        this.clientId = null;
        this.messageQueue = [];
    }

    connect() {
        if (this.connectionState === 'connecting' || this.connectionState === 'connected') {
            console.log('WebSocket already connected or connecting');
            return;
        }

        this.connectionState = 'connecting';
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/${this.sessionId}`;

        console.log(`Connecting to WebSocket: ${wsUrl}`);
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.connectionState = 'connected';
            this.reconnectAttempts = 0;
            this.emit('connected');

            // Send subscribe message
            this.sendMessage({
                type: 'subscribe',
                session_id: this.sessionId,
                include_previews: true
            });

            // Start ping interval
            this.startPing();

            // Process any queued messages
            this.processMessageQueue();
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleMessage(data);
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error);
            }
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.connectionState = 'error';
            this.emit('error', error);
        };

        this.ws.onclose = (event) => {
            console.log('WebSocket disconnected:', event.code, event.reason);
            this.connectionState = 'disconnected';
            this.stopPing();
            this.emit('disconnected', { code: event.code, reason: event.reason });

            // Attempt reconnection if not a normal closure
            if (event.code !== 1000 && event.code !== 1001) {
                this.attemptReconnect();
            }
        };
    }

    handleMessage(data) {
        console.log('Received message:', data.type, data);

        switch(data.type) {
            case 'connected':
                this.clientId = data.client_id;
                this.emit('connection_confirmed', data);
                break;

            case 'progress':
                this.emit('progress', {
                    stage: data.stage,
                    progress: data.progress,
                    message: data.message,
                    current_item: data.current_item,
                    total_items: data.total_items,
                    details: data.details
                });
                break;

            case 'stage_complete':
                this.emit('stage_complete', {
                    stage: data.stage,
                    success: data.success,
                    message: data.message,
                    duration: data.duration_seconds,
                    next_stage: data.next_stage,
                    results: data.results
                });
                break;

            case 'error':
                this.emit('error', {
                    code: data.error_code,
                    message: data.message,
                    details: data.details,
                    recoverable: data.recoverable
                });
                break;

            case 'preview_ready':
                this.emit('preview_ready', data);
                break;

            case 'split_complete':
                this.emit('split_complete', {
                    split_id: data.split_id,
                    success: data.success,
                    files_created: data.files_created,
                    download_url: data.download_url,
                    total_size: data.total_size_bytes,
                    duration: data.duration_seconds
                });
                break;

            case 'session_update':
                this.emit('session_update', data);
                break;

            case 'pong':
                // Pong received, connection is alive
                break;

            default:
                console.warn('Unknown message type:', data.type);
        }
    }

    sendMessage(message) {
        if (this.connectionState === 'connected' && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        } else {
            // Queue message for later
            this.messageQueue.push(message);
            console.log('WebSocket not ready, message queued:', message.type);
        }
    }

    processMessageQueue() {
        while (this.messageQueue.length > 0 && this.connectionState === 'connected') {
            const message = this.messageQueue.shift();
            this.sendMessage(message);
        }
    }

    startPing() {
        this.stopPing(); // Clear any existing interval
        this.pingInterval = setInterval(() => {
            if (this.connectionState === 'connected') {
                this.sendMessage({ type: 'ping' });
            }
        }, 30000); // Ping every 30 seconds
    }

    stopPing() {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
    }

    attemptReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('Max reconnection attempts reached');
            this.emit('reconnect_failed');
            return;
        }

        this.reconnectAttempts++;
        const delay = Math.min(
            this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1),
            30000 // Max 30 seconds
        );

        console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        this.emit('reconnecting', {
            attempt: this.reconnectAttempts,
            max_attempts: this.maxReconnectAttempts,
            delay: delay
        });

        setTimeout(() => {
            if (this.connectionState !== 'connected') {
                this.connect();
            }
        }, delay);
    }

    on(event, callback) {
        if (!this.listeners[event]) {
            this.listeners[event] = [];
        }
        this.listeners[event].push(callback);
        return () => this.off(event, callback); // Return unsubscribe function
    }

    off(event, callback) {
        if (this.listeners[event]) {
            this.listeners[event] = this.listeners[event].filter(cb => cb !== callback);
        }
    }

    emit(event, data) {
        if (this.listeners[event]) {
            this.listeners[event].forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in ${event} listener:`, error);
                }
            });
        }
    }

    close() {
        this.connectionState = 'disconnected';
        this.stopPing();
        this.messageQueue = [];

        if (this.ws) {
            this.ws.close(1000, 'Client closing connection');
            this.ws = null;
        }
    }

    getState() {
        return {
            connectionState: this.connectionState,
            reconnectAttempts: this.reconnectAttempts,
            clientId: this.clientId,
            sessionId: this.sessionId
        };
    }
}

// Export for use in other scripts
window.ProgressWebSocket = ProgressWebSocket;
