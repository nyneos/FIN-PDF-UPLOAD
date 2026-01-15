"""
Real-time log streaming for monitoring application behavior.
Provides SSE (Server-Sent Events) endpoint to stream logs to clients.
"""

import queue
import logging
import threading
import time
from typing import Optional


class StreamingLogHandler(logging.Handler):
    """
    Custom logging handler that broadcasts log records to all connected clients.
    Thread-safe implementation using Queue for each client.
    """
    
    def __init__(self):
        super().__init__()
        self.clients = []  # List of queues, one per client
        self.lock = threading.Lock()
        self.max_clients = 10  # Prevent DOS via too many log stream connections
        
    def emit(self, record):
        """Send log record to all connected clients."""
        try:
            # Format the log record
            log_entry = self.format(record)
            
            # Broadcast to all clients
            with self.lock:
                dead_clients = []
                for client_queue in self.clients:
                    try:
                        # Non-blocking put with size limit
                        if client_queue.qsize() < 1000:  # Prevent memory overflow
                            client_queue.put_nowait(log_entry)
                        else:
                            # Client is too slow, disconnect
                            dead_clients.append(client_queue)
                    except queue.Full:
                        dead_clients.append(client_queue)
                
                # Remove dead clients
                for dq in dead_clients:
                    if dq in self.clients:
                        self.clients.remove(dq)
                        
        except Exception:
            # Never let logging errors crash the app
            pass
    
    def add_client(self) -> Optional[queue.Queue]:
        """Register a new client for log streaming."""
        with self.lock:
            if len(self.clients) >= self.max_clients:
                return None  # Too many clients
            
            client_queue = queue.Queue(maxsize=1000)
            self.clients.append(client_queue)
            return client_queue
    
    def remove_client(self, client_queue: queue.Queue):
        """Unregister a client."""
        with self.lock:
            if client_queue in self.clients:
                self.clients.remove(client_queue)


# Global streaming handler (singleton)
_streaming_handler = None


def get_streaming_handler() -> StreamingLogHandler:
    """Get or create the global streaming log handler."""
    global _streaming_handler
    if _streaming_handler is None:
        _streaming_handler = StreamingLogHandler()
        
        # Set formatter
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        _streaming_handler.setFormatter(formatter)
        
        # Attach to root logger
        logging.getLogger().addHandler(_streaming_handler)
    
    return _streaming_handler


def log_stream_generator(client_queue: queue.Queue, timeout: int = 300):
    """
    Generator that yields log entries from client queue.
    
    Args:
        client_queue: Queue to read log entries from
        timeout: Maximum time to keep connection alive (seconds)
    
    Yields:
        SSE-formatted log entries
    """
    start_time = time.time()
    
    try:
        # Send initial connection message
        yield f"data: Connected to log stream at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        while True:
            # Check timeout
            if time.time() - start_time > timeout:
                yield f"data: [TIMEOUT] Stream closed after {timeout}s\n\n"
                break
            
            try:
                # Wait for log entry (with timeout to allow periodic checks)
                log_entry = client_queue.get(timeout=1.0)
                
                # Send as SSE format
                yield f"data: {log_entry}\n\n"
                
            except queue.Empty:
                # Send keepalive ping every second
                yield f": keepalive\n\n"
                continue
                
    except GeneratorExit:
        # Client disconnected
        pass
    finally:
        # Cleanup: remove client from handler
        # Only attempt cleanup if handler exists (avoid recreating it)
        if _streaming_handler is not None:
            _streaming_handler.remove_client(client_queue)
