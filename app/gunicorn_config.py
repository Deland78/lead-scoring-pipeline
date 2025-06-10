# gunicorn_config.py
import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', 5000)}"
backlog = 2048

# Worker processes
workers = int(os.environ.get('WORKERS', 2))  # Reduce workers to save memory
worker_class = "sync"
worker_connections = 1000
timeout = 120  # Increased timeout for model inference
keepalive = 2
max_requests = 1000
max_requests_jitter = 50

# Restart workers after this many requests to prevent memory leaks
preload_app = True  # Load app before forking workers (better for ML models)

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "lead_scoring_app"

# Memory management
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190