import logging
import os
import sys

def setup_logging():
    """Configures application-wide logging based on ADK standards.
    
    This function should be called at the very beginning of the application entry point.
    It reads the LOG_LEVEL environment variable (default: INFO) and configures
    the root logger to output structured logs to stdout.
    """
    
    # 1. Determine Log Level
    # Default to INFO, allow override via LOG_LEVEL env var
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # 2. Define Format
    # Standard ADK-friendly format including timestamp and logger name
    log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'

    # 3. Apply Configuration
    # Force=True allows re-configuration if standard libs already set defaults
    logging.basicConfig(
        level=log_level,
        format=log_format,
        stream=sys.stdout,
        force=True
    )
    
    # 4. Silence Noisy Third-Party Libraries
    # Raise level for chatty libraries to avoid noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("google.auth").setLevel(logging.WARNING)
    logging.getLogger("absl").setLevel(logging.WARNING) # For grpc/protobuf noise

    # Log confirmation
    logging.info(f"Logging configured at level: {log_level_str}")

if __name__ == "__main__":
    setup_logging()
    logging.info("Logging configuration test successful.")
    logging.debug("This is a debug message (visible if LOG_LEVEL=DEBUG)")
