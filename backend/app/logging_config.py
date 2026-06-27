import logging
import sys

from app.config import get_settings
from app.concurrency import request_id_var


class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_var.get()
        return True


def setup_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestIdFilter())
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | [%(request_id)s] | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
