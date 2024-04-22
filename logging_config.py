import logging
import logging.handlers

log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
handler = logging.handlers.RotatingFileHandler('api.log', maxBytes=1024 * 1024, backupCount=5)  
handler.setFormatter(logging.Formatter(log_format))

logger = logging.getLogger("uvicorn")  
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)