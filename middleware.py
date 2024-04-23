from fastapi import Request, Response
from datetime import datetime
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = datetime.now() 
        try:
            response = await call_next(request)
        except Exception as e:  # Catch potential routing errors
            logger.error(f"Invalid URL: {request.url} - Error: {e}")
            response = JSONResponse(status_code=404, content={"message": "Not found"})  

        process_time = (datetime.now() - start_time).microseconds

        url_path = request.url.path

        log_message = f"{url_path} - {request.method} - {response.status_code} - processed in {process_time} microseconds"
        
        # Log depending on success or failure
        if response.status_code >= 400: 
            logger.error(log_message)
        else:
            logger.info(log_message)

        return response