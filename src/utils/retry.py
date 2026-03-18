"""Retry helpers."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

import requests

T = TypeVar("T")

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}



def with_retry(
    func: Callable[[], T],
    *,
    logger: logging.Logger,
    max_retries: int,
    backoff_seconds: float,
    retry_on: tuple[type[Exception], ...] = (requests.RequestException,),
) -> T:
    attempt = 0
    while True:
        try:
            return func()
        except retry_on as exc:
            attempt += 1
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if attempt > max_retries or (status_code is not None and status_code not in RETRYABLE_STATUS_CODES):
                logger.error("Retry exhausted", exc_info=exc)
                raise
            sleep_seconds = backoff_seconds * (2 ** (attempt - 1))
            logger.warning("Transient API error, retrying", extra={"stage": "retry", "attempt": attempt, "sleep_seconds": sleep_seconds})
            time.sleep(sleep_seconds)
