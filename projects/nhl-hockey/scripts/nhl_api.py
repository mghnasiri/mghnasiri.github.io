"""
Shared, rate-safe GET for the NHL APIs.

api-web throttles bursts (429). Each script used to carry its own copy of
api_get that retried a URL on its own clock and gave up after ~3s, so
during a throttle window one player's fetch after another failed and those
players silently left the slate (a 2026-09-30 test run kept 15 of PIT's
24 skaters). Here a 429 holds back every later call until the server's
Retry-After has passed, and connections are reused.
"""

import time

import requests

_session = requests.Session()
_resume_at = 0.0  # time.monotonic() before which no call is sent
MAX_WAIT = 60     # cap on a single Retry-After / backoff sleep (seconds)


def _retry_after(resp, attempt):
    """Seconds to back off after a 429: the server's Retry-After when it
    gives a number, else exponential (2, 4, 8, ...)."""
    try:
        return min(float(resp.headers.get("Retry-After", "")), MAX_WAIT)
    except ValueError:
        return min(2 ** (attempt + 1), MAX_WAIT)


def api_get(url, timeout=15, attempts=4):
    """Parsed JSON from url; None on 404 or after `attempts` failures."""
    global _resume_at
    status = None
    for attempt in range(attempts):
        wait = _resume_at - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            resp = _session.get(url, timeout=timeout)
            status = resp.status_code
            if status == 200:
                return resp.json()
            if status == 404:
                return None
            if status == 429:
                _resume_at = time.monotonic() + _retry_after(resp, attempt)
                continue
        except (requests.RequestException, ValueError) as e:
            status = type(e).__name__
        if attempt < attempts - 1:
            time.sleep(min(2 ** attempt, MAX_WAIT))
    print(f"   WARNING: {url} -> {status} after {attempts} attempts")
    return None
