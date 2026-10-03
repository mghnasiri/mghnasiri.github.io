"""
Shared, rate-safe GET for the NHL APIs.

api-web throttles bursts (429). Each script used to carry its own copy of
api_get that retried a URL on its own clock and gave up after ~3s, so
during a throttle window one player's fetch after another failed and those
players silently left the slate (a 2026-09-30 test run kept 15 of PIT's
24 skaters). Here a 429 holds back every later call until the block has
passed, and connections are reused.

Once a burst trips the limit, api-web blocks for about a minute (2026-10-03
CI: 60-65s blocks after every ~50-170 calls), so a 429 backs off 15, 30,
then 60s: ~105s in all, past one block.
"""

import time

import requests

_session = requests.Session()
_resume_at = 0.0  # time.monotonic() before which no call is sent
MAX_WAIT = 60     # cap on a single Retry-After / backoff sleep (seconds)


def _retry_after(resp, attempt):
    """Seconds to back off after a 429: 15, 30, 60, ..., or the server's
    Retry-After if that is longer."""
    wait = min(15 * 2 ** attempt, MAX_WAIT)
    try:
        return max(wait, min(float(resp.headers.get("Retry-After", "")), MAX_WAIT))
    except ValueError:
        return wait


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
