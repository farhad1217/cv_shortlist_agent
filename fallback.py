"""Model fallback runner: retry with backoff on temporary errors, cool-down for a failing model,
and an error message that shows what EVERY model said (LangChain's with_fallbacks hides all but the first)."""
import time

_TRANSIENT = ("503", "unavailable", "overloaded", "high demand", "429", "rate limit", "rate_limit", "quota",
              "timeout", "timed out", "temporarily", "500", "502", "504", "connection", "tool_use_failed",
              "failed to call a function")
_down_until = {}  # model name -> unix time until which it is skipped


def transient(err) -> bool:
    s = str(err).lower()
    return any(t in s for t in _TRANSIENT)


def short(err, n=140):
    return " ".join(str(err).split())[:n]


class FallbackChain:
    def __init__(self, runnables, names, retries=2, base_delay=1.5, cooldown=45, accept=None, sleep=time.sleep):
        self.rs, self.names = list(runnables), list(names)
        self.retries, self.base_delay, self.cooldown = retries, base_delay, cooldown
        self.accept = accept or (lambda r: r is not None)
        self.sleep = sleep

    def _order(self):
        now = time.time()
        idx = list(range(len(self.rs)))
        ready = [i for i in idx if _down_until.get(self.names[i], 0) <= now]
        return ready + [i for i in idx if i not in ready]  # cooling-down models are tried last, not never

    def invoke(self, msgs, config=None):
        errors = []
        for i in self._order():
            name = self.names[i]
            for attempt in range(self.retries + 1):
                try:
                    r = self.rs[i].invoke(msgs, config=config)
                    if not self.accept(r):
                        raise ValueError("empty/invalid model output")
                    _down_until.pop(name, None)
                    return r
                except Exception as e:  # noqa
                    if attempt < self.retries and transient(e):
                        self.sleep(self.base_delay * (2 ** attempt))
                        continue
                    errors.append(f"{name}: {short(e)}")
                    if transient(e):
                        _down_until[name] = time.time() + self.cooldown
                    break
        raise RuntimeError(" | ".join(errors) or "no model configured")
