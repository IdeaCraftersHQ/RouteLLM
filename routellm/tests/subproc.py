"""Shared limits for tests that boot the server in a child interpreter.

Every such test pays a cold `import litellm` plus the server import
(seconds of CPU on an idle machine). Wall-clock time for that scales
with machine load, so a fixed literal timeout turns CPU starvation
into a spurious failure. The timeout here is only a hang ceiling:
the child is killed if it truly wedges, never as a speed assertion.
Readiness inside the child is always a condition poll with a deadline
(`WAIT_FOR`), never a fixed sleep.
"""

import os

# Hang ceiling for a child interpreter. Override on a slow runner.
SUBPROCESS_TIMEOUT = float(os.environ.get("ROUTELLM_TEST_SUBPROCESS_TIMEOUT", "600"))

# Condition-poll deadline, seconds. Returns as soon as the condition
# holds, so a generous value costs nothing on a healthy machine.
WAIT_DEADLINE = float(os.environ.get("ROUTELLM_TEST_WAIT_DEADLINE", "60"))
