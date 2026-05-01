# Test Index: test

| Test File | What It Tests | Key Functions |
|-----------|---------------|----------------|
| `duplicate-space.spec.ts` | Duplicate-Space.Spec | should prepend username when only space name is provided, should preserve space ID when username matches, should throw error when trying to use different username (+ 6 more) |
| `fetch-guard.spec.ts` | Fetch-Guard.Spec | only allows direct fetch calls in network/safe-fetch.ts, fetch usage guard |
| `gradio-caller.test.ts` | Gradio-Caller | extracts the suffix after hyphen, returns null when no hyphen exists, returns null for empty input (+ 5 more) |
| `gradio-progress-relay.test.ts` | Gradio-Progress-Relay | swallows progress send failures after disconnect, callGradioToolWithHeaders progress relay |
| `paper-search.spec.ts` | Paper-Search.Spec | format the published on date correctly, handles times with no decimal point, deals with bad inputs (+ 5 more) |
| `paper-summary.spec.ts` | Paper-Summary.Spec | should handle plain arXiv ID format, should handle arxiv: prefix, should handle arxiv. prefix (typo) (+ 7 more) |
| `space-files.spec.ts` | Space-Files.Spec | should list files for a static space with subdomain, should handle spaces without subdomain, should throw error for non-static spaces (+ 12 more) |
| `space-search.spec.ts` | Space-Search.Spec | read the test file, picked up other results, SpaceSearchService |
| `user-summary.spec.ts` | User-Summary.Spec | should extract user ID from plain username, should handle usernames with whitespace, should extract user ID from hf.co URLs (+ 18 more) |
| `jobs/command-translation.spec.ts` | Command-Translation.Spec | should parse timeout with seconds, should parse timeout with minutes, should parse timeout with hours (+ 51 more) |
| `jobs/formatters.spec.ts` | Formatters.Spec | should return message for empty job list, should format a single job as markdown table, should format multiple jobs (+ 12 more) |
| `jobs/sse-handler.spec.ts` | Sse-Handler.Spec | treats timeout-aborted SSE reads as expected truncation, throws non-timeout stream errors, fetchJobLogs |
| `jobs/uv-command.spec.ts` | Uv-Command.Spec | wraps inline scripts in a shell pipeline executed via /bin/sh, includes dependency and python flags when provided, uvCommand |

⚠️ IMPORTANT: Keep this index up to date as tests are added/removed/modified. This document helps future maintainers understand test coverage at a glance.