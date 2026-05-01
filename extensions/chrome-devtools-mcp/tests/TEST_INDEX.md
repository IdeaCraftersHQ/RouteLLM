# Test Index: tests

| Test File | What It Tests | Key Functions |
|-----------|---------------|----------------|
| `DevtoolsUtils.test.ts` | Devtoolsutils | deals with no trailing /, deals with a trailing /, deals with www (+ 14 more) |
| `McpContext.test.ts` | Mcpcontext | list pages, can store and retrieve the latest performance trace, should update default timeout when cpu throttling changes (+ 8 more) |
| `McpResponse.test.ts` | Mcpresponse | list pages, allows response text lines to be added, does not include anything in response if snapshot is null (+ 46 more) |
| `PageCollector.test.ts` | Pagecollector | works, request, clean up after navigation (+ 20 more) |
| `browser.test.ts` | Browser | detects display does not crash, cannot launch multiple times with the same profile, launches with the initial viewport (+ 2 more) |
| `cli.test.ts` | Cli | parses with default args, parses with browser url, parses with user data dir (+ 13 more) |
| `index.test.ts` | Index | calls a tool, calls a tool multiple times, has all tools (+ 4 more) |
| `third_party_notices.test.ts` | Third Party Notices | matches snapshot if exists, THIRD_PARTY_NOTICES |
| `daemon/client.test.ts` | Client | should start and stop daemon, should handle starting daemon when already running, should handle stopping daemon when not running (+ 8 more) |
| `daemon/utils.test.ts` | Utils | should ignore undefined or null values, should handle boolean values, should handle array values (+ 3 more) |
| `e2e/chrome-devtools.test.ts` | Chrome-Devtools | reports daemon status correctly, can start and stop the daemon, can invoke list_pages (+ 3 more) |
| `e2e/telemetry.test.ts` | Telemetry | handles SIGKILL, handles SIGTERM, handles POSIX process group SIGTERM (+ 1 more) |
| `formatters/ConsoleFormatter.test.ts` | Consoleformatter | ConsoleFormatter, toString/toJSON, toStringDetailed/toJSONDetailed |
| `formatters/IssueFormatter.test.ts` | Issueformatter | returns false for the issue with no description, returns false if there is no description file, returns false if can (+ 4 more) |
| `formatters/NetworkFormatter.test.ts` | Networkformatter | works, shows correct method, shows correct status for request with response code in 200 (+ 21 more) |
| `formatters/snapshotFormatter.test.ts` | Snapshotformatter | formats a snapshot with value properties, formats a snapshot with boolean properties, formats a snapshot with checked properties (+ 6 more) |
| `telemetry/ClearcutLogger.test.ts` | Clearcutlogger | sends correct payload, logs flag usage, logs daily active if needed (lastActive > 24h ago) (+ 6 more) |
| `telemetry/WatchdogClient.test.ts` | Watchdogclient | spawns watchdog process with correct arguments, passes log-file argument if provided, sends IPC messages via stdin (+ 2 more) |
| `telemetry/flagUtils.test.ts` | Flagutils | logs boolean flags directly with snake_case keys, logs boolean flags as false when false, logs enum flags as uppercase strings prefixed by snake case flag name (+ 9 more) |
| `telemetry/metricUtils.test.ts` | Metricutils | should bucketize values correctly, bucketizeLatency |
| `telemetry/persistence.test.ts` | Persistence | returns default state if file does not exist, returns stored state if file exists, saves state to file (+ 3 more) |
| `telemetry/watchdog/ClearcutSender.test.ts` | Clearcutsender | enriches events with app version, os type, and session id, accumulates events in buffer without immediate send, sends correct LogRequest format (+ 14 more) |
| `tools/console.test.ts` | Console | list messages, lists error messages, lists error objects (+ 17 more) |
| `tools/emulation.test.ts` | Emulation | returns undefined for undefined input, parses basic dimensions, parses dimensions with devicePixelRatio (+ 31 more) |
| `tools/extensions.test.ts` | Extensions | installs and uninstalls an extension and verifies it in chrome://extensions, lists installed extensions, reloads an extension (+ 2 more) |
| `tools/input.test.ts` | Input | clicks, double clicks, waits for navigation (+ 33 more) |
| `tools/lighthouse.test.ts` | Lighthouse | runs Lighthouse audit by default (navigation, desktop), restores emulation, runs Lighthouse in snapshot mode with mobile device (+ 3 more) |
| `tools/memory.test.ts` | Memory | with default options, memory, take_memory_snapshot |
| `tools/network.test.ts` | Network | list requests, list requests form current navigations only, list requests from previous navigations (+ 7 more) |
| `tools/pages.test.ts` | Pages | list pages, create a page, create a page in the background (+ 44 more) |
| `tools/performance.test.ts` | Performance | starts a trace recording, can navigate to about:blank and record a page reload, can autostop and store a recording (+ 12 more) |
| `tools/screencast.test.ts` | Screencast | starts a screencast recording with filePath, starts a screencast recording with temp file when no filePath, errors if a recording is already active (+ 7 more) |
| `tools/screenshot.test.ts` | Screenshot | with default options, ignores quality, with jpeg (+ 9 more) |
| `tools/script.test.ts` | Script | evaluates, runs in selected page, work for complex objects (+ 9 more) |
| `tools/snapshot.test.ts` | Snapshot | includes a snapshot, should work, should work with any-match array (+ 7 more) |
| `tools/slim/tools.test.ts` | Tools | evaluates, handles errors, navigates to correct page (+ 2 more) |
| `trace-processing/parse.test.ts` | Parse | can parse a Uint8Array from Tracing.stop()), can format results of a trace, will return a message if there is an error (+ 1 more) |

⚠️ IMPORTANT: Keep this index up to date as tests are added/removed/modified. This document helps future maintainers understand test coverage at a glance.