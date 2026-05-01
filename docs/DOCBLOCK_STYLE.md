<!-- markdownlint-disable MD013 MD033 -->
# Docblock Style Guide

Inspired by [Laravel's documentation philosophy](https://laravel.com/docs),
this guide ensures consistent, readable docstrings across routellm's
polyglot codebase (Go, Python, TypeScript).

**Core principle**: Clear description > parameters > returns > example (if
non-obvious). Assume reader knows the language; document *intent* and
*behavior*, not syntax.

---

## Philosophy

1. **Clarity first** — describe what the function/method *does* and *why*.
   Avoid jargon; explain side effects and guarantees.
2. **Progressive disclosure** — brief summary → params → returns → example.
3. **Only when needed** — skip obvious functions; use examples for
   non-obvious or surprising behavior.
4. **Language idiom** — respect each language's conventions while keeping
   structure parallel.

---

## Go

Go uses `// ` comment style (line comments). Package-level comments use
`/* */` or multiple `// ` lines.

### Function/Method Docblock

**Pattern**: `// FunctionName <summary>. <detail>.`

- Start with function name (matches go doc convention).
- One-line summary, followed by blank line if detail needed.
- Multi-paragraph for complex behavior or caveats.
- Params documented inline only if non-obvious; otherwise obvious from type.

### Simple Example

```go
// LoadRouter initializes a router from config.
// Returns a Router interface; concrete type depends on config.RouterType.
// Returns error if config is invalid or model file not found.
func LoadRouter(ctx context.Context, cfg *Config) (Router, error) {
	// ...
}
```

### Complex Example

```go
// CalculateThreshold computes the routing threshold for a given confidence
// level using the empirical distribution of model outputs.
//
// This method applies the DKL divergence correction to account for
// calibration differences between training and inference distributions.
// Threshold is guaranteed to be within [0, 1].
//
// Note: if fewer than 100 samples are available in the confidence bucket,
// the threshold degrades to a percentile-based estimate and may be less
// stable.
func (r *SWRankingRouter) CalculateThreshold(confLevel float32) float32 {
	// ...
}
```

### Struct/Interface Docblock

```go
// Router routes prompts to weak or strong models based on difficulty.
//
// Implementations must be safe for concurrent use (via Mutex or
// immutable state).
type Router interface {
	// Route returns true if prompt should route to strong model.
	Route(ctx context.Context, prompt string) (bool, error)
	// Score returns a float in [0, 1] indicating model-specific confidence.
	Score(ctx context.Context, prompt string) (float32, error)
}
```

### When to Skip

- Private (lowercase) functions: skip unless behavior is surprising.
- Test helpers: skip unless non-obvious.
- Trivial getters/setters: skip.

---

## Python

Use Google-style or NumPy-style docstrings (NumPy preferred for clarity).
Indentation: 4 spaces. Use `"""` for multi-line, single `"""` for one-liners.

### Function/Method Docblock

**Pattern (NumPy)**:

```
"""
Brief description (one line, no period).

Extended description if needed. Explain preconditions, side effects,
and guarantees.

Parameters
----------
param_name : type
    Description. If complex, span multiple lines with 4-space indent.
param2 : type
    Description.

Returns
-------
return_type
    Description of return value(s). For tuple, describe each element.

Raises
------
ValueError
    When precondition violated.
CustomError
    When specific condition occurs.

Examples
--------
>>> result = my_function(42)
>>> print(result)
expected_output

Notes
-----
- Use bullet form for caveats.
- Explain non-obvious design choices here.

See Also
--------
related_function : Brief context.
"""
```

### Simple Example

```python
def load_router_config(path: str) -> RouterConfig:
    """Load router configuration from YAML file.

    Parameters
    ----------
    path : str
        Path to YAML config file.

    Returns
    -------
    RouterConfig
        Parsed config object.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    ValueError
        If YAML is malformed or required keys missing.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    return RouterConfig(**data)
```

### Complex Example

```python
def calculate_strong_win_rate(
    self, prompt: str, threshold: float, comparisons: list[tuple[str, str]]
) -> float:
    """Calculate strong model win rate on comparisons given a threshold.

    This method implements the two-winner routing heuristic: for each
    comparison pair, the model with score > threshold is considered "strong".
    Win rate is computed as strong_wins / total_comparisons.

    Score normalization assumes [0, 1] range; scores outside this range
    are clipped to [0, 1] before threshold comparison.

    Parameters
    ----------
    prompt : str
        Prompt used to generate model outputs (for logging/debugging only).
    threshold : float
        Score threshold in [0, 1]. Models with score >= threshold route
        to strong; otherwise to weak.
    comparisons : list[tuple[str, str]]
        List of (weak_output, strong_output) pairs to evaluate.

    Returns
    -------
    float
        Win rate in [0, 1]. Returns 0.0 if comparisons is empty.

    Raises
    ------
    ValueError
        If threshold not in [0, 1] or comparisons contains non-string pairs.

    Notes
    -----
    - This is a simple heuristic; for production, consider confidence
      intervals via bootstrap.
    - Assumes weak_output and strong_output are both valid model outputs
      (no validation performed).
    """
    # ...
```

### Class Docblock

```python
class CausalLLMRouter(Router):
    """Route prompts via causal language model scoring.

    This router uses a fine-tuned LLM (e.g., Llama-2-13B) to predict
    whether a prompt is easy (weak model sufficient) or hard (strong
    model needed). Scores are obtained via logit difference of model
    tokens.

    Attributes
    ----------
    model : AutoModelForSequenceClassification
        Loaded HuggingFace model.
    tokenizer : AutoTokenizer
        Tokenizer matching the model.
    device : str
        Device to run inference on ("cpu" or "cuda").
    """
```

### When to Skip

- `__init__` with obvious parameters: document class instead.
- Private methods (`_func`): skip unless non-obvious.
- Properties with simple getters: skip.
- Test functions: skip unless testing edge cases.

---

## TypeScript / JavaScript

Use JSDoc format (`/** */`). Indent: 2 spaces. Type annotations in JSDoc.

### Function/Method Docblock

**Pattern**:

```javascript
/**
 * Brief description.
 *
 * Extended description if needed. Explain behavior, edge cases, side
 * effects.
 *
 * @param {type} paramName - Description.
 * @param {type} param2 - Description.
 * @returns {type} Description of return value.
 * @throws {ErrorType} When error condition occurs.
 *
 * @example
 * const result = await myFunction(42);
 * console.log(result); // expected output
 *
 * @see {@link relatedFunction}
 */
```

### Simple Example

```typescript
/**
 * Parses error response from the Context7 API.
 *
 * Extracts the server's error message, falling back to status-based
 * messages if JSON parsing fails.
 *
 * @param {Response} response - Fetch Response object.
 * @param {string} [apiKey] - Optional API key for fallback messages.
 * @returns {Promise<string>} Error message.
 */
async function parseErrorResponse(response: Response, apiKey?: string): Promise<string> {
  // ...
}
```

### Complex Example

```typescript
/**
 * Routes a prompt to weak or strong model based on difficulty score.
 *
 * Compares prompt embedding similarity to training data. If similarity
 * exceeds threshold, routes to strong model; otherwise weak. Similarity
 * is computed via cosine distance normalized to [0, 1].
 *
 * This router requires pre-computed embeddings; call `ensureEmbeddings()`
 * before first use.
 *
 * @param {string} prompt - User input prompt.
 * @param {number} [threshold=0.5] - Similarity threshold in [0, 1].
 * @returns {Promise<boolean>} True if should route to strong model.
 * @throws {Error} If embeddings not yet loaded.
 *
 * @example
 * const router = new SimilarityRouter(embeddingModel);
 * await router.ensureEmbeddings();
 * const shouldRouteStrong = await router.route("What is 2+2?", 0.7);
 * console.log(shouldRouteStrong); // false (easy prompt)
 */
async route(prompt: string, threshold: number = 0.5): Promise<boolean> {
  // ...
}
```

### Class/Interface Docblock

```typescript
/**
 * Routes prompts using Similarity Weighted Ranking algorithm.
 *
 * Maintains a weighted ranking of training samples based on prompt
 * similarity. Threshold is dynamic and adjusts based on observed
 * routing accuracy.
 *
 * Implements {@link Router} interface.
 *
 * @class
 */
class SWRankingRouter implements Router {
  /**
   * Initialize router with embedding model.
   *
   * @param {EmbeddingModel} model - Model for computing prompt embeddings.
   * @param {number} [timeout=30000] - Request timeout in milliseconds.
   */
  constructor(model: EmbeddingModel, timeout?: number) {
    // ...
  }

  /**
   * Compute normalized weight for each sample based on similarity.
   *
   * @param {number[]} similarities - Array of similarity scores.
   * @returns {number[]} Normalized weights summing to 1.0.
   */
  getWeightings(similarities: number[]): number[] {
    // ...
  }
}
```

### When to Skip

- Private methods: skip unless non-obvious.
- Simple getters/setters: skip.
- Test utilities: skip unless testing complex scenarios.

---

## Examples: When to Include

Include an example if:

1. **Behavior is surprising or non-obvious**.
   - Thresholds, side effects, state mutations.
   - Example: "Returns normalized score in [0, 1]." (OK without example)
   - Non-obvious: "Threshold is adaptive; see example for expected range."

2. **Parameter/return format is complex or ambiguous**.
   - Example: "Returns tuple (score, confidence)." (OK without)
   - Non-obvious: "Returns dict with keys 'easy', 'hard', 'uncertain'."

3. **Function chains or multi-step setup**.
   - Example: Router requires `ensureEmbeddings()` before use.

4. **Error cases or edge behavior**.
   - Example: Empty input → returns 0.0; null → raises ValueError.

### Example Format

**Go**: Use `Example_` test-like comments; keep short.

```go
// Example:
//   router, _ := LoadRouter(ctx, cfg)
//   score, _ := router.Score(ctx, "What is 2+2?")
//   fmt.Println(score) // 0.75
```

**Python**: Doctest-style in NumPy `Examples` section.

```python
Examples
--------
>>> router = load_router_config("config.yaml")
>>> score = router.calculate_strong_win_rate("prompt", 0.5, [])
>>> print(score)
0.0
```

**TypeScript**: JSDoc `@example` tag; markdown-friendly.

```typescript
* @example
* const router = new BERTRouter(model);
* const shouldRoute = await router.route("Is AI safe?");
* console.log(shouldRoute); // true (hard prompt)
```

---

## Examples: When to Skip

Skip examples if:

1. **Function behavior is obvious from signature and description**.
   - `isValid(item: Item): boolean` — clear without example.

2. **Function is a simple wrapper or adapter**.
   - `parseInt(str: string): int` — no example needed.

3. **Example would just repeat docstring**.
   - Bad: "Create a new router." with example `router = Router()`.

4. **Setup is complex and belongs in a guide, not docstring**.
   - Instead: "See `docs/setup-routers.md` for full example."

---

## Common Mistakes to Avoid

### ❌ Over-Documentation

```go
// Bad: repeats signature without adding value
// ProcessRequest takes a ctx context.Context and returns an error.
func ProcessRequest(ctx context.Context) error { }
```

### ✅ Focused Documentation

```go
// ProcessRequest blocks until request is fully processed or ctx cancels.
// Returns error if validation fails; returns nil on success.
func ProcessRequest(ctx context.Context) error { }
```

---

### ❌ Jargon Without Explanation

```python
"""Compute DKL divergence correction w/ empirical distribution."""
```

### ✅ Explained Jargon

```python
"""Compute DKL divergence correction to account for distribution shift
between training and inference data."""
```

---

### ❌ Parameters Without Types

```typescript
/**
 * Route prompt to strong or weak model.
 * @param prompt - User input.
 * @param threshold - Confidence cutoff.
 */
```

### ✅ Parameters With Types and Context

```typescript
/**
 * Route prompt to strong or weak model.
 * @param {string} prompt - User input prompt.
 * @param {number} threshold - Confidence cutoff in [0, 1].
 */
```

---

### ❌ Missing Edge Cases

```python
def split_data(data: list) -> tuple:
    """Split data into train and test."""
```

### ✅ Edge Cases Documented

```python
def split_data(data: list, ratio: float = 0.8) -> tuple:
    """Split data into train and test.

    Parameters
    ----------
    data : list
        Input data. Must not be empty.
    ratio : float
        Train fraction in (0, 1). Default 0.8.

    Returns
    -------
    tuple
        (train_data, test_data) where sum of lengths = len(data).

    Raises
    ------
    ValueError
        If data is empty or ratio not in (0, 1).
    """
```

---

### ❌ Vague Return Descriptions

```go
// Calculates the score and returns it.
func Score(prompt string) float32 { }
```

### ✅ Precise Return Descriptions

```go
// Score computes a confidence score using the similarity-weighted
// ranking algorithm. Score is normalized to [0, 1], where 0 indicates
// strong model is needed and 1 indicates weak model is sufficient.
func Score(prompt string) float32 { }
```

---

## Checklist

Before committing, verify:

- [ ] **Description** is clear (intent, not syntax).
- [ ] **Parameters** list type and purpose (not self-evident from type).
- [ ] **Returns** describe format and range (if non-obvious).
- [ ] **Errors** document expected exceptions/outcomes.
- [ ] **Examples** included only if behavior is surprising.
- [ ] **Formatting** follows language conventions (Go/Python/TS style).
- [ ] **No jargon** without explanation; no copy-paste from comments.
- [ ] **No redundancy** with signature; docstring adds information.

---

## References

- [Go Code Review Comments](https://github.com/golang/go/wiki/CodeReviewComments)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html)
- [JSDoc](https://jsdoc.app/)
- [Laravel Docs Philosophy](https://laravel.com/docs)

---

**Last updated**: 2026-04-30  
**Maintained by**: routellm maintainers
