# RouteLLM Unique Selling Propositions (USPs)

RouteLLM is the **Multi-Protocol Router for LLMs**. It solves the cost-intelligence dilemma for production apps.

## 1. Cost Reduction (up to 85%)
Unlike direct model calls, RouteLLM routes simpler queries to cheaper models.
- **MF Router**: Matrix Factorization trained on preference data.
- **Threshold Calibration**: Tune cost-quality trade-off for any query distribution.

## 2. Performance Parity (95% GPT-4)
Maintains high quality on benchmarks like MT Bench and GSM8K.
- **Intelligent Routing**: Only uses strong models when complexity demands it.
- **Paper Backed**: Research from LMSYS and Anyscale.

## 3. Drop-in OpenAI Compatibility
Works with any existing OpenAI client by changing only the `base_url` or using the Python SDK.
- **Server Mode**: Launch an OpenAI-compatible server in seconds.
- **Middleware**: Intent-based routing for specialized domains (coding, math).

## 4. Multi-Protocol & Multi-Provider
Leverages LiteLLM to support Anthropic, Gemini, Bedrock, Together AI, and local models (Ollama).
- **Go Proxy (Planned)**: Sub-ms overhead for the routing layer.
- **Universal Bridge**: Route across any set of providers with one config.
