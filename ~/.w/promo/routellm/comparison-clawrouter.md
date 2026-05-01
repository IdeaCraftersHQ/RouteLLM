# Feature Comparison: RouteLLM vs. ClawRouter

## 1. Core Philosophy
| Feature | RouteLLM | ClawRouter |
| :--- | :--- | :--- |
| **Primary Goal** | **Cost-Quality Pareto Optimization** | **Agent-Native Cost Reduction** |
| **Routing Logic** | **Learned Preference Models** (Predictive) | **Heuristic/Weighted Scorer** (15-dim) |
| **Decision Location** | Server-side (Python/Go) | Client-side (Local Proxy) |
| **Payment Model** | Traditional (API Keys) | Decentralized (x402 / USDC on Base/Solana) |

## 2. Verified Technical Performance (Avg across MT Bench, GSM8K, MMLU)
| Metric | RouteLLM | ClawRouter |
| :--- | :--- | :--- |
| **Quality Parity** | **95.0%** (Verified) | **89.0%** (Estimated Avg) |
| **Cost Savings** | **84.3%** (Verified) | **88.5%** (Reported Avg) |
| **Routing Latency** | **Variable** (BERT inference / MF embeddings) | **Ultra-low** (<1ms, local execution) |
| **Fallbacks** | Configurable via LiteLLM | Automatic to `gpt-oss-120b` (Free tier) |

## 3. Key Differentiators
- **RouteLLM**: Focuses on **Cost-Quality Pareto Optimization**. It uses preference data to *predict* the absolute limit of cost savings while maintaining a specific quality threshold. It is a research-backed framework (LMSYS/Anyscale).
- **ClawRouter**: Focuses on **Agent-Native Cost Reduction**. It provides a non-custodial payment layer (no API keys needed) and a very fast, rule-based local router for high-frequency agent calls where cost is the primary constraint.

## 4. Integration
- **RouteLLM**: Drop-in OpenAI replacement; Python SDK; OpenAI-compatible server.
- **ClawRouter**: TypeScript-based local proxy; optimized for autonomous agent ecosystems.
