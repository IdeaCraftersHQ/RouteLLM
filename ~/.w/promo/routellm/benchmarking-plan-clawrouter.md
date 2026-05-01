# Benchmarking Plan: RouteLLM vs. ClawRouter

## 1. Objectives
- Compare **routing accuracy**: Does RouteLLM's learned approach outperform ClawRouter's heuristic approach on the same budget?
- Compare **latency**: Measure the overhead of RouteLLM's MF/BERT routers vs. ClawRouter's local scoring.
- Compare **cost-savings**: Verify real-world savings on standardized datasets.

## 2. Methodology
- **Dataset**: Use **MT Bench** and **GSM8K** to evaluate quality vs. cost.
- **Environment**: 
  - Standardized model pair: `gpt-4o` (Strong) and `deepseek-v3` (Weak).
  - Both routers configured for the same budget (e.g., 50% Strong model calls).

## 3. Metrics
| Metric | Measurement |
| :--- | :--- |
| **Pass@1** | Accuracy on benchmark tasks. |
| **Cost per 1k tokens** | Blended average cost of routed calls. |
| **Routing Latency** | Time taken for the router to decide (TTFD - Time To First Decision). |
| **Success Rate** | Percentage of calls successfully routed to a model. |

## 4. Execution Phases
1. **Foundation**: Set up a standardized evaluation harness in Python.
2. **Integration**: Create a wrapper for ClawRouter to call its routing logic via its CLI or proxy.
3. **Execution**: Run evaluations on MT Bench and GSM8K.
4. **Synthesis**: Create a comparison report with visualizations (Cost-Quality Pareto Front).
