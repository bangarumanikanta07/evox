from typing import List, Dict, Any

def compute_pareto_front(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Computes Pareto-optimal non-dominated solutions in 3D:
    Performance (maximize), Latency (minimize), Complexity (minimize).
    """
    for i, a in enumerate(models):
        dominated = False
        perf_a = a["metrics"].get("accuracy") or a["metrics"].get("r2") or 0
        lat_a = a["metrics"]["inferenceLatencyMs"]
        comp_a = a["metrics"]["modelComplexityScore"]

        for j, b in enumerate(models):
            if i == j:
                continue
            perf_b = b["metrics"].get("accuracy") or b["metrics"].get("r2") or 0
            lat_b = b["metrics"]["inferenceLatencyMs"]
            comp_b = b["metrics"]["modelComplexityScore"]

            if perf_b >= perf_a and lat_b <= lat_a and comp_b <= comp_a:
                if perf_b > perf_a or lat_b < lat_a or comp_b < comp_a:
                    dominated = True
                    break

        a["isParetoOptimal"] = not dominated

    return models
