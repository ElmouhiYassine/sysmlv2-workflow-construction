from dataclasses import dataclass


@dataclass
class ProcessingMetrics:
    # Total number of instances we expect to process.
    total_instances: int = 0
    # Number of instances seen in the main loop.
    total_processed: int = 0

    # Success / failure counters updated by the caller.
    ok: int = 0
    graph_none: int = 0
    nodes_issue: int = 0
    transform_fail: int = 0

    def counted_total(self) -> int:
        # Total accounted outcomes based on explicit counters.
        return self.ok + self.graph_none + self.nodes_issue + self.transform_fail

    def unaccounted(self) -> int:
        # Difference between processed items and counted outcomes.
        return self.total_processed - self.counted_total()



def print_metrics(metrics: ProcessingMetrics) -> None:
    # Print a compact summary used by the batch scripts.
    print(f"Total instances: {metrics.total_instances}")
    print(f"Processed: {metrics.total_processed}")
    print(f"OK: {metrics.ok}")
    print(f"graph_none: {metrics.graph_none}")
    print(f"nodes_issue: {metrics.nodes_issue}")
    print(f"transform_fail: {metrics.transform_fail}")
    print(f"Counted sum: {metrics.counted_total()}")

    if metrics.unaccounted() != 0:
        print(f"Unaccounted: {metrics.unaccounted()}")

    if metrics.total_instances > 0:
        ratio = (metrics.ok / metrics.total_instances) * 100
        print(f"OK ratio: {ratio:.2f}%")

