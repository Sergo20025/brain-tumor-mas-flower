from __future__ import annotations


def coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def format_class_histogram(histogram: dict[str, int], limit: int = 4) -> str:
    items = sorted(histogram.items(), key=lambda item: item[0])
    visible = items[:limit]
    text = ", ".join(f"{label}={count}" for label, count in visible)
    if len(items) > limit:
        text += ", ..."
    return text


def print_agent_log(
    agent_name: str,
    message: str,
    *,
    partition_id: int | None = None,
    client_id: str | None = None,
    round_number: int | None = None,
) -> None:
    prefix_parts = [agent_name]
    if client_id is not None:
        prefix_parts.append(f"client={client_id}")
    if partition_id is not None:
        prefix_parts.append(f"partition={partition_id}")
    if round_number is not None:
        prefix_parts.append(f"round={round_number}")
    print(f"[{' | '.join(prefix_parts)}] {message}", flush=True)
