from __future__ import annotations

from typing import Any

from flwr.app import Context
from flwr.clientapp import ClientApp

from brain_tumor_fl.agents.compute_agent import BrainTumorClient, ComputeAgent
from brain_tumor_fl.utils import print_agent_log


def _context_to_run_config(context: Context) -> dict[str, Any]:
    return dict(context.run_config)


def client_fn(context: Context):
    run_config = _context_to_run_config(context)
    partition_id = int(
        context.node_config.get(
            "partition-id",
            max(int(context.node_id) - 1, 0),
        )
    )
    print_agent_log(
        "ClientApp",
        f"create client app for node_id={context.node_id}",
        partition_id=partition_id,
    )
    compute_agent = ComputeAgent(partition_id=partition_id, config=run_config)
    return BrainTumorClient(compute_agent).to_client()


app = ClientApp(client_fn=client_fn)
