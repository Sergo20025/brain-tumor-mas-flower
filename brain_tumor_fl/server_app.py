from __future__ import annotations

from typing import Any

from flwr.app import Context
from flwr.server import ServerAppComponents, ServerConfig
from flwr.serverapp import ServerApp

from brain_tumor_fl.agents.aggregation_agent import AggregationAgent
from brain_tumor_fl.agents.monitoring_agent import MonitoringAgent
from brain_tumor_fl.agents.storage_agent import StorageAgent
from brain_tumor_fl.strategy import TrustAwareFedAvg
from brain_tumor_fl.utils import coerce_bool


def _context_to_run_config(context: Context) -> dict[str, Any]:
    return dict(context.run_config)


def server_fn(context: Context) -> ServerAppComponents:
    run_config = _context_to_run_config(context)

    storage_agent = StorageAgent(run_config)
    monitoring_agent = MonitoringAgent(save_path=str(run_config["save-metrics-path"]))
    aggregation_agent = AggregationAgent(
        decentralized_mode=coerce_bool(run_config["decentralized-mode"])
    )

    strategy = TrustAwareFedAvg(
        storage_agent=storage_agent,
        monitoring_agent=monitoring_agent,
        aggregation_agent=aggregation_agent,
        run_config=run_config,
    )

    config = ServerConfig(num_rounds=int(run_config["num-server-rounds"]))
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
