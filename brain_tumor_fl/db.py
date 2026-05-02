from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency guard
    load_dotenv = None

try:
    from sqlalchemy import (
        Boolean,
        DateTime,
        Float,
        ForeignKey,
        Integer,
        String,
        Text,
        create_engine,
        select,
    )
    from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker
except ImportError:  # pragma: no cover - optional dependency guard
    DeclarativeBase = object  # type: ignore[assignment]
    Mapped = object  # type: ignore[assignment]
    Session = object  # type: ignore[assignment]
    mapped_column = None
    relationship = None
    sessionmaker = None
    create_engine = None
    select = None


def _utcnow() -> datetime:
    return datetime.now(UTC)


SQLALCHEMY_AVAILABLE = create_engine is not None


if SQLALCHEMY_AVAILABLE:

    class Base(DeclarativeBase):
        pass


    class ExperimentRecord(Base):
        __tablename__ = "experiments"

        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        experiment_name: Mapped[str] = mapped_column(String(255), nullable=False)
        experiment_dir: Mapped[str] = mapped_column(Text, nullable=False)
        mode: Mapped[str] = mapped_column(String(64), nullable=False)
        strategy_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
        dataset_name: Mapped[str] = mapped_column(String(128), nullable=False)
        model_name: Mapped[str] = mapped_column(String(128), nullable=False)
        partition_mode: Mapped[str] = mapped_column(String(128), nullable=False)
        dirichlet_alpha: Mapped[float | None] = mapped_column(Float, nullable=True)
        soft_mix_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
        soft_min_extra_classes: Mapped[int | None] = mapped_column(Integer, nullable=True)
        topology_mode: Mapped[str | None] = mapped_column(String(64), nullable=True)
        topology_extra_offset: Mapped[int | None] = mapped_column(Integer, nullable=True)
        async_mode: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
        async_dropout_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
        max_async_dropouts: Mapped[int | None] = mapped_column(Integer, nullable=True)
        heterogeneous_nodes: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
        num_clients: Mapped[int] = mapped_column(Integer, nullable=False)
        num_server_rounds: Mapped[int] = mapped_column(Integer, nullable=False)
        local_epochs: Mapped[int] = mapped_column(Integer, nullable=False)
        batch_size: Mapped[int] = mapped_column(Integer, nullable=False)
        learning_rate: Mapped[float] = mapped_column(Float, nullable=False)
        weight_decay: Mapped[float] = mapped_column(Float, nullable=False)
        use_pretrained: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
        status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
        started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)
        finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
        error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
        best_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
        best_f1_macro: Mapped[float | None] = mapped_column(Float, nullable=True)
        best_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
        raw_config_json: Mapped[str] = mapped_column(Text, nullable=False)

        round_metrics: Mapped[list["RoundMetricRecord"]] = relationship(
            back_populates="experiment",
            cascade="all, delete-orphan",
        )
        client_metrics: Mapped[list["ClientMetricRecord"]] = relationship(
            back_populates="experiment",
            cascade="all, delete-orphan",
        )
        artifacts: Mapped[list["ArtifactRecord"]] = relationship(
            back_populates="experiment",
            cascade="all, delete-orphan",
        )


    class RoundMetricRecord(Base):
        __tablename__ = "round_metrics"

        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"), nullable=False, index=True)
        round_number: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
        train_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
        train_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
        train_f1: Mapped[float | None] = mapped_column(Float, nullable=True)
        val_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
        val_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
        val_f1: Mapped[float | None] = mapped_column(Float, nullable=True)
        participating_clients: Mapped[float | None] = mapped_column(Float, nullable=True)
        skipped_clients: Mapped[float | None] = mapped_column(Float, nullable=True)
        global_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
        global_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
        global_f1_macro: Mapped[float | None] = mapped_column(Float, nullable=True)
        global_precision_macro: Mapped[float | None] = mapped_column(Float, nullable=True)
        global_recall_macro: Mapped[float | None] = mapped_column(Float, nullable=True)

        experiment: Mapped["ExperimentRecord"] = relationship(back_populates="round_metrics")


    class ClientMetricRecord(Base):
        __tablename__ = "client_metrics"

        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"), nullable=False, index=True)
        round_number: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
        client_name: Mapped[str] = mapped_column(String(64), nullable=False)
        partition_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
        num_examples: Mapped[int | None] = mapped_column(Integer, nullable=True)
        trust_score: Mapped[float | None] = mapped_column(Float, nullable=True)
        resource_profile: Mapped[str | None] = mapped_column(String(64), nullable=True)
        resource_batch_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
        virtual_delay_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
        train_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
        train_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
        train_f1: Mapped[float | None] = mapped_column(Float, nullable=True)
        val_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
        val_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
        val_f1: Mapped[float | None] = mapped_column(Float, nullable=True)
        train_time_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
        update_l2_norm: Mapped[float | None] = mapped_column(Float, nullable=True)
        skipped: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
        completed_round: Mapped[int | None] = mapped_column(Integer, nullable=True)
        server_round: Mapped[int | None] = mapped_column(Integer, nullable=True)

        experiment: Mapped["ExperimentRecord"] = relationship(back_populates="client_metrics")


    class ArtifactRecord(Base):
        __tablename__ = "artifacts"

        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"), nullable=False, index=True)
        artifact_type: Mapped[str] = mapped_column(String(64), nullable=False)
        file_path: Mapped[str] = mapped_column(Text, nullable=False)
        description: Mapped[str | None] = mapped_column(Text, nullable=True)
        created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

        experiment: Mapped["ExperimentRecord"] = relationship(back_populates="artifacts")

else:
    Base = None
    ExperimentRecord = None
    RoundMetricRecord = None
    ClientMetricRecord = None
    ArtifactRecord = None


def load_database_url() -> str:
    if load_dotenv is not None:
        load_dotenv(override=False)
    return os.getenv("DATABASE_URL", "").strip()


def create_database_session_factory():
    if not SQLALCHEMY_AVAILABLE:
        raise RuntimeError("SQLAlchemy is not available. Install database dependencies first.")

    database_url = load_database_url()
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured. Check your .env file.")

    engine = create_engine(database_url, future=True, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, future=True, expire_on_commit=False)


class ExperimentDatabaseRecorder:
    def __init__(
        self,
        run_config: dict[str, Any],
        mode: str,
        strategy_name: str | None = None,
    ) -> None:
        self.run_config = dict(run_config)
        self.mode = mode
        self.strategy_name = strategy_name
        self.enabled = False
        self.engine = None
        self.session_factory = None
        self.experiment_id: int | None = None

        if load_dotenv is not None:
            load_dotenv(override=False)

        database_url = os.getenv("DATABASE_URL", "").strip()
        if not database_url or not SQLALCHEMY_AVAILABLE:
            return

        self.engine = create_engine(database_url, future=True, pool_pre_ping=True)
        self.session_factory = sessionmaker(bind=self.engine, future=True, expire_on_commit=False)
        Base.metadata.create_all(self.engine)
        self.experiment_id = self._create_experiment_record()
        self.enabled = True

    def _create_experiment_record(self) -> int:
        metrics_path = Path(str(self.run_config["save-metrics-path"]))
        experiment_dir = metrics_path.parent.resolve()
        experiment_name = experiment_dir.name
        payload = ExperimentRecord(
            experiment_name=experiment_name,
            experiment_dir=str(experiment_dir),
            mode=self.mode,
            strategy_name=self.strategy_name,
            dataset_name=str(self.run_config.get("dataset-root", "")),
            model_name=str(self.run_config.get("model-name", "efficientnet_b0")),
            partition_mode=str(self.run_config.get("partition-mode", "")),
            dirichlet_alpha=self._float_or_none(self.run_config.get("dirichlet-alpha")),
            soft_mix_ratio=self._float_or_none(self.run_config.get("soft-mix-ratio")),
            soft_min_extra_classes=self._int_or_none(self.run_config.get("soft-min-extra-classes")),
            topology_mode=self._str_or_none(self.run_config.get("topology-mode")),
            topology_extra_offset=self._int_or_none(self.run_config.get("topology-extra-offset")),
            async_mode=bool(self.run_config.get("async-mode", False)),
            async_dropout_rate=self._float_or_none(self.run_config.get("async-dropout-rate")),
            max_async_dropouts=self._int_or_none(self.run_config.get("max-async-dropouts")),
            heterogeneous_nodes=bool(self.run_config.get("heterogeneous-nodes", False)),
            num_clients=int(self.run_config.get("num-clients", 0)),
            num_server_rounds=int(self.run_config.get("num-server-rounds", 0)),
            local_epochs=int(self.run_config.get("local-epochs", 0)),
            batch_size=int(self.run_config.get("batch-size", 0)),
            learning_rate=float(self.run_config.get("learning-rate", 0.0)),
            weight_decay=float(self.run_config.get("weight-decay", 0.0)),
            use_pretrained=bool(self.run_config.get("use-pretrained", False)),
            status="running",
            raw_config_json=json.dumps(self.run_config, ensure_ascii=True, sort_keys=True),
        )
        with self.session_factory() as session:
            session.add(payload)
            session.commit()
            session.refresh(payload)
            return int(payload.id)

    def record_round(
        self,
        round_number: int,
        aggregated_metrics: dict[str, float],
        client_reports: list[dict[str, Any]],
    ) -> None:
        if not self.enabled or self.experiment_id is None:
            return

        with self.session_factory() as session:
            round_row = self._get_or_create_round_row(session, round_number)
            round_row.train_loss = self._float_or_none(aggregated_metrics.get("train_loss"))
            round_row.train_accuracy = self._float_or_none(aggregated_metrics.get("train_accuracy"))
            round_row.train_f1 = self._float_or_none(aggregated_metrics.get("train_f1"))
            round_row.val_loss = self._float_or_none(aggregated_metrics.get("val_loss"))
            round_row.val_accuracy = self._float_or_none(aggregated_metrics.get("val_accuracy"))
            round_row.val_f1 = self._float_or_none(aggregated_metrics.get("val_f1"))
            round_row.participating_clients = self._float_or_none(aggregated_metrics.get("participating_clients"))
            round_row.skipped_clients = self._float_or_none(aggregated_metrics.get("skipped_clients"))

            session.query(ClientMetricRecord).filter_by(
                experiment_id=self.experiment_id,
                round_number=round_number,
            ).delete()

            for report in client_reports:
                session.add(
                    ClientMetricRecord(
                        experiment_id=self.experiment_id,
                        round_number=round_number,
                        client_name=str(report.get("client_id", "")),
                        partition_id=self._int_or_none(report.get("partition_id")),
                        num_examples=self._int_or_none(report.get("num_examples")),
                        trust_score=self._float_or_none(report.get("trust_score")),
                        resource_profile=self._str_or_none(report.get("resource_profile")),
                        resource_batch_size=self._int_or_none(report.get("resource_batch_size")),
                        virtual_delay_sec=self._float_or_none(report.get("virtual_delay_sec")),
                        train_loss=self._float_or_none(report.get("train_loss")),
                        train_accuracy=self._float_or_none(report.get("train_accuracy")),
                        train_f1=self._float_or_none(report.get("train_f1")),
                        val_loss=self._float_or_none(report.get("val_loss")),
                        val_accuracy=self._float_or_none(report.get("val_accuracy")),
                        val_f1=self._float_or_none(report.get("val_f1")),
                        train_time_sec=self._float_or_none(report.get("train_time_sec")),
                        update_l2_norm=self._float_or_none(report.get("update_l2_norm")),
                        skipped=bool(report.get("skipped", False)),
                        completed_round=self._int_or_none(report.get("completed_round")),
                        server_round=self._int_or_none(report.get("server_round")),
                    )
                )

            session.commit()

    def record_global_eval(
        self,
        round_number: int,
        loss: float,
        metrics: dict[str, float],
    ) -> None:
        if not self.enabled or self.experiment_id is None:
            return

        with self.session_factory() as session:
            round_row = self._get_or_create_round_row(session, round_number)
            round_row.global_loss = float(loss)
            round_row.global_accuracy = self._float_or_none(metrics.get("accuracy"))
            round_row.global_f1_macro = self._float_or_none(metrics.get("f1_macro"))
            round_row.global_precision_macro = self._float_or_none(metrics.get("precision_macro"))
            round_row.global_recall_macro = self._float_or_none(metrics.get("recall_macro"))

            experiment = session.get(ExperimentRecord, self.experiment_id)
            current_accuracy = self._float_or_none(metrics.get("accuracy"))
            current_f1 = self._float_or_none(metrics.get("f1_macro"))
            current_loss = float(loss)
            if experiment is not None:
                if experiment.best_accuracy is None or (current_accuracy is not None and current_accuracy > experiment.best_accuracy):
                    experiment.best_accuracy = current_accuracy
                if experiment.best_f1_macro is None or (current_f1 is not None and current_f1 > experiment.best_f1_macro):
                    experiment.best_f1_macro = current_f1
                if experiment.best_loss is None or current_loss < experiment.best_loss:
                    experiment.best_loss = current_loss

            session.commit()

    def register_artifact(self, artifact_type: str, file_path: Path | str, description: str | None = None) -> None:
        if not self.enabled or self.experiment_id is None:
            return

        artifact_path = Path(file_path)
        if not artifact_path.exists():
            return

        with self.session_factory() as session:
            existing = session.scalar(
                select(ArtifactRecord).where(
                    ArtifactRecord.experiment_id == self.experiment_id,
                    ArtifactRecord.file_path == str(artifact_path.resolve()),
                )
            )
            if existing is None:
                session.add(
                    ArtifactRecord(
                        experiment_id=self.experiment_id,
                        artifact_type=artifact_type,
                        file_path=str(artifact_path.resolve()),
                        description=description,
                    )
                )
                session.commit()

    def finalize(self, status: str, error_message: str | None = None) -> None:
        if not self.enabled or self.experiment_id is None:
            return

        with self.session_factory() as session:
            experiment = session.get(ExperimentRecord, self.experiment_id)
            if experiment is None:
                return
            experiment.status = status
            experiment.finished_at = _utcnow()
            experiment.error_message = error_message
            session.commit()

    def _get_or_create_round_row(self, session: Session, round_number: int) -> RoundMetricRecord:
        row = session.scalar(
            select(RoundMetricRecord).where(
                RoundMetricRecord.experiment_id == self.experiment_id,
                RoundMetricRecord.round_number == round_number,
            )
        )
        if row is None:
            row = RoundMetricRecord(
                experiment_id=int(self.experiment_id),
                round_number=round_number,
            )
            session.add(row)
            session.flush()
        return row

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        if value is None:
            return None
        return float(value)

    @staticmethod
    def _int_or_none(value: Any) -> int | None:
        if value is None:
            return None
        return int(value)

    @staticmethod
    def _str_or_none(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None
