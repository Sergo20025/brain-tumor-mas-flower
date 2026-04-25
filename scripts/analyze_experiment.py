from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_json_stream(text: str) -> list[dict]:
    decoder = json.JSONDecoder()
    rows: list[dict] = []
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        row, next_index = decoder.raw_decode(text, index)
        rows.append(row)
        index = next_index
    return rows


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.extend(_parse_json_stream(line))
    return rows


def select_last_run(rows: list[dict]) -> list[dict]:
    if not rows:
        return rows

    runs: list[list[dict]] = []
    current_run: list[dict] = []
    prev_round: int | None = None
    for row in rows:
        round_id = int(row["round"])
        if prev_round is not None and round_id <= prev_round:
            runs.append(current_run)
            current_run = []
        current_run.append(row)
        prev_round = round_id
    if current_run:
        runs.append(current_run)
    return runs[-1]


def build_report_text(
    experiment_dir: Path,
    round_rows: list[dict],
    eval_rows: list[dict],
    partition_mode: str,
) -> str:
    last_round = round_rows[-1]
    final_fit = last_round.get("aggregated_metrics", {})
    final_eval = eval_rows[-1]

    best_acc_row = max(eval_rows, key=lambda row: float(row.get("accuracy", 0.0)))
    best_f1_row = max(eval_rows, key=lambda row: float(row.get("f1_macro", 0.0)))
    best_loss_row = min(eval_rows, key=lambda row: float(row.get("loss", float("inf"))))

    mode = last_round.get("mode", "unknown")
    topology = last_round.get("topology_mode", "")
    strategy = last_round.get("strategy_name", "")

    title_parts = [f"Эксперимент: {experiment_dir.name}"]
    title_parts.append(f"Режим: {mode}")
    if topology:
        title_parts.append(f"Топология: {topology}")
    if strategy:
        title_parts.append(f"Стратегия: {strategy}")
    title_parts.append(f"Режим разбиения: {partition_mode}")

    lines = [
        "\n".join(title_parts),
        "",
        "Итог по последнему раунду",
        (
            f"Раунд {int(final_eval['round'])}: "
            f"loss={float(final_eval['loss']):.4f}, "
            f"accuracy={float(final_eval['accuracy']):.4f}, "
            f"f1_macro={float(final_eval['f1_macro']):.4f}"
        ),
        (
            f"Локальные агрегированные метрики: "
            f"train_loss={float(final_fit.get('train_loss', 0.0)):.4f}, "
            f"train_accuracy={float(final_fit.get('train_accuracy', 0.0)):.4f}, "
            f"val_loss={float(final_fit.get('val_loss', 0.0)):.4f}, "
            f"val_accuracy={float(final_fit.get('val_accuracy', 0.0)):.4f}, "
            f"val_f1={float(final_fit.get('val_f1', 0.0)):.4f}"
        ),
        "",
        "Лучшие значения",
        (
            f"Лучшая accuracy: round {int(best_acc_row['round'])}, "
            f"value={float(best_acc_row['accuracy']):.4f}"
        ),
        (
            f"Лучшая f1_macro: round {int(best_f1_row['round'])}, "
            f"value={float(best_f1_row['f1_macro']):.4f}"
        ),
        (
            f"Минимальный loss: round {int(best_loss_row['round'])}, "
            f"value={float(best_loss_row['loss']):.4f}"
        ),
        "",
        "Краткий анализ",
    ]

    start_eval = eval_rows[0]
    end_eval = eval_rows[-1]
    acc_gain = float(end_eval["accuracy"]) - float(start_eval["accuracy"])
    f1_gain = float(end_eval["f1_macro"]) - float(start_eval["f1_macro"])
    lines.append(
        (
            f"Глобальное качество изменилось от accuracy={float(start_eval['accuracy']):.4f}, "
            f"f1_macro={float(start_eval['f1_macro']):.4f} "
            f"до accuracy={float(end_eval['accuracy']):.4f}, "
            f"f1_macro={float(end_eval['f1_macro']):.4f}."
        )
    )
    lines.append(
        f"Прирост составил {acc_gain:.4f} по accuracy и {f1_gain:.4f} по f1_macro."
    )

    if int(best_f1_row["round"]) < int(end_eval["round"]):
        lines.append(
            (
                f"Лучшее значение f1_macro было достигнуто до конца обучения "
                f"(на раунде {int(best_f1_row['round'])}), что может указывать "
                f"на выход модели на плато."
            )
        )
    else:
        lines.append(
            "Лучшее значение f1_macro достигнуто в конце обучения, что указывает на продолжающуюся сходимость."
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--partition-mode", default="dirichlet")
    parser.add_argument("--output-name", default="experiment_analysis.txt")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    round_rows = select_last_run(load_jsonl(experiment_dir / "round_metrics.jsonl"))
    eval_rows = select_last_run(load_jsonl(experiment_dir / "global_eval_metrics.jsonl"))
    report_text = build_report_text(
        experiment_dir=experiment_dir,
        round_rows=round_rows,
        eval_rows=eval_rows,
        partition_mode=args.partition_mode,
    )
    output_path = experiment_dir / args.output_name
    output_path.write_text(report_text, encoding="utf-8")
    print(f"Saved analysis to: {output_path}")


if __name__ == "__main__":
    main()
