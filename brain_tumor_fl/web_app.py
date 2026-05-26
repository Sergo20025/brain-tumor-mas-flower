from __future__ import annotations

import base64
import hmac
import html
import json
import os
import tempfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from fastapi import Cookie, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from PIL import Image
from torchvision import transforms

from brain_tumor_fl.data import IMAGENET_MEAN, IMAGENET_STD
from brain_tumor_fl.model import build_model
from brain_tumor_fl.training import get_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_EXPERIMENT_DIR = (
    OUTPUTS_DIR
    / "experiments"
    / "decentralized"
    / "exp_decentralized_augmented_ring_async_heterogeneous_mri_r60_d20"
)
DEFAULT_CHECKPOINT = OUTPUTS_DIR / "checkpoints" / "best_model.pt"
HISTORY_PATH = OUTPUTS_DIR / "web_prediction_history.jsonl"
STATIC_DIR = Path(__file__).resolve().parent / "static"
MISIS_LOGO_PATH = STATIC_DIR / "misis_logo.png"

WEB_USERNAME = os.getenv("BRAIN_TUMOR_WEB_USER", "admin")
WEB_PASSWORD = os.getenv("BRAIN_TUMOR_WEB_PASSWORD", "brain2026")
SESSION_SECRET = os.getenv("BRAIN_TUMOR_WEB_SECRET", "brain-tumor-demo-secret")

DISPLAY_NAMES = {
    "glioma": "Глиома",
    "meningioma": "Менингиома",
    "pituitary": "Опухоль гипофиза",
    "notumor": "Опухоль не обнаружена",
}

SAMPLE_IMAGES = {
    "glioma": PROJECT_ROOT / "brain_tumor_mri" / "Testing" / "glioma" / "Te-gl_1.jpg",
    "meningioma": PROJECT_ROOT
    / "brain_tumor_mri"
    / "Testing"
    / "meningioma"
    / "Te-aug-me_1.jpg",
    "pituitary": PROJECT_ROOT / "brain_tumor_mri" / "Testing" / "pituitary" / "Te-pi_1.jpg",
    "notumor": PROJECT_ROOT / "brain_tumor_mri" / "Testing" / "notumor" / "Te-no_1.jpg",
}

EXPERIMENT_SUMMARY = [
    ("Brain Tumor MRI", "ring", 30, 0.8187, 0.8122, 0.5801),
    ("Brain Tumor MRI", "augmented_ring", 30, 0.8381, 0.8334, 0.5366),
    ("Brain Tumor MRI", "full_graph", 30, 0.8450, 0.8388, 0.4958),
    ("Brain Tumor MRI", "augmented_ring без агентной логики", 30, 0.8519, 0.8466, 0.4542),
    ("Brain Tumor MRI", "augmented_ring, async + heterogeneous", 60, 0.7963, 0.7849, 0.9173),
    ("CIFAR-100", "augmented_ring", 30, 0.2088, 0.1601, 3.2170),
    ("CIFAR-100", "full_graph", 30, 0.3230, 0.2898, 2.5807),
]

app = FastAPI(title="Brain Tumor MRI Federated Learning Demo")


def page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(
        f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --blue: #0b2a66;
      --accent: #0647ff;
      --cyan: #22c7dd;
      --green: #0f9f68;
      --pink: #e91e82;
      --bg: #f4f7fb;
      --line: #dbe5f2;
      --text: #14213d;
      --muted: #667085;
      --panel: rgba(255,255,255,.95);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      color: var(--text);
      background:
        linear-gradient(90deg, rgba(244,247,251,.98), rgba(244,247,251,.9)),
        url("/asset/background-mri") center / cover fixed;
    }}
    a {{ color: var(--accent); text-decoration: none; }}
    .topbar {{
      min-height: 74px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 22px;
      padding: 10px 32px;
      background: rgba(255,255,255,.97);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(8px);
      position: sticky;
      top: 0;
      z-index: 20;
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 14px;
      min-width: 280px;
    }}
    .misis-logo {{
      width: 224px;
      height: auto;
      display: block;
    }}
    .dept {{
      display: flex;
      align-items: center;
      gap: 10px;
      color: #111827;
      font-size: 13px;
      line-height: 1.15;
      min-width: 230px;
      justify-content: flex-end;
    }}
    .dept-grid {{
      display: grid;
      grid-template-columns: repeat(3, 18px);
      gap: 3px;
      font-weight: 800;
      color: #fff;
      text-align: center;
      line-height: 18px;
      font-size: 12px;
    }}
    .dept-grid span {{ border-radius: 50%; background: #84c318; }}
    .dept-grid span:nth-child(1) {{ background: var(--pink); }}
    .dept-grid span:nth-child(2),
    .dept-grid span:nth-child(4),
    .dept-grid span:nth-child(5) {{ background: #20aeea; }}
    .nav {{ display: flex; gap: 18px; align-items: center; font-size: 15px; white-space: nowrap; }}
    .wrap {{ max-width: 1220px; margin: 0 auto; padding: 30px 24px 44px; }}
    .hero {{
      display: grid;
      grid-template-columns: 1.15fr .85fr;
      gap: 22px;
      align-items: stretch;
      margin-bottom: 22px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 12px 34px rgba(11, 42, 102, .08);
    }}
    .pad {{ padding: 24px; }}
    h1 {{ margin: 0 0 10px; font-size: 32px; color: var(--blue); }}
    h2 {{ margin: 0 0 16px; font-size: 22px; color: var(--blue); }}
    h3 {{ margin: 0 0 8px; font-size: 17px; color: var(--blue); }}
    p {{ margin: 0 0 12px; line-height: 1.5; }}
    ul {{ margin: 10px 0 0; padding-left: 20px; }}
    li {{ margin: 7px 0; }}
    .muted {{ color: var(--muted); }}
    .badge {{
      display: inline-block;
      padding: 5px 9px;
      border-radius: 999px;
      background: #eaf2ff;
      color: var(--blue);
      font-weight: 700;
      font-size: 13px;
      margin: 3px 4px 3px 0;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin: 18px 0 22px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }}
    .card .label {{ color: var(--muted); font-size: 13px; margin-bottom: 8px; }}
    .card .value {{ color: var(--blue); font-weight: 800; font-size: 26px; }}
    .grid2 {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    .plot {{ width: 100%; display: block; border-radius: 6px; border: 1px solid var(--line); background: #fff; }}
    .login {{ min-height: calc(100vh - 74px); display: grid; place-items: center; padding: 24px; }}
    .login-card {{ width: min(460px, 100%); }}
    label {{ display: block; font-weight: 700; color: var(--blue); margin: 12px 0 6px; }}
    input[type="text"], input[type="password"], input[type="file"] {{
      width: 100%;
      padding: 12px 13px;
      border: 1px solid #c9d6e8;
      border-radius: 7px;
      font: inherit;
      background: #fff;
    }}
    button, .button {{
      border: 0;
      border-radius: 7px;
      background: var(--accent);
      color: #fff;
      font-weight: 800;
      padding: 12px 18px;
      cursor: pointer;
      font: inherit;
      display: inline-block;
    }}
    .button.secondary {{ background: #eaf2ff; color: var(--blue); }}
    .button-row {{ margin-top: 16px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
    .result {{ display: grid; grid-template-columns: 320px 1fr; gap: 22px; margin-top: 18px; align-items: start; }}
    .preview {{ width: 100%; max-height: 340px; object-fit: contain; border-radius: 8px; border: 1px solid var(--line); background: #111827; }}
    .prediction-box {{ border: 2px solid rgba(20,164,108,.35); background: #f0fff8; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 11px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ color: var(--blue); background: #f7faff; }}
    .bar-bg {{ height: 10px; background: #e7eef8; border-radius: 999px; overflow: hidden; min-width: 120px; }}
    .bar {{ height: 100%; background: var(--cyan); }}
    .warn {{ color: #b42318; font-weight: 700; }}
    .note {{ border-left: 4px solid var(--cyan); background: #f0fbff; padding: 13px 15px; border-radius: 6px; margin-top: 16px; }}
    @media (max-width: 980px) {{
      .topbar {{ align-items: flex-start; flex-direction: column; }}
      .dept {{ justify-content: flex-start; }}
      .hero, .grid2, .result {{ grid-template-columns: 1fr; }}
      .cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>{body}</body>
</html>"""
    )


def header() -> str:
    return """
<div class="topbar">
  <div class="brand">
    <img class="misis-logo" src="/asset/misis-logo" alt="МИСИС Университет">
  </div>
  <div class="nav">
    <a href="/about">О системе</a>
    <a href="/dashboard">Метрики</a>
    <a href="/predict">Предсказание</a>
    <a href="/history">История</a>
    <a href="/logout">Выход</a>
  </div>
  <div class="dept">
    <div class="dept-grid" aria-hidden="true">
      <span>+</span><span>0</span><span>1</span>
      <span>0</span><span>0</span><span>1</span>
      <span>1</span><span>1</span><span>1</span>
    </div>
    <div>Кафедра<br>инженерной<br>кибернетики</div>
  </div>
</div>
"""


def sign_session(value: str) -> str:
    digest = hmac.new(SESSION_SECRET.encode(), value.encode(), "sha256").hexdigest()
    return f"{value}.{digest}"


def verify_session(session: str | None) -> bool:
    if not session or "." not in session:
        return False
    value, digest = session.rsplit(".", 1)
    expected = hmac.new(SESSION_SECRET.encode(), value.encode(), "sha256").hexdigest()
    return value == WEB_USERNAME and hmac.compare_digest(digest, expected)


def require_login(session: str | None) -> None:
    if not verify_session(session):
        raise HTTPException(status_code=307, headers={"Location": "/login"})


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def metric_value(row: dict[str, Any], *names: str) -> float | None:
    metrics = row.get("aggregated_metrics", row)
    for name in names:
        value = metrics.get(name)
        if isinstance(value, int | float):
            return float(value)
    return None


def fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def latest_metrics() -> dict[str, Any]:
    global_rows = read_jsonl(DEFAULT_EXPERIMENT_DIR / "global_eval_metrics.jsonl")
    round_rows = read_jsonl(DEFAULT_EXPERIMENT_DIR / "round_metrics.jsonl")
    global_latest = global_rows[-1] if global_rows else {}
    round_latest = round_rows[-1] if round_rows else {}
    return {
        "round": global_latest.get("round") or round_latest.get("round") or "—",
        "accuracy": metric_value(global_latest, "accuracy", "test_accuracy"),
        "f1": metric_value(global_latest, "f1_macro", "test_f1"),
        "loss": metric_value(global_latest, "loss", "test_loss"),
        "train_loss": metric_value(round_latest, "train_loss"),
        "train_accuracy": metric_value(round_latest, "train_accuracy"),
        "participating": metric_value(round_latest, "participating_clients"),
        "skipped": metric_value(round_latest, "skipped_clients"),
    }


def experiment_table() -> str:
    rows = "\n".join(
        f"""
<tr>
  <td>{dataset}</td>
  <td>{scenario}</td>
  <td>{round_number}</td>
  <td>{accuracy:.4f}</td>
  <td>{f1:.4f}</td>
  <td>{loss:.4f}</td>
</tr>"""
        for dataset, scenario, round_number, accuracy, f1, loss in EXPERIMENT_SUMMARY
    )
    return f"""
<table>
  <thead>
    <tr><th>Датасет</th><th>Сценарий</th><th>Раунд</th><th>Accuracy</th><th>F1-macro</th><th>Loss</th></tr>
  </thead>
  <tbody>{rows}</tbody>
</table>
"""


def available_plot(filename: str) -> bool:
    return (DEFAULT_EXPERIMENT_DIR / "plots" / filename).exists()


def sample_path(class_name: str) -> Path:
    path = SAMPLE_IMAGES.get(class_name)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Sample image was not found")
    return path


def background_mri_path() -> Path:
    for class_name in ("glioma", "pituitary", "notumor", "meningioma"):
        path = SAMPLE_IMAGES.get(class_name)
        if path and path.exists():
            return path
    raise HTTPException(status_code=404, detail="Background image was not found")


def inference_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


@lru_cache(maxsize=1)
def load_model() -> tuple[torch.nn.Module, list[str], str | int]:
    if not DEFAULT_CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint was not found: {DEFAULT_CHECKPOINT}")
    checkpoint = torch.load(DEFAULT_CHECKPOINT, map_location="cpu")
    classes = list(checkpoint.get("classes") or ["glioma", "meningioma", "notumor", "pituitary"])
    model = build_model(
        num_classes=len(classes),
        use_pretrained=bool(checkpoint.get("use_pretrained", False)),
        model_name=str(checkpoint.get("model_name", "efficientnet_b0")),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(get_device())
    model.eval()
    return model, classes, checkpoint.get("round", "unknown")


def image_to_data_url(path: Path) -> str:
    with Image.open(path) as image:
        image.thumbnail((760, 480))
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            out_path = Path(tmp.name)
        image.convert("RGB").save(out_path, format="JPEG", quality=90)
    data = base64.b64encode(out_path.read_bytes()).decode("ascii")
    out_path.unlink(missing_ok=True)
    return f"data:image/jpeg;base64,{data}"


def predict(path: Path, source_name: str | None = None, save_history: bool = True) -> dict[str, Any]:
    model, classes, checkpoint_round = load_model()
    image = Image.open(path).convert("RGB")
    tensor = inference_transform()(image).unsqueeze(0).to(get_device())
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().tolist()
    rows = sorted(zip(classes, probabilities, strict=True), key=lambda item: item[1], reverse=True)
    predicted_class, confidence = rows[0]
    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source_name": source_name or path.name,
        "predicted_class": predicted_class,
        "display_name": DISPLAY_NAMES.get(predicted_class, predicted_class),
        "confidence": float(confidence),
        "probabilities": [
            {
                "class_name": class_name,
                "display_name": DISPLAY_NAMES.get(class_name, class_name),
                "probability": float(probability),
            }
            for class_name, probability in rows
        ],
        "checkpoint_round": checkpoint_round,
    }
    if save_history:
        append_jsonl(HISTORY_PATH, result)
    return result


def render_result(result: dict[str, Any], image_data_url: str) -> str:
    probability_rows = "\n".join(
        f"""
<tr>
  <td>{row["display_name"]}<br><span class="muted">{row["class_name"]}</span></td>
  <td>{row["probability"] * 100:.2f}%</td>
  <td><div class="bar-bg"><div class="bar" style="width:{row["probability"] * 100:.1f}%"></div></div></td>
</tr>"""
        for row in result["probabilities"]
    )
    probability_cards = "\n".join(
        f"""
<div class="card">
  <div class="label">{row["display_name"]}</div>
  <div class="value" style="font-size:22px">{row["probability"] * 100:.1f}%</div>
</div>"""
        for row in result["probabilities"]
    )
    return f"""
<div class="result">
  <img class="preview" src="{image_data_url}" alt="Загруженное MRI-изображение">
  <div>
    <div class="prediction-box">
      <h2>Результат: {result["display_name"]}</h2>
      <p><b>Уверенность:</b> {result["confidence"] * 100:.2f}%</p>
      <p><b>Класс модели:</b> {result["predicted_class"]}</p>
      <p><b>Раунд checkpoint:</b> {result["checkpoint_round"]}</p>
    </div>
    <div class="cards" style="grid-template-columns:repeat(2,minmax(0,1fr));margin-top:0">{probability_cards}</div>
    <table>
      <thead><tr><th>Класс</th><th>Вероятность</th><th></th></tr></thead>
      <tbody>{probability_rows}</tbody>
    </table>
  </div>
</div>
"""


@app.get("/", response_class=HTMLResponse)
def root(session: str | None = Cookie(default=None)):
    return RedirectResponse("/dashboard" if verify_session(session) else "/login")


@app.get("/login", response_class=HTMLResponse)
def login_page(session: str | None = Cookie(default=None)):
    if verify_session(session):
        return RedirectResponse("/dashboard")
    return page(
        "Вход",
        f"""
<div class="login">
  <div class="panel login-card pad">
    <div class="brand" style="margin-bottom:18px">
      <img class="misis-logo" src="/asset/misis-logo" alt="МИСИС Университет">
    </div>
    <h1>Демо веб-сервиса</h1>
    <p class="muted">Классификация опухолей головного мозга по MRI-изображениям с использованием федеративного и децентрализованного обучения.</p>
    <form method="post" action="/login">
      <label>Логин</label>
      <input type="text" name="username" autocomplete="username" autofocus>
      <label>Пароль</label>
      <input type="password" name="password" autocomplete="current-password">
      <div class="button-row">
        <button type="submit">Войти</button>
        <span class="muted">demo: {html.escape(WEB_USERNAME)} / {html.escape(WEB_PASSWORD)}</span>
      </div>
    </form>
  </div>
</div>
""",
    )


@app.post("/login")
def login_submit(username: str = Form(...), password: str = Form(...)):
    if username != WEB_USERNAME or password != WEB_PASSWORD:
        return RedirectResponse("/login?error=1", status_code=303)
    response = RedirectResponse("/dashboard", status_code=303)
    response.set_cookie("session", sign_session(username), httponly=True, samesite="lax")
    return response


@app.get("/logout")
def logout():
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie("session")
    return response


@app.get("/about", response_class=HTMLResponse)
def about(session: str | None = Cookie(default=None)):
    require_login(session)
    return page(
        "О системе",
        header()
        + """
<main class="wrap">
  <section class="hero">
    <div class="panel pad">
      <h1>Система классификации Brain Tumor MRI</h1>
      <p>Веб-сервис демонстрирует полный цикл работы разработанной системы: просмотр результатов распределенного обучения и классификацию нового MRI-изображения по сохраненному checkpoint модели.</p>
      <p>В основе используется EfficientNet-B0, а федеративный сценарий организован с помощью Flower. Распределенная среда эмулируется программно на 10 вычислительных узлах.</p>
      <div>
        <span class="badge">Flower</span>
        <span class="badge">PyTorch</span>
        <span class="badge">EfficientNet-B0</span>
        <span class="badge">10 узлов</span>
        <span class="badge">non-IID</span>
        <span class="badge">augmented_ring</span>
      </div>
    </div>
    <div class="panel pad">
      <h2>Ключевая идея</h2>
      <p>Исходные изображения остаются на локальных узлах. Между участниками передаются параметры модели и промежуточные обновления, а не сами медицинские данные.</p>
      <p class="note">Сервис является демонстрационным и не предназначен для постановки медицинского диагноза.</p>
    </div>
  </section>
  <section class="grid2">
    <div class="panel pad">
      <h2>Агенты системы</h2>
      <ul>
        <li><b>StorageAgent</b> — агент хранения и подготовки данных: загрузка датасета, preprocessing и разбиение выборки между узлами.</li>
        <li><b>ComputeAgent</b> — вычислительный агент: локальное обучение модели на данных конкретного узла.</li>
        <li><b>MonitoringAgent</b> — агент мониторинга: расчет метрик, нормы обновления и доверительной оценки trust-score.</li>
        <li><b>AggregationAgent</b> — агент агрегации: взвешенное объединение параметров по выбранной топологии.</li>
      </ul>
    </div>
    <div class="panel pad">
      <h2>Что можно показать</h2>
      <ul>
        <li>вход в сервис по логину и паролю;</li>
        <li>графики accuracy, F1-macro и loss;</li>
        <li>таблицу результатов экспериментов;</li>
        <li>загрузку MRI-изображения и результат классификации;</li>
        <li>историю выполненных предсказаний и API-ответ.</li>
      </ul>
    </div>
  </section>
</main>
""",
    )


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(session: str | None = Cookie(default=None)):
    require_login(session)
    metrics = latest_metrics()
    plots = [
        ("F1-macro", "f1_macro_dynamics_non_iid.png"),
        ("Accuracy", "accuracy_dynamics_non_iid.png"),
        ("Loss", "loss_dynamics_non_iid.png"),
        ("Распределение классов", "class_distribution_shards_quantity_skew.png"),
    ]
    plot_cards = "\n".join(
        f"""
<div class="panel pad">
  <h3>{title}</h3>
  {"<img class='plot' src='/plot/" + filename + "' alt='" + title + "'>" if available_plot(filename) else "<p class='muted'>График не найден.</p>"}
</div>"""
        for title, filename in plots
    )
    return page(
        "Метрики обучения",
        header()
        + f"""
<main class="wrap">
  <section class="hero">
    <div class="panel pad">
      <h1>Метрики распределенного обучения</h1>
      <p>Панель показывает сохраненные результаты эксперимента Brain Tumor MRI в децентрализованной постановке.</p>
      <p class="muted">Сценарий: 10 узлов, non-IID разбиение, топология augmented_ring, асинхронность и гетерогенность вычислительных узлов.</p>
      <div>
        <span class="badge">shards_quantity_skew</span>
        <span class="badge">async</span>
        <span class="badge">fast / medium / slow</span>
      </div>
    </div>
    <div class="panel pad">
      <h2>Параметры демо</h2>
      <p><b>Модель:</b> EfficientNet-B0</p>
      <p><b>Checkpoint:</b> outputs/checkpoints/best_model.pt</p>
      <p><b>Классы:</b> glioma, meningioma, pituitary, notumor</p>
    </div>
  </section>
  <section class="cards">
    <div class="card"><div class="label">Раунд</div><div class="value">{metrics["round"]}</div></div>
    <div class="card"><div class="label">Test accuracy</div><div class="value">{fmt(metrics["accuracy"])}</div></div>
    <div class="card"><div class="label">Test F1-macro</div><div class="value">{fmt(metrics["f1"])}</div></div>
    <div class="card"><div class="label">Test loss</div><div class="value">{fmt(metrics["loss"])}</div></div>
  </section>
  <section class="panel pad" style="margin-bottom:18px">
    <h2>Итоговые результаты экспериментов</h2>
    {experiment_table()}
  </section>
  <section class="grid2">{plot_cards}</section>
</main>
""",
    )


@app.get("/plot/{filename}")
def plot_file(filename: str, session: str | None = Cookie(default=None)):
    require_login(session)
    if "/" in filename or "\\" in filename:
        raise HTTPException(status_code=404)
    path = DEFAULT_EXPERIMENT_DIR / "plots" / filename
    if not path.exists():
        raise HTTPException(status_code=404)
    return FileResponse(path)


@app.get("/asset/background-mri")
def background_mri():
    return FileResponse(background_mri_path())


@app.get("/asset/misis-logo")
def misis_logo():
    return FileResponse(MISIS_LOGO_PATH)


@app.get("/asset/sample/{class_name}")
def sample_asset(class_name: str, session: str | None = Cookie(default=None)):
    require_login(session)
    return FileResponse(sample_path(class_name))


@app.get("/predict", response_class=HTMLResponse)
def predict_page(session: str | None = Cookie(default=None)):
    require_login(session)
    return render_predict()


@app.get("/predict/sample/{class_name}", response_class=HTMLResponse)
def predict_sample(class_name: str, session: str | None = Cookie(default=None)):
    require_login(session)
    path = sample_path(class_name)
    result = predict(path, source_name=f"sample:{class_name}")
    return render_predict(result=result, image_data_url=image_to_data_url(path))


def render_predict(
    result: dict[str, Any] | None = None,
    image_data_url: str | None = None,
    error: str | None = None,
) -> HTMLResponse:
    sample_buttons = "\n".join(
        f"<a class='button secondary' href='/predict/sample/{class_name}'>{DISPLAY_NAMES[class_name]}</a>"
        for class_name in ("glioma", "meningioma", "pituitary", "notumor")
    )
    result_html = ""
    if error:
        result_html = f"<p class='warn'>{html.escape(error)}</p>"
    elif result and image_data_url:
        result_html = render_result(result, image_data_url)
    return page(
        "Предсказание",
        header()
        + f"""
<main class="wrap">
  <section class="panel pad">
    <h1>Классификация MRI-изображения</h1>
    <p class="muted">Сервис выполняет preprocessing 224×224, загружает checkpoint модели и показывает вероятности по классам через softmax.</p>
    <form method="post" action="/predict" enctype="multipart/form-data">
      <label>Загрузить свое изображение</label>
      <input type="file" name="image" accept="image/*" required>
      <div class="button-row"><button type="submit">Получить результат</button></div>
    </form>
    <div class="note">
      <b>Быстрая демонстрация:</b>
      <div class="button-row" style="margin-top:10px">{sample_buttons}</div>
    </div>
    <p class="muted" style="margin-top:14px">Важно: результат является демонстрацией работы модели и не является медицинским диагнозом.</p>
    {result_html}
  </section>
</main>
""",
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict_submit(
    image: UploadFile = File(...),
    session: str | None = Cookie(default=None),
):
    require_login(session)
    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await image.read())
        upload_path = Path(tmp.name)
    try:
        result = predict(upload_path, source_name=image.filename)
        image_data_url = image_to_data_url(upload_path)
        return render_predict(result=result, image_data_url=image_data_url)
    except Exception as exc:
        return render_predict(error=f"Не удалось выполнить предсказание: {exc}")
    finally:
        upload_path.unlink(missing_ok=True)


@app.post("/api/predict")
async def api_predict(image: UploadFile = File(...)):
    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await image.read())
        upload_path = Path(tmp.name)
    try:
        return JSONResponse(predict(upload_path, source_name=image.filename))
    finally:
        upload_path.unlink(missing_ok=True)


@app.get("/api/health")
def api_health():
    return {
        "status": "ok",
        "checkpoint": str(DEFAULT_CHECKPOINT.relative_to(PROJECT_ROOT)),
        "experiment": str(DEFAULT_EXPERIMENT_DIR.relative_to(PROJECT_ROOT)),
    }


@app.get("/history", response_class=HTMLResponse)
def history(session: str | None = Cookie(default=None)):
    require_login(session)
    rows = list(reversed(read_jsonl(HISTORY_PATH)[-20:]))
    if rows:
        body_rows = "\n".join(
            f"""
<tr>
  <td>{html.escape(str(row.get("timestamp", "—")))}</td>
  <td>{html.escape(str(row.get("source_name", "—")))}</td>
  <td>{html.escape(DISPLAY_NAMES.get(str(row.get("predicted_class", "")), str(row.get("display_name", "—"))))}</td>
  <td>{float(row.get("confidence", 0.0)) * 100:.2f}%</td>
</tr>"""
            for row in rows
        )
        table = f"""
<table>
  <thead><tr><th>Время</th><th>Файл</th><th>Результат</th><th>Уверенность</th></tr></thead>
  <tbody>{body_rows}</tbody>
</table>"""
    else:
        table = "<p class='muted'>История пока пустая. Выполните предсказание на странице классификации.</p>"
    return page(
        "История предсказаний",
        header()
        + f"""
<main class="wrap">
  <section class="panel pad">
    <h1>История предсказаний</h1>
    <p class="muted">Здесь отображаются последние результаты, полученные через веб-интерфейс или API.</p>
    {table}
  </section>
</main>
""",
    )
