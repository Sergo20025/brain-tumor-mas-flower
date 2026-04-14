# Brain Tumor MRI MAS + Flower

Проект реализует мультиагентную систему федеративного обучения для классификации MRI-изображений опухолей головного мозга с использованием Flower.

Архитектура основана на трех основных ролях:

- `StorageAgent`: загружает датасет, формирует глобальный test split и клиентские non-IID partitions.
- `ComputeAgent`: обучает локальную модель `EfficientNet-B0` и возвращает метрики/обновления.
- `AggregationAgent` + `MonitoringAgent`: агрегируют локальные веса и адаптивно взвешивают вклад клиентов по качеству и стабильности обновлений.

## Ожидаемая структура датасета

Поддерживаются два формата:

1. Один каталог с классами:

```text
brain_tumor_mri/
  glioma/
  meningioma/
  notumor/
  pituitary/
```

2. Разделенные каталоги train/test:

```text
brain_tumor_mri/
  train/
    glioma/
    meningioma/
    notumor/
    pituitary/
  test/
    glioma/
    meningioma/
    notumor/
    pituitary/
```

Также распознаются каталоги `Training`/`Testing`.

## Установка

```bash
pip install -e .
```

## Запуск Flower simulation

```bash
flwr run .
```

## Запуск обучения с логами сразу в терминал

```powershell
.\run_training.ps1
```

Пример с явными параметрами:

```powershell
.\run_training.ps1 -NumServerRounds 3 -NumClients 10 -UsePretrained $false
```

## Пример запуска с коротким экспериментом

```bash
flwr run . --run-config "dataset-root='brain_tumor_mri' num-server-rounds=3 num-clients=10 partition-mode='dirichlet' dirichlet-alpha=0.5 decentralized-mode=true"
```

Для Windows-консоли при проблемах с кодировкой удобно запускать так:

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
flwr run .
```

## Что делает `decentralized-mode`

Это не чистый peer-to-peer режим. В рамках ограничений Flower включается псевдодецентрализованная агентная логика:

- учитывается доверительный коэффициент клиента;
- медленные или нестабильные обновления получают меньший вес;
- агрегация учитывает не только размер локальной выборки, но и локальные метрики качества.

## Основные файлы

- `brain_tumor_fl/client_app.py`: определение `ClientApp`
- `brain_tumor_fl/server_app.py`: определение `ServerApp`
- `brain_tumor_fl/strategy.py`: кастомная стратегия агрегации
- `brain_tumor_fl/agents/`: агентные роли
- `brain_tumor_fl/data.py`: загрузка датасета и разбиение на 10 клиентов
- `brain_tumor_fl/model.py`: `EfficientNet-B0`

## Troubleshooting

- Если `flwr run .` завершается с `Exit Code 701`, значит в окружении отсутствует `ray`, который Flower использует для локальной simulation.
- В этой среде приложение и агенты успешно проверены импортом, созданием `ServerApp/ClientApp` и локальным `fit/evaluate`, но полный `flwr run` зависит от наличия совместимого `ray` backend.
