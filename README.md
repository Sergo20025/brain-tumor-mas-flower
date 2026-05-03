# Brain Tumor MRI Federated Learning System

Проект реализует систему федеративного и децентрализованного обучения для классификации опухолей головного мозга по МРТ-изображениям. В репозитории есть как основной медицинский сценарий `Brain Tumor MRI`, так и дополнительные benchmark-эксперименты на `CIFAR-10` и `CIFAR-100`.

Система поддерживает:

- федеративное обучение через Flower;
- децентрализованное обучение без центрального агрегатора;
- топологии `ring`, `augmented_ring`, `full_graph`;
- асинхронную агрегацию;
- временное выпадение узлов;
- гетерогенность данных и вычислительных узлов;
- сохранение метрик и артефактов экспериментов в `Neon/PostgreSQL`.

## Основные возможности

- Классификация MRI-изображений по 4 классам:
  - `glioma`
  - `meningioma`
  - `notumor`
  - `pituitary`
- Поддержка `EfficientNet-B0` и `ResNet-50`
- Разбиения данных:
  - `iid`
  - `quantity_skew`
  - `shards_quantity_skew`
  - `shards_quantity_skew_soft`
  - `dirichlet`
- Автоматическое построение графиков и текстового анализа эксперимента
- Импорт уже завершенных экспериментов в БД

## Архитектура

Ключевые модули системы:

- `StorageAgent`  
  Загружает датасет и формирует локальные выборки клиентов.

- `ComputeAgent`  
  Выполняет локальное обучение модели и возвращает обновления и метрики.

- `MonitoringAgent`  
  Оценивает качество локального обновления и вычисляет `trust-score`.

- `AggregationAgent`  
  Выполняет локальную агрегацию модели с учетом топологии и доверительных весов.

Главные файлы:

- `brain_tumor_fl/decentralized_simulation.py`  
  Основной сценарий децентрализованного обучения
- `brain_tumor_fl/server_simulation.py`  
  Централизованный / федеративный сценарий
- `brain_tumor_fl/decentralized_baseline.py`  
  Базовый децентрализованный режим без агентной логики
- `brain_tumor_fl/data.py`  
  Подготовка данных и разбиений
- `brain_tumor_fl/model.py`  
  Модели `efficientnet_b0` и `resnet50`
- `brain_tumor_fl/db.py`  
  Интеграция с БД для хранения экспериментов и метрик

## Поддерживаемые датасеты

### 1. Brain Tumor MRI

Ожидаемая структура:

```text
brain_tumor_mri/
  glioma/
  meningioma/
  notumor/
  pituitary/
```

или:

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

Также поддерживаются каталоги `Training` / `Testing`.

### 2. CIFAR-10

Используется как дополнительный benchmark для проверки поведения системы в 10-классовом сценарии.

### 3. CIFAR-100

Используется как более сложный benchmark и stress-test для async / heterogeneous / non-IID постановки.

## Установка

### Быстрый вариант

```bash
pip install -e .
```

### Через conda

Подробности см. в [SETUP_CONDA.md](SETUP_CONDA.md).

## Основные сценарии запуска

### 1. Децентрализованный MRI-эксперимент

```powershell
.\scripts\run_exp_decentralized_augmented_ring.ps1
```

### 2. MRI async + heterogeneous

```powershell
.\scripts\run_exp_decentralized_augmented_ring_async_heterogeneous.ps1
```

### 3. CIFAR-10 async + heterogeneous

```powershell
.\scripts\run_exp_cifar10_augmented_ring_async_heterogeneous_r70.ps1
```

### 4. CIFAR-100 async + heterogeneous + soft partition

```powershell
.\scripts\run_exp_cifar100_augmented_ring_async_heterogeneous_soft20_r70.ps1
```

### 5. Универсальный запуск с параметрами

```powershell
.\scripts\run_decentralized_experiment.ps1 `
  -ExperimentName "cifar100/decentralized/exp_example" `
  -DatasetRoot "cifar100" `
  -PartitionMode "shards_quantity_skew_soft" `
  -SoftMixRatio 0.20 `
  -SoftMinExtraClasses 12 `
  -TopologyMode "augmented_ring" `
  -AsyncMode $true `
  -AsyncDropoutRate 0.1 `
  -MaxAsyncDropouts 2 `
  -HeterogeneousNodes $true
```

## Асинхронный и децентрализованный режимы

В децентрализованном режиме узлы:

- обучают модель локально;
- обмениваются параметрами только с соседями по графу;
- агрегируют обновления с учетом `trust-score`;
- могут работать асинхронно;
- могут временно выпадать из отдельного раунда.

Поддерживаемые топологии:

- `ring`
- `augmented_ring`
- `full_graph`

## Гетерогенность

Реализованы два вида гетерогенности:

- **по данным**  
  разные объемы локальных данных и разные классовые распределения;

- **по узлам**  
  профили `fast`, `medium`, `slow`, отличающиеся:
  - `batch_size`
  - `dropout_weight`
  - `virtual_delay_sec`

## Результаты экспериментов

После запуска результаты сохраняются в:

```text
outputs/experiments/<experiment_name>/
```

Обычно там находятся:

- `round_metrics.jsonl`
- `global_eval_metrics.jsonl`
- `checkpoints/`
- `plots/`
- `experiment_analysis.txt`

## Графики и анализ

Для каждого эксперимента автоматически строятся:

- динамика `accuracy`
- динамика `loss`
- динамика `f1_macro`
- boxplot локальных train loss
- `trust_score` по клиентам
- `update_l2_norm` по клиентам
- распределение классов между клиентами
- coverage классов
- async participation dynamics
- время локального обучения по клиентам

Повторная генерация расширенных графиков:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\regenerate_extended_plots.ps1"
```

## Интеграция с БД

Проект поддерживает сохранение результатов экспериментов в `Neon/PostgreSQL`.

В БД сохраняются:

- параметры эксперимента;
- метрики по раундам;
- локальные клиентские метрики;
- пути к checkpoint;
- пути к графикам и аналитическим артефактам.

Сами изображения и локальные датасеты в БД не хранятся.

### Настройка `.env`

Создай файл `.env` в корне проекта:

```env
DATABASE_URL=postgresql+psycopg://...
DATABASE_URL_DIRECT=postgresql+psycopg://...
```

### Импорт уже завершенных экспериментов в БД

```powershell
python .\scripts\import_experiments_to_db.py --root outputs\experiments --force
```

## Предсказание по одному изображению

```powershell
python .\scripts\predict_image.py --image path\to\image.jpg --checkpoint path\to\best_model.pt
```

## Полезные скрипты

- `scripts/analyze_experiment.py`  
  Текстовый анализ результатов эксперимента
- `scripts/compare_experiments.py`  
  Сравнение нескольких запусков
- `scripts/plot_experiment.py`  
  Построение графиков
- `scripts/import_experiments_to_db.py`  
  Импорт исторических запусков в БД
- `scripts/build_academic_report.py`  
  Подготовка материалов для академического отчета
- `scripts/build_predict_report.py`  
  Подготовка материалов по модулю предсказания

## Ограничения и замечания

- Полный `flwr run .` зависит от совместимого `ray` backend.
- В async-режиме соседние узлы могут использовать последние доступные, а не самые свежие параметры.
- Для `CIFAR-100` жесткие non-IID-сценарии являются значительно более сложными, чем MRI и CIFAR-10.

## Лицензия

`Apache-2.0`
