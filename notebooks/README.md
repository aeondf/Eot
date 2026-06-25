# Ноутбуки спайка

Готовые к запуску ноутбуки для двух кандидатов (вариант 3 — файнтюн — не спайкуем).

| Ноутбук | Подход | Вход | Модель |
|---|---|---|---|
| `01_smart_turn_v3.ipynb` | аудио | PCM 16kHz | pipecat-ai/smart-turn-v3 |
| `02_livekit_turn_detector.ipynb` | текст | транскрипт | livekit/turn-detector (rev `v0.4.1-intl`) |

## Запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Обе модели качаются с HuggingFace при первом запуске и работают через
`onnxruntime` на CPU — torch не требуется.

## Пересборка ноутбуков

Код ноутбуков генерируется из `_build_notebooks.py`:

```bash
python notebooks/_build_notebooks.py
```

## Что проверено

Оба ноутбука прогнаны end-to-end, выводы сохранены в ячейках. Результаты и
выводы — в `docs/06_spike_results.md`.
