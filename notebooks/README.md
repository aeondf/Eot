# Ноутбуки спайка

Готовые к запуску ноутбуки для двух кандидатов.

| Ноутбук | Подход | Вход | Модель |
|---|---|---|---|
| `01_smart_turn_v3.ipynb` | аудио | PCM 16kHz | pipecat-ai/smart-turn-v3 |
| `02_livekit_turn_detector.ipynb` | текст | транскрипт | livekit/turn-detector (rev `v0.4.1-intl`) |

## Запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Обе модели качаются с HuggingFace при первом запуске и работают через
`onnxruntime` на CPU — torch не требуется.
