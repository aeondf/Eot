"""Генератор ноутбуков для спайка. Запуск: python notebooks/_build_notebooks.py"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

# ============================================================
# Notebook 1: Pipecat Smart Turn v3 (аудио)
# ============================================================
nb = new_notebook()
c = nb.cells

c.append(new_markdown_cell(
"""# Спайк 1 — Pipecat Smart Turn v3 (аудио-модель)

Семантический VAD: смотрит на **сырое аудио** (не транскрипт) и говорит,
закончил ли человек реплику.

- Модель: Whisper Tiny encoder + linear head, ~8M параметров
- Вход: PCM 16kHz mono, до 8 секунд (берётся хвост реплики)
- Выход: вероятность `complete` (0..1), уже после sigmoid
- 23 языка включая русский
- HF: https://huggingface.co/pipecat-ai/smart-turn-v3

Запускаем через `onnxruntime` — torch не нужен."""))

c.append(new_markdown_cell("## Установка зависимостей"))
c.append(new_code_cell(
"""# !pip install onnxruntime transformers huggingface_hub librosa numpy
# для генерации тестовой речи (опционально): !pip install gtts soundfile"""))

c.append(new_markdown_cell("## Загрузка модели"))
c.append(new_code_cell(
"""import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor
from huggingface_hub import hf_hub_download

# варианты: smart-turn-v3.2-cpu.onnx / smart-turn-v3.2-gpu.onnx
MODEL_FILE = "smart-turn-v3.2-cpu.onnx"
onnx_path = hf_hub_download("pipecat-ai/smart-turn-v3", MODEL_FILE)

session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
feature_extractor = WhisperFeatureExtractor(chunk_length=8)  # 8 сек контекста

print("inputs :", [(i.name, i.shape) for i in session.get_inputs()])
print("outputs:", [(o.name, o.shape) for o in session.get_outputs()])"""))

c.append(new_markdown_cell("## Функция предсказания"))
c.append(new_code_cell(
'''def predict_endpoint(audio: np.ndarray, sr: int = 16000) -> float:
    """Возвращает вероятность того, что реплика завершена (0..1).

    audio — моно float32 в диапазоне [-1, 1]. Если sr != 16000, ресемплим.
    """
    import librosa
    audio = audio.astype(np.float32)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    # нормализация в [-1, 1]
    peak = np.max(np.abs(audio)) if audio.size else 1.0
    if peak > 1.0:
        audio = audio / peak
    # берём последние 8 секунд (где заканчивается реплика)
    if len(audio) > 8 * 16000:
        audio = audio[-8 * 16000:]

    feats = feature_extractor(
        audio, sampling_rate=16000, return_tensors="np",
        padding="max_length", max_length=8 * 16000,
        truncation=True, do_normalize=True,
    )
    x = feats.input_features.squeeze(0).astype(np.float32)[None]   # [1, 80, 800]
    out = session.run(None, {"input_features": x})[0]
    score = float(out[0][0])                                       # уже sigmoid
    return score'''))

c.append(new_markdown_cell("## Смоук-тест (модель запускается)"))
c.append(new_code_cell(
"""dummy = (np.random.randn(16000) * 0.1).astype(np.float32)  # 1 сек шума
print("smoke score:", round(predict_endpoint(dummy), 4))"""))

c.append(new_markdown_cell(
"""## Тест на реальной русской речи (gTTS)

Генерируем несколько фраз через Google TTS.

> Важно: TTS читает **любую** фразу с завершающей интонацией, поэтому
> `incomplete` примеры будут получать завышенный score. Для честной оценки
> нужны реальные записи звонков с оборванными репликами. Здесь — только
> проверка что пайплайн работает на живом аудио."""))
c.append(new_code_cell(
'''import os
try:
    from gtts import gTTS
    import librosa
    samples = {
        "complete_1":   "Здравствуйте, я хотел бы узнать баланс по своей карте.",
        "complete_2":   "Спасибо большое, до свидания.",
        "incomplete_1": "Мне нужно",
        "incomplete_2": "А скажите пожалуйста какой у меня",
    }
    os.makedirs("/tmp/tts", exist_ok=True)
    print(f"{'sample':<16}{'score':>8}  expected")
    for name, text in samples.items():
        path = f"/tmp/tts/{name}.mp3"
        if not os.path.exists(path):
            gTTS(text, lang="ru").save(path)
        audio, _ = librosa.load(path, sr=16000, mono=True)
        exp = "complete" if name.startswith("complete") else "incomplete"
        print(f"{name:<16}{predict_endpoint(audio):>8.4f}  {exp}")
except Exception as e:
    print("gTTS недоступен, пропускаем:", e)'''))

c.append(new_markdown_cell(
"""## Замер скорости инференса"""))
c.append(new_code_cell(
"""import time
audio = (np.random.randn(8 * 16000) * 0.1).astype(np.float32)
predict_endpoint(audio)  # прогрев
t0 = time.perf_counter()
N = 50
for _ in range(N):
    predict_endpoint(audio)
print(f"avg latency: {(time.perf_counter()-t0)/N*1000:.1f} ms / вызов")"""))

c.append(new_markdown_cell(
"""## Как встроить в наш пайплайн

```
если (predict_endpoint(turn_audio) > 0.9 И тишина > 1с)
ИЛИ тишина > 5с
→ отправляем в LLM
```

Модель зовём на накопленном аудио текущей реплики (хвост до 8 сек),
параллельно с ASR — транскрипт ждать не нужно.

### Следующий шаг
Заменить gTTS на реальные записи звонков, разметить complete/incomplete,
построить confusion matrix и подобрать порог (см. `docs/05_metrics.md`)."""))

nbf.write(nb, "notebooks/01_smart_turn_v3.ipynb")
print("wrote notebooks/01_smart_turn_v3.ipynb")

# ============================================================
# Notebook 2: LiveKit turn-detector multilingual (текст)
# ============================================================
nb = new_notebook()
c = nb.cells

c.append(new_markdown_cell(
"""# Спайк 2 — LiveKit turn-detector multilingual (текстовая модель)

Семантический детектор конца реплики по **транскрипту** (от ASR).

- Базовая модель: Qwen2.5-0.5B-Instruct, дистиллирована, ONNX INT8
- Вход: история диалога (до 6 ходов) в формате Qwen chat template,
  у последней реплики пользователя убран закрывающий `<|im_end|>`
- Выход: вероятность того, что дальше идёт `<|im_end|>` = реплика завершена
- 14 языков включая русский, для каждого свой порог в `languages.json`
- HF: https://huggingface.co/livekit/turn-detector

ВАЖНО: мультиязычная модель лежит **не в main**, а в ревизии `v0.4.1-intl`
(файл `onnx/model_q8.onnx` + `languages.json`). В main лежит старая English
SmolLM2-версия — она работает иначе."""))

c.append(new_markdown_cell("## Установка зависимостей"))
c.append(new_code_cell(
"""# !pip install onnxruntime transformers huggingface_hub jinja2 numpy"""))

c.append(new_markdown_cell("## Загрузка модели (ревизия v0.4.1-intl)"))
c.append(new_code_cell(
"""import json
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

REPO = "livekit/turn-detector"
REVISION = "v0.4.1-intl"          # мультиязычная модель

tokenizer = AutoTokenizer.from_pretrained(REPO, revision=REVISION)
onnx_path = hf_hub_download(REPO, "onnx/model_q8.onnx", revision=REVISION)
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

# пороги по языкам (подобраны авторами под высокий recall)
languages = json.load(open(hf_hub_download(REPO, "languages.json", revision=REVISION)))
RU_THRESHOLD = languages["ru"]["threshold"]

print("output:", [(o.name, o.shape) for o in session.get_outputs()])
print("ru threshold:", RU_THRESHOLD, "| meta:", languages["ru"])"""))

c.append(new_markdown_cell(
"""## Функция предсказания

Модель отдаёт вероятность EOU для каждой позиции — берём последнюю.
ONNX уже содержит softmax, дополнительной нормализации не нужно."""))
c.append(new_code_cell(
'''def eou_prob(text: str, context: list[dict] | None = None) -> float:
    """Вероятность что реплика пользователя завершена (0..1).

    context — предыдущие ходы диалога, напр.
        [{"role": "assistant", "content": "..."},
         {"role": "user", "content": "..."}]
    """
    messages = (context or []) + [{"role": "user", "content": text}]
    s = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    s = s.rsplit("<|im_end|>", 1)[0]      # убираем закрывающий тег у последней реплики
    ids = tokenizer(s, return_tensors="np", add_special_tokens=False)["input_ids"].astype(np.int64)
    prob = session.run(None, {"input_ids": ids})[0]   # [1, seq]
    return float(prob[0, -1])'''))

c.append(new_markdown_cell("## Тест на русских фразах"))
c.append(new_code_cell(
'''tests = [
    ("Здравствуйте, я хотел бы узнать баланс по своей карте", "complete"),
    ("Мне нужно",                                            "incomplete"),
    ("Скажите пожалуйста, какой у меня",                     "incomplete"),
    ("Я хочу оплатить счёт за электричество",                "complete"),
    ("Мой номер телефона восемь девятьсот",                  "incomplete"),
    ("Спасибо, до свидания",                                 "complete"),
    ("А можно ли",                                           "incomplete"),
    ("Какие у вас есть тарифы?",                             "complete"),
]

print(f"ru threshold = {RU_THRESHOLD}")
print(f"{'expected':<11}{'score':>8} {'pred':>11}  text")
for text, exp in tests:
    p = eou_prob(text)
    pred = "complete" if p > RU_THRESHOLD else "incomplete"
    mark = "OK" if pred == exp else "XX"
    print(f"{exp:<11}{p:>8.4f} {pred:>11} {mark}  {text}")'''))

c.append(new_markdown_cell(
"""## Замечание про порог

Дефолтный `ru threshold = 0.0032` подобран авторами под **замену VAD** —
максимальный recall (TPR ~0.99), поэтому почти всё классифицируется как
`complete`. Для нашей формулы важен не бинарный вердикт модели, а сам
**score** — мы сравниваем его со своим порогом (например 0.9).

Score упорядочен правильно: `"А можно ли"` → ~0.0003, `"Спасибо, до свидания"` → ~0.94."""))

c.append(new_markdown_cell("## Эффект контекста диалога"))
c.append(new_code_cell(
'''context = [
    {"role": "assistant", "content": "Назовите номер вашей карты"},
]
phrase = "Пять тысяч двести"
print("без контекста:", round(eou_prob(phrase), 4))
print("с контекстом :", round(eou_prob(phrase, context), 4))'''))

c.append(new_markdown_cell("## Замер скорости инференса"))
c.append(new_code_cell(
"""import time
eou_prob("прогрев")
t0 = time.perf_counter()
N = 50
for _ in range(N):
    eou_prob("Я хочу оплатить счёт за электричество")
print(f"avg latency: {(time.perf_counter()-t0)/N*1000:.1f} ms / вызов")"""))

c.append(new_markdown_cell(
"""## Как встроить в наш пайплайн

```
если (eou_prob(accumulated_text) > 0.9 И тишина > 1с)
ИЛИ тишина > 5с
→ отправляем в LLM
```

Зовём на накопленном транскрипте от Qwen ASR. Минус — ждём транскрипт
(добавляет задержку), плюс — понимает смысл.

### Следующий шаг
Прогнать на транскриптах реальных звонков, сравнить со Smart Turn по
FP/FN при нашем пороге (см. `docs/05_metrics.md`)."""))

nbf.write(nb, "notebooks/02_livekit_turn_detector.ipynb")
print("wrote notebooks/02_livekit_turn_detector.ipynb")
