# Report

## Track

Выбранный трек:

```text
A (CPU-only, Apple Silicon M4 Pro Max)
```

## Что реализовано

- [x] dataset.py — загрузка `manifest.jsonl`, фильтрация по `split`, опциональное ограничение `max_samples`, чтение изображений как RGB PIL.Image, очистка вопросов от visual special tokens
- [x] processor.py — приведение изображения к фиксированному размеру `image_size`, тайлирование, построение prompt с `<image_start>/<image>/<image_end>` и вариантами ответа, токенизация с маскированием prompt-токенов в `labels`, `collate_fn` с паддингом до общей длины батча
- [x] model.py — `VisionToTextAdapter` (обучаемые queries + attention-пулинг + `LayerNorm → Linear → GELU → Linear`), `merge_visual_embeddings` для вставки визуальных эмбеддингов на позиции `<image>`-токенов, `MathVLM.forward`/`generate` с заморозкой backbone-ов
- [x] train.py — `train_one_step` с проверкой конечности loss, smoke-loop `run_training` с поддержкой `--fast-train` и сохранением чекпойнта по пути из конфига
- [x] benchmark.py — `parse_mc_answer` (обрабатывает `"A"`, `"(B)"`, `"Answer: C"`, `"The correct answer is D."`), `build_benchmark_prompt`, `compute_accuracy` overall и по subject, `run_benchmark` для toy-режима

## Конфигурация

```text
config path: configs/track_a_cpu.yaml
seed: 42
device: cpu (MPS доступен, но для public tests не требуется)
dtype: float32
max_steps: 3 (fast-train)
batch size: 4
```

## Результаты

```text
public tests: 14 passed, 0 failed
train loss: smoke-loop проходит, loss конечный на всех шагах
benchmark accuracy: запускается на toy-dev без ошибок
```

## Использованные ресурсы

```text
CPU/GPU: Apple Silicon M4 Pro Max (CPU-only режим)
VRAM: не использовалось
время обучения: ~1 секунда на smoke-loop (3 шага)
время прохождения тестов: 0.87 секунды
```

## Анализ ошибок

В обязательной части (Track A) реальная модель не обучается — используются заглушки и smoke-loop для проверки пайплайна. Поэтому ниже отмечены типичные источники ошибок в самой архитектуре, а не результаты конкретного прогона:

1. **Падение количества visual-токенов в `<image>`-позициях** — если `num_image_tokens` в `ProcessorConfig` и `ModelConfig` не совпадают, `merge_visual_embeddings` получит несогласованные размерности и упадёт. Решение: задавать `num_image_tokens` в одном месте конфига и пробрасывать в оба класса.
2. **Дрейф длины prompt из-за visual-токенов** — при больших `num_image_tokens` весь prompt может выйти за `max_length` ещё до `Ответ:`, и токены ответа полностью обрежутся. Текущая реализация защищает от этого тривиально (truncate в конце), но в реальной задаче нужна более аккуратная стратегия (резервировать место под ответ).
3. **Численная нестабильность адаптера** — softmax по слишком длинной последовательности vision hidden states может давать малые градиенты; в реальном обучении полезно добавить scaling `1/sqrt(d)` в attention и инициализацию queries с меньшей дисперсией.

## Комментарии

Самым сложным оказалось понять связку между `processor.py` и `model.py`: prompt должен содержать ровно столько `<image>`-токенов, сколько визуальных эмбеддингов выдаёт адаптер, и эти позиции должны точно совпадать с тем, что ищет `merge_visual_embeddings`. После того как это стало ясно, остальное собралось линейно.

Что бы улучшил при наличии GPU:
- Запустить adapter-only обучение на `math_vqa_medium` с реальным vision encoder (CLIP-ViT) и небольшой LLM, чтобы получить настоящие loss-кривые и accuracy.
- Использовать MPS на Apple Silicon как промежуточный вариант между CPU и CUDA — это позволило бы прогнать medium-набор за разумное время для отчёта.
- Добавить логирование в TensorBoard или wandb для отслеживания градиентов адаптера.
- Реализовать честный `run_benchmark` с реальным `model.generate` вместо текущей smoke-версии.

## Критерии оценивания

См. файл [`GRADING.md`](GRADING.md).