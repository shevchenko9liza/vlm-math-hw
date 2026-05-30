# Report

## Track

Выбранный трек:

```text
B (small-GPU style adapter-only training on Apple Silicon MPS) + extended MathVista real VLM evaluation
```

## Что реализовано

- dataset.py — загрузка `manifest.jsonl`, фильтрация по `split`, опциональное ограничение `max_samples`, чтение изображений как RGB PIL.Image, очистка вопросов от visual special tokens
- processor.py — приведение изображения к фиксированному размеру `image_size`, тайлирование, построение prompt с `<image_start>/<image>/<image_end>` и вариантами ответа, токенизация с маскированием prompt-токенов в `labels`, `collate_fn` с паддингом до общей длины батча
- model.py — `VisionToTextAdapter` (обучаемые queries + attention-пулинг + `LayerNorm → Linear → GELU → Linear`), `merge_visual_embeddings` для вставки визуальных эмбеддингов на позиции `<image>`-токенов, `MathVLM.forward`/`generate` с заморозкой backbone-ов
- train.py — `train_one_step` с проверкой конечности loss, smoke-loop `run_training` с поддержкой `--fast-train` и сохранением чекпойнта по пути из конфига
- benchmark.py — `parse_mc_answer` (обрабатывает `"A"`, `"(B)"`, `"Answer: C"`, `"The correct answer is D."`), `build_benchmark_prompt`, `compute_accuracy` overall и по subject, `run_benchmark` для toy-режима
- scripts/train_adapter_mps.py + scripts/advanced_vlm_experiment.py — расширенный Track B-style harness для adapter-only обучения CLIP+Qwen на Apple Silicon MPS, с frozen vision encoder и frozen LLM

## Конфигурация

```text
base smoke config path: configs/track_a_cpu.yaml
seed: 42
base device: cpu
base dtype: float32
base max_steps: 3 (fast-train)

Track B adapter config:
manifest: assets/math_vqa_medium/manifest.jsonl
vision model: openai/clip-vit-base-patch32 (frozen)
language model: Qwen/Qwen2.5-1.5B-Instruct (frozen)
trainable part: adapter only
device: mps
dtype: bfloat16
num_image_tokens: 16
epochs: 2
learning rate: 0.0003
gradient accumulation: 8
```

## Результаты

```text
public tests: 14 passed, 0 failed
train loss: smoke-loop проходит, loss конечный на всех шагах
benchmark accuracy: запускается на toy-dev без ошибок
```

## Track B: adapter-only обучение на math_vqa_medium

Для Track B был отдельно запущен adapter-only training: vision encoder и LLM заморожены, обучается только небольшой adapter.

Команда:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/advanced_vlm_experiment.py train \
  --manifest assets/math_vqa_medium/manifest.jsonl \
  --device mps \
  --epochs 2 \
  --lr 0.0003 \
  --num-image-tokens 16 \
  --eval-max-samples 40 \
  --cpu-timing-steps 0 \
  --run-id track-b-medium-adapter-mps
```

Итог:

```text
dataset: math_vqa_medium
train/dev: 180 / 40
adapter parameters: 3.56M
frozen vision+LM parameters: 1630.76M
trainable fraction: 0.218%
epoch losses: 1.0252 -> 0.6141
dev baseline before training: 0.4750
dev final best checkpoint: 0.4000
training time: 73.8 sec on MPS
checkpoint: artifacts/adapter.pt
full run dir: artifacts/advanced/runs/track-b-medium-adapter-mps
```

Этот запуск закрывает Track B как adapter-only training, loss конечный, backbone frozen, adapter сохранён. При этом качество adapter на маленьком medium-dev не стало главным claim, dev accuracy не улучшилась относительно random-adapter baseline. Поэтому для содержательной quality-оценки ниже используется отдельный real VLM benchmark на MathVista.

## Расширенная проверка: MathVista

Toy-набор `assets/toy_math_vqa/` использовался только как smoke-check пайплайна. Для внешней quality-проверки был подготовлен небольшой MathVista `testmini` subset:

```bash
.venv/bin/python scripts/prepare_mathvista_testmini.py \
  --out assets/mathvista_testmini \
  --max-samples 50 \
  --streaming \
  --multiple-choice-only

.venv/bin/python -m hw.benchmark --config configs/eval_mathvista_testmini.yaml
```

Результат текущего benchmark baseline:

```text
dataset: MathVista testmini, multiple-choice only
n: 50
overall accuracy: 0.2600
prediction file: artifacts/mathvista_predictions.jsonl
```

Также был выделен строгий A-D subset (`assets/mathvista_testmini/manifest_mc4.jsonl`), совместимый с CLIP+Qwen adapter evaluator:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/advanced_vlm_experiment.py eval-adapter \
  --manifest assets/mathvista_testmini/manifest_mc4.jsonl \
  --split testmini \
  --max-samples 22 \
  --adapter artifacts/advanced/runs/clip-qwen-v3-option-aug-init-v2-lr1e4/adapter_best.pt \
  --device mps \
  --run-id mathvista-mc4-adapter-v3-option-aug-22
```

```text
dataset: MathVista testmini, strict A-D multiple-choice subset
n: 22
adapter accuracy: 0.5000
prediction distribution: A=8, B=5, C=6, D=3
```

Важно: `0.2600` — это не результат полноценно обученной VLM-модели, а baseline-путь для проверки формата prompt/answer parsing и подсчёта метрик. `0.5000` — честная проверка текущего синтетически обученного адаптера на маленьком MathVista-compatible subset.

## Отдельный VLM-track: MathVista score-mode

Для расширенной части добавлен отдельный evaluator `scripts/eval_real_vlm_mathvista.py`: он загружает настоящую vision-language модель, передаёт ей изображение + вопрос + варианты, а затем выбирает ответ по log-probability следующего option token. Финальный выбранный режим — `inference_mode=score`, потому что он не зависит от fragile text parsing и даёт `invalid_prediction_count = 0`.

Зависимости для этого трека добавлены в `pyproject.toml` optional `ml`: `torchvision` нужен image processor-ам, `num2words` нужен SmolVLM processor-у. Для большого MathVista прогона добавлен флаг `--max-image-side 1024`: он сохраняет aspect ratio, но ограничивает слишком крупные изображения, которые иначе могут вызвать MPS out-of-memory.

Подготовка полного scoreable MathVista subset:

```bash
.venv/bin/python scripts/prepare_mathvista_testmini.py \
  --out assets/mathvista_testmini_1000 \
  --max-samples 1000 \
  --streaming \
  --multiple-choice-only

.venv/bin/python scripts/advanced_vlm_experiment.py blank-images \
  --manifest assets/mathvista_testmini_1000/manifest.jsonl \
  --dataset-dir assets/mathvista_testmini_1000_blank \
  --overwrite
```

MathVista `testmini` содержит 1000 строк, но score-mode применим только к multiple-choice rows. После полного scan получился честный scoreable subset: `n = 540` multiple-choice examples, из них `273` strict A-D examples.

Финальные команды:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/eval_real_vlm_mathvista.py \
  --manifest assets/mathvista_testmini_1000/manifest.jsonl \
  --split testmini \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --backend qwen2_5_vl \
  --device mps \
  --dtype float16 \
  --local-files-only \
  --inference-mode score \
  --max-image-side 1024 \
  --run-id qwen25vl-3b-mathvista-testmini1000-score \
  --overwrite \
  --print-every 50

PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/eval_real_vlm_mathvista.py \
  --manifest assets/mathvista_testmini_1000_blank/manifest.jsonl \
  --split testmini \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --backend qwen2_5_vl \
  --device mps \
  --dtype float16 \
  --local-files-only \
  --inference-mode score \
  --max-image-side 1024 \
  --run-id qwen25vl-3b-mathvista-testmini1000-score-blank \
  --overwrite \
  --print-every 50
```

Результаты:


| Model / setting                 | Dataset                               | n   | Accuracy | Balanced acc. | Invalid | Collapse |
| ------------------------------- | ------------------------------------- | --- | -------- | ------------- | ------- | -------- |
| Parser smoke baseline           | MathVista testmini MC                 | 50  | 0.2600   | -             | -       | -        |
| CLIP+Qwen synthetic adapter     | MathVista strict A-D                  | 22  | 0.5000   | 0.5354        | 0       | false    |
| SmolVLM2-256M zero-shot         | MathVista testmini MC                 | 50  | 0.4200   | 0.2565        | 0       | false    |
| Qwen2.5-VL-3B score pilot       | MathVista testmini MC                 | 50  | 0.7400   | 0.6454        | 0       | false    |
| Qwen2.5-VL-3B score pilot blank | MathVista testmini MC blank           | 50  | 0.5400   | 0.3824        | 0       | false    |
| Qwen2.5-VL-3B score final       | MathVista testmini MC full scan       | 540 | 0.7167   | 0.7170        | 0       | false    |
| Qwen2.5-VL-3B score final blank | MathVista testmini MC full scan blank | 540 | 0.5093   | 0.3499        | 0       | false    |


Qwen2.5-VL-3B в score-mode даёт сильный real VLM baseline `0.7167` на полном scoreable MathVista testmini subset. Blank-image ablation на тех же строках падает до `0.5093`; visual delta равен `+20.7 п.п.` (`0.7167 - 0.5093`). Это показывает, что прирост не сводится к текстовым priors и распределению вариантов, реальные изображения существенно помогают.

Pilot-результат на 50 examples (`0.7400`, blank `0.5400`) совпадает по выводу с большим прогоном, но финальными считаются метрики `n = 540`. Reasoned prompt был проверен отдельно, но не выбран как основной, при строгом parsing он давал invalid generations, тогда как score-mode стабильно возвращает один из допустимых вариантов.

## Использованные ресурсы

```text
Обязательная совместимость: CPU smoke pipeline проходит public tests.
Track B обучение: CLIP+Qwen adapter-only на Apple Silicon MPS.
Расширенная quality evaluation: Qwen2.5-VL-3B score evaluation на Apple Silicon MPS.
VRAM: unified memory через MPS
время base smoke-loop: ~1 секунда на 3 шага
время Track B adapter training: 73.8 секунды на 2 эпохи medium train
время MathVista real/blank score eval: 360.5 + 137.9 секунды
время прохождения тестов: 0.71 секунды
```

## Анализ ошибок

В базовой CPU-compatible части используется smoke-loop для проверки пайплайна. В Track B-style части adapter действительно обучался на `math_vqa_medium`, но маленький adapter на 180 train examples не улучшил dev accuracy относительно random-adapter baseline. Поэтому ниже разделены архитектурные риски и реальные наблюдения:

1. **Падение количества visual-токенов в `<image>`-позициях** — если `num_image_tokens` в `ProcessorConfig` и `ModelConfig` не совпадают, `merge_visual_embeddings` получит несогласованные размерности и упадёт. Решение: задавать `num_image_tokens` в одном месте конфига и пробрасывать в оба класса.
2. **Дрейф длины prompt из-за visual-токенов** — при больших `num_image_tokens` весь prompt может выйти за `max_length` ещё до `Ответ:`, и токены ответа полностью обрежутся. Текущая реализация защищает от этого тривиально (truncate в конце), но в реальной задаче нужна более аккуратная стратегия (резервировать место под ответ).
3. **Численная нестабильность адаптера** — softmax по vision hidden states сейчас считается без scaling `1/sqrt(d)`, поэтому в более длинных visual sequences возможны saturation и слабые градиенты.
4. **Малый medium-набор не гарантирует рост dev accuracy** — в Track B run loss снизился (`1.0252 -> 0.6141`), но dev accuracy упала с `0.4750` baseline до `0.4000` best checkpoint. Это похоже на сочетание маленькой выборки, сильных text priors у LLM и недостаточного alignment adapter-а.
5. **Real VLM benchmark зависит от изображения** — на MathVista scoreable subset Qwen2.5-VL даёт `0.7167`, а blank-image ablation `0.5093`, то есть реальные изображения дают `+20.7 п.п.`.

## Комментарии

Самым сложным оказалось понять связку между `processor.py` и `model.py`: prompt должен содержать ровно столько `<image>`-токенов, сколько визуальных эмбеддингов выдаёт адаптер, и эти позиции должны точно совпадать с тем, что ищет `merge_visual_embeddings`. После того как это стало ясно, остальное собралось линейно.

Что бы улучшила при наличии более сильного GPU/CUDA:

- Интегрировать real VLM scoring/evaluation прямо в основной benchmark-интерфейс, чтобы hw.benchmark умел не только smoke baseline, но и честный model.generate/score path.
- Продолжить adapter training на большем synthetic/medium mixture с регулярными ablations: real / blank / shuffled / counterfactual.
- Для adapter attention добавить scaling 1/sqrt(d) и проверить stability по loss/gradient norms.
- Запустить LoRA/SFT или adapter fine-tuning на более сильной GPU и сравнить против Qwen2.5-VL score baseline на MathVista.

## Критерии оценивания

См. файл [GRADING.md](GRADING.md).
