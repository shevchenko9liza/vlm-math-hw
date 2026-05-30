from __future__ import annotations
import argparse
import json
import math
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import torch
import transformers
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionModel,
    get_cosine_schedule_with_warmup,
)
from hw.constants import IGNORE_INDEX, IMAGE_TOKEN
from hw.dataset import MathVQADataset
from hw.model import VisionToTextAdapter, merge_visual_embeddings


@dataclass
class ExpConfig:
    manifestpath: str = "assets/math_vqa_medium/manifest.jsonl"
    visionmodel: str = "openai/clip-vit-base-patch32"
    languagemodel: str = "Qwen/Qwen2.5-1.5B-Instruct"
    numimagetokens: int = 16
    maxlength: int = 256
    batchsize: int = 1
    gradaccum: int = 8
    learningrate: float = 1e-3
    weightdecay: float = 0.01
    warmupratio: float = 0.1
    epochs: int = 4
    gradclipnorm: float = 1.0
    labelsmoothing: float = 0.0
    evalmaxsamples: int = 40
    evaleveryepoch: bool = True
    seed: int = 42
    device: str = "mps"
    dtype: str = "bfloat16"
    cputimingsteps: int = 10
    artifactsdir: str = "artifacts"
    logpath: str = "artifacts/training_log.json"
    adapterpath: str = "artifacts/adapter_best.pt"
    finaladapterpath: str = "artifacts/adapter_final.pt"
    predictionspath: str = "artifacts/dev_predictions.jsonl"
    initadapterpath: str | None = None
    blankuniformweight: float = 0.0
    blankrgb: tuple[int, int, int] = (248, 248, 248)

def setseed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def collectenv(cfg: ExpConfig) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": cfg.device,
        "dtype": cfg.dtype,
        "mps_available": torch.backends.mps.is_available(),
    }

def getdtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


class QwenVLM(nn.Module):
    def __init__(self, cfg: ExpConfig) -> None:
        super().__init__()
        dtype = getdtype(cfg.dtype)
        self.vision = CLIPVisionModel.from_pretrained(cfg.visionmodel, torch_dtype=dtype)
        self.language = AutoModelForCausalLM.from_pretrained(cfg.languagemodel, torch_dtype=dtype)
        visionhidden = self.vision.config.hidden_size
        texthidden = self.language.config.hidden_size
        self.adapter = VisionToTextAdapter(vision_hidden_size=visionhidden, text_hidden_size=texthidden, num_image_tokens=cfg.numimagetokens).to(dtype)
        for p in self.vision.parameters():
            p.requires_grad = False
        for p in self.language.parameters():
            p.requires_grad = False
        self.imagetokenid: int = -1
        self.dtype = dtype
    def encodevision(self, pixelvalues: torch.Tensor) -> torch.Tensor:
        visionout = self.vision(pixel_values=pixelvalues.to(self.dtype))
        return self.adapter(visionout.last_hidden_state)
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        visualembeds = self.encodevision(batch["pixelvalues"])
        textembeds = self.language.model.embed_tokens(batch["inputids"])
        merged = merge_visual_embeddings(textembeds, batch["inputids"], visualembeds, self.imagetokenid)
        out = self.language(inputs_embeds=merged, attention_mask=batch["attentionmask"], labels=batch["labels"])
        return {"loss": out.loss, "logits": out.logits}


class BatchBuilder:
    def __init__(self, cfg: ExpConfig) -> None:
        self.cfg = cfg
        self.imageproc = CLIPImageProcessor.from_pretrained(cfg.visionmodel)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.languagemodel)
        added = self.tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.imagetokenid = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        self.addedtokens = added
        self.choicetokenids = torch.tensor(
            [self.tokenizer(letter, add_special_tokens=False)["input_ids"][-1] for letter in ["A", "B", "C", "D"]],
            dtype=torch.long,
        )
        blankimage = Image.new("RGB", (224, 224), tuple(cfg.blankrgb))
        self.blankpixelvalues = self.imageproc(images=blankimage, return_tensors="pt")["pixel_values"][0]
    def buildmessages(self, question: str, options: list[str]) -> list[dict]:
        imageblock = IMAGE_TOKEN * self.cfg.numimagetokens
        opts = "\n".join(options)
        usercontent = (
            f"{imageblock}\n"
            f"You will see an image with a math problem (chart, plot, geometric figure, or table).\n"
            f"Carefully analyze the visual information and choose the correct option.\n\n"
            f"Question: {question}\n"
            f"Options:\n{opts}\n\n"
            f"Respond with a single letter (A, B, C, or D)."
        )
        return [
            {"role": "system", "content": "You are an expert visual math reasoner. Answer multiple-choice math questions based on images."},
            {"role": "user", "content": usercontent},
        ]
    def encodesample(self, sample, answeroverride: str | None = None) -> dict[str, torch.Tensor]:
        messages = self.buildmessages(sample.question, sample.options)
        prompttext = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        answer = answeroverride if answeroverride is not None else sample.answer
        fulltext = prompttext + answer + self.tokenizer.eos_token
        promptids = self.tokenizer(prompttext, add_special_tokens=False)["input_ids"]
        fullids = self.tokenizer(fulltext, add_special_tokens=False)["input_ids"]
        fullids = fullids[: self.cfg.maxlength]
        labels = [IGNORE_INDEX] * min(len(promptids), len(fullids))
        labels += fullids[len(promptids):]
        labels = labels[: self.cfg.maxlength]
        pixels = self.imageproc(images=sample.image, return_tensors="pt")["pixel_values"][0]
        return {
            "inputids": torch.tensor(fullids, dtype=torch.long),
            "attentionmask": torch.ones(len(fullids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixelvalues": pixels,
        }

    def collate(self, items: list[dict]) -> dict[str, torch.Tensor]:
        padid = self.tokenizer.pad_token_id
        maxlen = max(it["inputids"].shape[0] for it in items)
        def padto(t: torch.Tensor, value: int) -> torch.Tensor:
            deficit = maxlen - t.shape[0]
            return t if deficit == 0 else torch.cat([t, torch.full((deficit,), value, dtype=t.dtype)])
        return {
            "inputids": torch.stack([padto(it["inputids"], padid) for it in items]),
            "attentionmask": torch.stack([padto(it["attentionmask"], 0) for it in items]),
            "labels": torch.stack([padto(it["labels"], IGNORE_INDEX) for it in items]),
            "pixelvalues": torch.stack([it["pixelvalues"] for it in items]),
        }


@torch.no_grad()
def evaluate(model: QwenVLM, builder: BatchBuilder, dataset, device: str, maxsamples: int) -> dict[str, Any]:
    model.eval()
    letters = ["A", "B", "C", "D"]
    total = min(len(dataset), maxsamples)
    correct = 0
    predictions: list[dict] = []
    bysubject: dict[str, list[bool]] = defaultdict(list)
    predcounter: Counter = Counter()
    for index in range(total):
        sample = dataset[index]
        scores = []
        for letter in letters:
            item = builder.encodesample(sample, answeroverride=letter)
            batch = builder.collate([item])
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            scores.append(-out["loss"].item())
        pred = letters[int(np.argmax(scores))]
        isright = (pred == sample.answer)
        correct += int(isright)
        predcounter[pred] += 1
        bysubject[sample.subject].append(isright)
        predictions.append({
            "id": sample.id,
            "subject": sample.subject,
            "question": sample.question,
            "options": sample.options,
            "gold": sample.answer,
            "pred": pred,
            "correct": isright,
            "scores": {letter: float(score) for letter, score in zip(letters, scores)},
        })
    overall = correct / max(1, total)
    persubject = {
        subject: {"accuracy": sum(results) / len(results), "n": len(results)}
        for subject, results in bysubject.items()
    }
    return {
        "overall": overall,
        "per_subject": persubject,
        "prediction_distribution": dict(predcounter),
        "n_evaluated": total,
        "predictions": predictions,
    }

def printevalreport(label: str, report: dict[str, Any]) -> None:
    print(f"\n[{label}] overall = {report['overall']:.4f} on {report['n_evaluated']} examples")
    print(f"  prediction distribution: {report['prediction_distribution']}")
    if report["per_subject"]:
        print(f"per-subject accuracy:")
        for subject, stat in sorted(report["per_subject"].items()):
            print(f"{subject:20s} acc={stat['accuracy']:.3f} (n={stat['n']})")


def blankuniformloss(model: QwenVLM, builder: BatchBuilder, batch: dict[str, torch.Tensor], device: str) -> torch.Tensor:
    blankbatch = dict(batch)
    blankpixels = builder.blankpixelvalues.to(device).unsqueeze(0).expand_as(batch["pixelvalues"])
    blankbatch["pixelvalues"] = blankpixels
    out = model(blankbatch)

    labels = batch["labels"]
    hasanswer = labels.ne(IGNORE_INDEX).any(dim=1)
    if not bool(hasanswer.any()):
        return out["logits"].new_zeros(())

    firstanswerpos = labels.ne(IGNORE_INDEX).int().argmax(dim=1)
    predpos = torch.clamp(firstanswerpos - 1, min=0)
    batchindex = torch.arange(labels.shape[0], device=device)
    choicetokens = builder.choicetokenids.to(device)
    choicelogits = out["logits"][batchindex, predpos][:, choicetokens]
    choicelogits = choicelogits[hasanswer].float()
    logprobs = torch.log_softmax(choicelogits, dim=-1)
    return (-logprobs.mean(dim=-1) - math.log(4.0)).mean()


def train(
    cfg: ExpConfig,
    model: QwenVLM,
    builder: BatchBuilder,
    trainloader: DataLoader,
    devdataset,
    device: str,
) -> dict[str, Any]:
    nupdates = (len(trainloader) * cfg.epochs) // cfg.gradaccum
    nwarmup = max(1, int(cfg.warmupratio * nupdates))
    print(f"\nTraining schedule: {nupdates} optimizer updates, {nwarmup} warmup")
    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=cfg.learningrate, weight_decay=cfg.weightdecay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=nwarmup, num_training_steps=nupdates)
    losslog: list[dict] = []
    epochevals: list[dict] = []
    bestacc = -1.0
    starttime = time.time()
    globalstep = 0
    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad()
        epochlosses: list[float] = []
        for step, batch in enumerate(trainloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            taskloss = out["loss"]
            blankloss = None
            loss = taskloss
            if cfg.blankuniformweight > 0:
                blankloss = blankuniformloss(model, builder, batch, device)
                loss = loss + cfg.blankuniformweight * blankloss
            if not torch.isfinite(loss):
                print(f"non-finite loss at epoch {epoch} step {step}, skipping")
                continue
            (loss / cfg.gradaccum).backward()
            lossval = float(loss.detach())
            tasklossval = float(taskloss.detach())
            blanklossval = float(blankloss.detach()) if blankloss is not None else 0.0
            epochlosses.append(lossval)
            losslog.append({
                "epoch": epoch,
                "step": globalstep,
                "loss": lossval,
                "task_loss": tasklossval,
                "blank_uniform_loss": blanklossval,
                "lr": scheduler.get_last_lr()[0],
            })
            globalstep += 1
            if (step + 1) % cfg.gradaccum == 0:
                torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), max_norm=cfg.gradclipnorm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if step % 20 == 0:
                lr = scheduler.get_last_lr()[0]
                if cfg.blankuniformweight > 0:
                    print(
                        f"epoch {epoch} step {step:3d}/{len(trainloader)} "
                        f"loss={lossval:.4f} task={tasklossval:.4f} blank={blanklossval:.4f} lr={lr:.2e}"
                    )
                else:
                    print(f"epoch {epoch} step {step:3d}/{len(trainloader)} loss={lossval:.4f} lr={lr:.2e}")
        avgloss = sum(epochlosses) / max(1, len(epochlosses))
        print(f"Epoch {epoch} done, avg loss = {avgloss:.4f}")
        if cfg.evaleveryepoch:
            evalreport = evaluate(model, builder, devdataset, device, cfg.evalmaxsamples)
            printevalreport(f"eval @ epoch {epoch}", evalreport)
            epochevals.append({"epoch": epoch, "avg_loss": avgloss, "eval": evalreport})
            if evalreport["overall"] > bestacc:
                bestacc = evalreport["overall"]
                torch.save(model.adapter.state_dict(), cfg.adapterpath)
                print(f"[checkpoint] best so far ({bestacc:.4f}) saved to {cfg.adapterpath}")
    trainsec = time.time() - starttime
    print(f"\nTraining done in {trainsec:.1f} sec on {device}")
    return {
        "losslog": losslog,
        "epoch_evals": epochevals,
        "train_seconds": trainsec,
        "best_acc": bestacc,
    }


def measurecputime(model: QwenVLM, trainloader: DataLoader, nsteps: int, gradaccum: int) -> dict[str, float]:
    if nsteps <= 0:
        return {"cpu_steps": 0, "cpu_seconds": 0.0, "cpu_sec_per_step": 0.0}
    modelcpu = model.to("cpu")
    modelcpu.train()
    starttime = time.time()
    seen = 0
    for batch in trainloader:
        batch = {k: v.to("cpu") for k, v in batch.items()}
        out = modelcpu(batch)
        if torch.isfinite(out["loss"]):
            (out["loss"] / gradaccum).backward()
        seen += 1
        if seen >= nsteps:
            break
    elapsed = time.time() - starttime
    return {"cpu_steps": seen, "cpu_seconds": elapsed, "cpu_sec_per_step": elapsed / max(1, seen)}


def runexperiment(cfg: ExpConfig) -> None:
    setseed(cfg.seed)
    Path(cfg.artifactsdir).mkdir(parents=True, exist_ok=True)
    device = cfg.device
    print(f"\nConfig:")
    for key, value in asdict(cfg).items():
        print(f"  {key:20s} = {value}")
    print(f"\nEnvironment:")
    for key, value in collectenv(cfg).items():
        print(f"  {key:20s} = {value}")
    builder = BatchBuilder(cfg)
    model = QwenVLM(cfg)
    if builder.addedtokens > 0:
        model.language.resize_token_embeddings(len(builder.tokenizer))
    model.imagetokenid = builder.imagetokenid
    model.to(device)
    if cfg.initadapterpath:
        model.adapter.load_state_dict(torch.load(cfg.initadapterpath, map_location=device))
        print(f"loaded initial adapter from {cfg.initadapterpath}")
    nadapter = sum(p.numel() for p in model.adapter.parameters() if p.requires_grad)
    nfrozen = sum(p.numel() for p in model.vision.parameters()) + sum(p.numel() for p in model.language.parameters())
    print(f"\nParameter budget:")
    print(f"adapter (trainable): {nadapter / 1e6:.2f}M")
    print(f"frozen (vision+LM): {nfrozen / 1e6:.2f}M")
    print(f"trainable fraction: {100 * nadapter / (nadapter + nfrozen):.3f}%")
    traindataset = MathVQADataset(cfg.manifestpath, split="train")
    devdataset = MathVQADataset(cfg.manifestpath, split="dev")
    print(f"\nData: train={len(traindataset)}, dev={len(devdataset)}")
    encoded = [builder.encodesample(traindataset[i]) for i in range(len(traindataset))]
    trainloader = DataLoader(encoded, batch_size=cfg.batchsize, shuffle=True, collate_fn=builder.collate)
    baseline_label = "initial adapter" if cfg.initadapterpath else "random adapter"
    print(f"Baseline evaluation ({baseline_label}, before training)")
    baselinereport = evaluate(model, builder, devdataset, device, cfg.evalmaxsamples)
    printevalreport("baseline", baselinereport)
    trainresult = train(cfg, model, builder, trainloader, devdataset, device)
    if Path(cfg.adapterpath).exists():
        model.adapter.load_state_dict(torch.load(cfg.adapterpath, map_location=device))
        print(f"loaded best checkpoint from {cfg.adapterpath}")
    finalreport = evaluate(model, builder, devdataset, device, cfg.evalmaxsamples)
    printevalreport("final (best checkpoint)", finalreport)
    cputiming = measurecputime(model, trainloader, cfg.cputimingsteps, cfg.gradaccum)
    mpssecperstep = trainresult["train_seconds"] / max(1, len(trainresult["losslog"]))
    if cputiming["cpu_steps"] > 0:
        speedup = cputiming["cpu_sec_per_step"] / max(1e-9, mpssecperstep)
        print(f"CPU vs MPS timing comparison")
        print(f"MPS: {mpssecperstep:.3f} sec/step (averaged over training)")
        print(f"CPU: {cputiming['cpu_sec_per_step']:.3f} sec/step on {cputiming['cpu_steps']} steps")
        print(f"speedup MPS over CPU: {speedup:.2f}x")
    else:
        speedup = None
        print(f"CPU timing skipped")
    torch.save(model.adapter.state_dict(), cfg.finaladapterpath)
    with open(cfg.predictionspath, "w") as f:
        for pred in finalreport["predictions"]:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    print(f"\nFinal adapter saved to {cfg.finaladapterpath}")
    print(f"Dev predictions saved to {cfg.predictionspath}")
    summary = {
        "config": asdict(cfg),
        "environment": collectenv(cfg),
        "params": {
            "adapter_trainable_m": nadapter / 1e6,
            "frozen_m": nfrozen / 1e6,
        },
        "data": {"train": len(traindataset), "dev": len(devdataset)},
        "baseline_eval": {k: v for k, v in baselinereport.items() if k != "predictions"},
        "final_eval": {k: v for k, v in finalreport.items() if k != "predictions"},
        "improvement": finalreport["overall"] - baselinereport["overall"],
        "training": {
            "losslog": trainresult["losslog"],
            "epoch_evals": [{"epoch": e["epoch"], "avg_loss": e["avg_loss"],
                             "eval_overall": e["eval"]["overall"]} for e in trainresult["epoch_evals"]],
            "train_seconds": trainresult["train_seconds"],
            "best_acc": trainresult["best_acc"],
        },
        "timing": {**cputiming, "mps_sec_per_step": mpssecperstep, "speedup": speedup},
    }
    with open(cfg.logpath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"full log saved to {cfg.logpath}")
    print(f"res: baseline {baselinereport['overall']:.4f} -> final {finalreport['overall']:.4f}")
    print(f"improvement: {(finalreport['overall'] - baselinereport['overall']) * 100:+.1f} percentage points")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cpu"])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32", "float16"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--numimagetokens", type=int, default=16)
    parser.add_argument("--init-adapter", type=str, default=None)
    parser.add_argument("--cpu-timing-steps", type=int, default=10)
    parser.add_argument("--blank-uniform-weight", type=float, default=0.0)
    args = parser.parse_args()
    cfg = ExpConfig(
        device=args.device,
        epochs=args.epochs,
        dtype=args.dtype,
        learningrate=args.lr,
        numimagetokens=args.numimagetokens,
        initadapterpath=args.init_adapter,
        cputimingsteps=args.cpu_timing_steps,
        blankuniformweight=args.blank_uniform_weight,
    )
    runexperiment(cfg)


if __name__ == "__main__":
    main()
