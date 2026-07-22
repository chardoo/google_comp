from google.colab import drive
drive.mount('/content/drive')

import os, glob
WORK_DIR = '/content/drive/MyDrive/waxal_whisper'
print("WORK_DIR exists:", os.path.exists(WORK_DIR))
print()
# show everything under waxal_whisper
for root, dirs, files in os.walk(WORK_DIR):
    depth = root.replace(WORK_DIR, "").count(os.sep)
    if depth <= 2:
        print(root)
        for d in dirs: print("   [dir]", d)

# WAXAL ASR v2 - Multilingual Whisper Fine-tuning
### Memory-optimized for Google Colab T4

Fixes applied vs. the previous version:
- Notebook JSON was truncated/invalid -> rebuilt complete
- Removed 8-bit quantization for training (incompatible with plain `Trainer` + fp16)
- Kept the *raw* eval set alive so the baseline comparison can read `audio`/`transcription`
- `Dataset.from_generator` now uses `gen_kwargs` (picklable, cacheable)
- Gradient checkpointing uses `use_reentrant=False`
- `compute_metrics` no longer mutates `label_ids` in place


# Cell 1: Setup & Mount Drive
import os
try:
    from google.colab import drive
    drive.mount('/content/drive')
except (ImportError, ValueError):
    print("(not in Colab, or already mounted)")

WORK_DIR = '/content/drive/MyDrive/waxal_whisper' if os.path.isdir('/content/drive') else './waxal_whisper'
os.makedirs(WORK_DIR, exist_ok=True)
print('Working dir:', WORK_DIR)

# Cell 2: Install dependencies
!pip install -q -U "transformers>=4.44" datasets accelerate jiwer evaluate librosa soundfile psutil
print("Dependencies installed")

# Cell 3: Hugging Face login
from huggingface_hub import login
login()

# Cell 4: Configuration
MODEL_ID   = "openai/whisper-small"
DATASET_ID = "google/WaxalNLP"
LANGS      = ["lin", "sna", "lug"]

# ----- DATA -----
PER_LANG_TRAIN = 300
PER_LANG_EVAL  = 50

# ----- QUALITY FILTER -----
MIN_CHARS          = 5
MAX_TOKENS_APPROX  = 80
MAX_NONLETTER_FRAC = 0.15

# ----- AUGMENTATION -----
AUG_PROB       = 0.5
TELEPHONE_PROB = 0.1   # applied *inside* augment(), independent of AUG_PROB

# ----- TRAINING -----
BATCH       = 4
GRAD_ACCUM  = 8
MAX_STEPS   = 1000
SAVE_EVERY  = 250
EVAL_EVERY  = 250
LR          = 1e-5
EARLY_STOP_PATIENCE = 4

# ----- DECODING -----
NUM_BEAMS         = 3
REPETITION_PENALTY = 1.15
NO_REPEAT_NGRAM   = 4

OUT_DIR = f"{WORK_DIR}/whisper-waxal-v2"
os.makedirs(OUT_DIR, exist_ok=True)
print("Model:", MODEL_ID, "| out:", OUT_DIR)

# Cell 5: Memory utilities
import gc, psutil, torch, numpy as np

def memory_status(stage=""):
    print(f"--- Memory {stage} ---")
    if torch.cuda.is_available():
        print(f"  GPU allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"  GPU reserved : {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    vm = psutil.virtual_memory()
    print(f"  RAM used: {vm.used/1024**3:.2f} GB | available: {vm.available/1024**3:.2f} GB")

def cleanup_memory(stage="after cleanup"):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    memory_status(stage)

memory_status("initial")

# Cell 6: Text normalization & quality filter
import re

# NOTE: the original regex required an opening bracket AND a closing bracket, but
# listed "\n" and "(" as openers while allowing "]" as a closer -> mismatched pairs.
NOISE_MARKER = re.compile(r"[\[\(<]\s*(noise|inaudible|unclear|silence|music|laugh|cough|unintelligible)[^\]\)>]*[\]\)>]", re.I)

def normalize_text(t) -> str:
    return re.sub(r"\s+", " ", str(t).strip())

def is_good_transcript(t) -> bool:
    if t is None:
        return False
    t = str(t).strip()
    if len(t) < MIN_CHARS:
        return False
    if NOISE_MARKER.search(t):
        return False
    if len(t.split()) > MAX_TOKENS_APPROX:
        return False
    letters = sum(ch.isalpha() or ch.isspace() for ch in t)
    if letters / max(len(t), 1) < (1 - MAX_NONLETTER_FRAC):
        return False
    return True

for s in ["Mombe idzo dziri kuratidza", "[noise]", "ok", "###@@@!!! ###@@@"]:
    print(f"  {s!r:32} -> {is_good_transcript(s)}")

# Cell 7: Streaming dataset loader
from datasets import (get_dataset_config_names, Dataset, concatenate_datasets,
                      load_dataset, Audio)

_CONFIGS = None

def resolve_config(lang):
    global _CONFIGS
    if _CONFIGS is None:
        _CONFIGS = get_dataset_config_names(DATASET_ID)
    cands = sorted([c for c in _CONFIGS if c.startswith(lang)],
                   key=lambda c: ("tts" in c, "asr" not in c, len(c)))
    return cands[0] if cands else f"{lang}_asr"

# Top-level generator + gen_kwargs => picklable and hashable for the datasets cache.
def _gen_examples(lang, cfg, split, n):
    s = load_dataset(DATASET_ID, cfg, split=split, streaming=True)
    s = s.cast_column("audio", Audio(sampling_rate=16_000))
    seen = kept = 0
    for ex in s:
        seen += 1
        if kept >= n or seen > n * 5:
            break
        if not is_good_transcript(ex.get("transcription")):
            continue
        kept += 1
        yield {
            "audio": np.asarray(ex["audio"]["array"], dtype=np.float32),
            "transcription": ex["transcription"],
            "language": lang,
            "speaker_id": str(ex.get("speaker_id", f"{lang}_spk_{kept}")),
        }
        if kept % 50 == 0:
            gc.collect()

def stream_take(lang, split, n):
    cfg = resolve_config(lang)
    print(f"  loading {lang}/{split} from config {cfg}")
    return Dataset.from_generator(
        _gen_examples,
        gen_kwargs={"lang": lang, "cfg": cfg, "split": split, "n": n},
    )

print("Loader ready")

# Cell 8: Load data
train_parts, eval_parts = [], []
for lang in LANGS:
    print(f"\n{lang}")
    train_parts.append(stream_take(lang, "train", PER_LANG_TRAIN))
    cleanup_memory(lang + " train")
    try:
        eval_parts.append(stream_take(lang, "validation", PER_LANG_EVAL))
    except Exception as e:
        print(f"  no validation split ({type(e).__name__}), falling back to train")
        eval_parts.append(stream_take(lang, "train", PER_LANG_EVAL))
    cleanup_memory(lang + " eval")

train_ds = concatenate_datasets(train_parts).shuffle(seed=42)
eval_ds  = concatenate_datasets(eval_parts)          # KEEP THIS - Cell 17 needs raw audio
print(f"\ntrain: {len(train_ds)} | eval: {len(eval_ds)}")

# Cell 9: Audio augmentation
import librosa

rng = np.random.default_rng(42)

def augment(arr):
    a = np.asarray(arr, dtype=np.float32)
    if rng.random() < 0.5:                       # speed perturbation
        a = librosa.effects.time_stretch(a, rate=float(rng.uniform(0.9, 1.1)))
    if rng.random() < 0.5:                       # additive noise at target SNR
        snr_db = float(rng.uniform(10, 30))
        rms = np.sqrt(np.mean(a ** 2)) + 1e-9
        noise_rms = rms / (10 ** (snr_db / 20))
        a = a + rng.normal(0, noise_rms, a.shape).astype(np.float32)
    if rng.random() < TELEPHONE_PROB:            # narrowband / telephone
        a = librosa.resample(a, orig_sr=16_000, target_sr=8_000)
        a = librosa.resample(a, orig_sr=8_000,  target_sr=16_000)
    return np.clip(a, -1.0, 1.0).astype(np.float32)

_s = np.asarray(train_ds[0]["audio"], dtype=np.float32)
print(f"original {len(_s)} -> augmented {len(augment(_s))}")

# Cell 10: Processor & feature preparation
from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained(MODEL_ID, language=None, task="transcribe")

def make_prepare(do_augment: bool):
    def prepare(example):
        arr = np.asarray(example["audio"], dtype=np.float32)
        if do_augment and rng.random() < AUG_PROB:
            arr = augment(arr)
        out = {}
        out["input_features"] = processor.feature_extractor(
            arr, sampling_rate=16_000).input_features[0]
        out["labels"] = processor.tokenizer(
            normalize_text(example["transcription"])).input_ids
        return out
    return prepare

def preprocess(dataset, do_augment):
    """Straight .map with a small writer_batch_size keeps peak RAM low."""
    return dataset.map(
        make_prepare(do_augment),
        remove_columns=dataset.column_names,
        writer_batch_size=32,
        desc="augment+featurize" if do_augment else "featurize",
    )

# Cell 11: Preprocess
train_prep = preprocess(train_ds, do_augment=True)
eval_prep  = preprocess(eval_ds,  do_augment=False)
print(f"train_prep: {len(train_prep)} | eval_prep: {len(eval_prep)}")

del train_parts, eval_parts, train_ds   # eval_ds stays: Cell 17 reads raw audio from it
cleanup_memory()

# Cell 12: Data collator
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features],
            return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)
        if self.decoder_start_token_id is not None and \
           (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
print("Collator ready")

# Cell 13: Load model
from transformers import WhisperForConditionalGeneration

# 8-bit weights are frozen int8 -> they cannot be fine-tuned by a plain Seq2SeqTrainer,
# and load_in_8bit + fp16=True raises. Load in fp32 and let fp16 autocast handle memory.
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
if torch.cuda.is_available():
    model = model.to("cuda")

model.generation_config.language = None
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

model.generation_config.num_beams = NUM_BEAMS
model.generation_config.repetition_penalty = REPETITION_PENALTY
model.generation_config.no_repeat_ngram_size = NO_REPEAT_NGRAM

model.config.apply_spec_augment = True
model.config.mask_time_prob = 0.05
model.config.mask_feature_prob = 0.05

collator.decoder_start_token_id = model.config.decoder_start_token_id
cleanup_memory("after model load")

# Cell 14: Metrics
import evaluate
wer_m = evaluate.load("wer")
cer_m = evaluate.load("cer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    label_ids = np.where(pred.label_ids == -100,
                         processor.tokenizer.pad_token_id, pred.label_ids)

    pred_str  = [normalize_text(p) for p in processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)]
    label_str = [normalize_text(l) for l in processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

    keep = [(p, l) for p, l in zip(pred_str, label_str) if l.strip()]
    if not keep:
        return {"wer": 1.0, "cer": 1.0, "score": 1.0}
    pred_str, label_str = map(list, zip(*keep))
    wer = wer_m.compute(predictions=pred_str, references=label_str)
    cer = cer_m.compute(predictions=pred_str, references=label_str)
    return {"wer": wer, "cer": cer, "score": 0.5 * wer + 0.5 * cer}

# Cell 15: Training setup
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
import glob

args = Seq2SeqTrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_steps=50,
    max_steps=MAX_STEPS,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    fp16=torch.cuda.is_available(),
    eval_strategy="steps",
    save_strategy="steps",          # must match eval_strategy for load_best_model_at_end
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=SAVE_EVERY,
    eval_steps=EVAL_EVERY,
    logging_steps=50,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="score",
    greater_is_better=False,
    save_total_limit=1,
    seed=42,
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    remove_unused_columns=False,    # our collator needs the raw dict keys
    optim="adamw_torch",
    max_grad_norm=1.0,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_prep,
    eval_dataset=eval_prep,
    data_collator=collator,
    compute_metrics=compute_metrics,
    processing_class=processor,     # use tokenizer=processor on transformers < 4.46
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
)

ckpts = sorted(glob.glob(f"{OUT_DIR}/checkpoint-*"), key=lambda p: int(p.split("-")[-1]))
resume = ckpts[-1] if ckpts else None
print("Resume from:", resume)

# Cell 16: Train
memory_status("before training")
try:
    trainer.train(resume_from_checkpoint=resume)
except torch.cuda.OutOfMemoryError:
    print("OOM. Try BATCH=2, PER_LANG_TRAIN lower, or whisper-tiny/base.")
    cleanup_memory()
    raise

trainer.save_model(f"{OUT_DIR}/final")
processor.save_pretrained(f"{OUT_DIR}/final")
cleanup_memory("after training")
print("Saved to", f"{OUT_DIR}/final")

# Cell 17: Baseline vs fine-tuned
# Reads raw waveforms from eval_ds (NOT eval_prep - that one only has
# input_features/labels, so ex["audio"] would raise KeyError).

def quick_score(mdl, proc, ds, n=30):
    mdl.eval()
    mdl.config.use_cache = True
    preds, refs = [], []
    n = min(n, len(ds))
    for i in range(n):
        ex = ds[i]
        feats = proc.feature_extractor(
            np.asarray(ex["audio"], dtype=np.float32),
            sampling_rate=16_000, return_tensors="pt"
        ).input_features.to(mdl.device, dtype=next(mdl.parameters()).dtype)
        with torch.no_grad():
            ids = mdl.generate(feats, max_new_tokens=225, num_beams=NUM_BEAMS,
                               repetition_penalty=REPETITION_PENALTY,
                               no_repeat_ngram_size=NO_REPEAT_NGRAM)
        preds.append(normalize_text(proc.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]))
        refs.append(normalize_text(ex["transcription"]))
        if i % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    w = wer_m.compute(predictions=preds, references=refs)
    c = cer_m.compute(predictions=preds, references=refs)
    return w, c, 0.5 * w + 0.5 * c

base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
if torch.cuda.is_available():
    base_model = base_model.to("cuda")
base_model.generation_config.language = None
base_model.generation_config.forced_decoder_ids = None

bw, bc, bs = quick_score(base_model, processor, eval_ds)
del base_model
cleanup_memory()

fw, fc, fs = quick_score(model, processor, eval_ds)

print(f"BASELINE  : WER={bw:.3f} CER={bc:.3f} score={bs:.3f}")
print(f"FINE-TUNED: WER={fw:.3f} CER={fc:.3f} score={fs:.3f}")
print("Fine-tuning helped." if fs < bs else "Fine-tune is worse than baseline - needs more data.")

import numpy as np
import torch
from datasets import load_dataset, Audio
print("np, torch ready")

# Cell 18: Inference on the test set
import pandas as pd

TEST_CSV   = "Test.csv"
SAMPLE_CSV = "SampleSubmission.csv"

try:
    test = pd.read_csv(TEST_CSV)
    id_col = test.columns[0]
    print(f"{len(test)} test ids | example: {test[id_col].iloc[0]}")
except FileNotFoundError:
    print("Test.csv not found - using a dummy frame")
    test = pd.DataFrame({"id": ["test_1", "test_2"]})
    id_col = "id"

def build_lookup():
    lookup, need = {}, set(test[id_col])
    for lang in LANGS:
        if not need:
            break
        cfg = resolve_config(lang)
        for split in ["test", "validation", "train"]:
            if not need:
                break
            try:
                s = load_dataset(DATASET_ID, cfg, split=split, streaming=True)
                s = s.cast_column("audio", Audio(sampling_rate=16_000))
                for ex in s:
                    ex_id = ex.get("id")
                    if ex_id in need:
                        lookup[ex_id] = np.asarray(ex["audio"]["array"], dtype=np.float32)
                        need.discard(ex_id)
                        if not need:
                            break
            except Exception as e:
                print(f"  skip {cfg}/{split}: {type(e).__name__}: {e}")
    print(f"resolved {len(lookup)}/{len(test)} audio clips")
    return lookup

lookup = build_lookup()

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

model = WhisperForConditionalGeneration.from_pretrained(f"{OUT_DIR}/final").to(
    "cuda" if torch.cuda.is_available() else "cpu")
processor = WhisperProcessor.from_pretrained(f"{OUT_DIR}/final")

# language-agnostic + anti-repetition decoding (matches training)
model.generation_config.language = None
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.generation_config.num_beams = 5
model.generation_config.repetition_penalty = 1.15
model.generation_config.no_repeat_ngram_size = 4
print("model loaded from", f"{OUT_DIR}/final")

# Cell 19: Generate predictions & write submission
import gc, numpy as np, torch, pandas as pd
model.eval()
model.config.use_cache = True

rows = []
for k, ex_id in enumerate(test[id_col].tolist()):
    audio = lookup.get(ex_id)
    if audio is None:
        rows.append({id_col: ex_id, "transcription": ""})
        continue
    feats = processor.feature_extractor(
        audio, sampling_rate=16_000, return_tensors="pt"
    ).input_features.to(model.device, dtype=next(model.parameters()).dtype)
    with torch.no_grad():
        ids = model.generate(feats, max_new_tokens=225, num_beams=NUM_BEAMS,
                             repetition_penalty=REPETITION_PENALTY,
                             no_repeat_ngram_size=NO_REPEAT_NGRAM)
    rows.append({id_col: ex_id,
                 "transcription": normalize_text(
                     processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0])})
    if k % 25 == 0:
        print(f"  {k}/{len(test)}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

sub = pd.DataFrame(rows)
try:
    sample = pd.read_csv(SAMPLE_CSV)
    sub = sub[[c for c in sample.columns if c in sub.columns]]
except FileNotFoundError:
    pass

out_csv = f"{OUT_DIR}/submission.csv"
sub.to_csv(out_csv, index=False)
print("Wrote", out_csv)
sub.head()

import pandas as pd

# rows is a list of (id, transcription) from the transcribe cell
assert 'rows' in dir() and len(rows) > 0, "rows is empty — re-run the transcribe cell first"

out = pd.DataFrame(rows, columns=["ID", "Target"])
out["Target"] = out["Target"].fillna("").astype(str)

# safety: make sure every test ID is present, in the right order
test = pd.read_csv("Test.csv")
id_col = test.columns[0]
out = test[[id_col]].rename(columns={id_col: "ID"}).merge(out, on="ID", how="left")
out["Target"] = out["Target"].fillna("")

out.to_csv("submission.csv", index=False)
out.to_csv(f"{WORK_DIR}/submission.csv", index=False)

empty = (out["Target"].str.strip() == "").sum()
print(f"rows: {len(out)} | empty: {empty}")
print(out.head(10).to_string())



