import json
import os
import random
import re
import string
from copy import deepcopy
from dataclasses import dataclass
from os.path import join as pjoin
from shutil import copyfile
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from datasets import Audio, load_dataset, load_metric
from IPython.display import HTML, display
from tqdm import tqdm
from transformers import (
    Data2VecAudioForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    get_linear_schedule_with_warmup,
)

EXP_FOLDER = "logdirs/torch_asr_on_ukrainian_data2vec_cosinev3"
BASE_MODEL_PATH = "Respeecher/ukrainian-data2vec"


def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    display(HTML(df.to_html()))


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


def compute_wer_stats(pred_logits, label_ids, processor):
    pred_ids = np.argmax(pred_logits, axis=-1)

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(label_ids, group_tokens=False)

    return pred_str, label_str


def torch_loop(
    dataloader,
    inp_model,
    inp_optimizer,
    inp_processor,
    inp_scheduler=None,
    inp_scaler=None,
    mode="train",
    device="cpu",
):
    if mode == "train":
        inp_model.train()
    else:
        inp_model.eval()
    all_pred_strs = []
    all_label_strs = []
    all_losses = []
    with torch.inference_mode(mode=(mode != "train")):
        for text in tqdm(dataloader):
            if mode == "train":
                inp_optimizer.zero_grad()
            text = {k: v.to(device) for k, v in text.items()}
            with torch.cuda.amp.autocast(enabled=inp_scaler is not None):
                model_output = inp_model(**text)
                loss = model_output["loss"]
            if mode == "train":
                # TODO: Try gradient clipping
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                if inp_scaler is not None:
                    inp_scaler.scale(loss).backward()
                    inp_scaler.step(inp_optimizer)
                    scale = inp_scaler.get_scale()
                    inp_scaler.update()
                else:
                    loss.backward()
                    inp_optimizer.step()
                if inp_scheduler is not None:
                    # Solution Taken from here
                    # https://discuss.pytorch.org/t/optimizer-step-before-lr-scheduler-step-error-using-gradscaler/92930/8
                    if inp_scaler is not None:
                        if scale <= inp_scaler.get_scale():
                            inp_scheduler.step()
                    else:
                        inp_scheduler.step()
            pred_strs, label_strs = compute_wer_stats(
                model_output["logits"].detach().cpu(), text["labels"].detach().cpu(), processor=inp_processor
            )
            all_pred_strs.extend(pred_strs)
            all_label_strs.extend(label_strs)
            all_losses.append(model_output["loss"].detach().cpu().numpy())

    wer_metric = load_metric("wer")
    wer = wer_metric.compute(predictions=all_pred_strs, references=all_label_strs)

    return wer, np.mean(all_losses)


def run_train_valid(
    train_torch_dataloader,
    valid_torch_dataloader,
    inp_model,
    inp_optimizer,
    inp_processor,
    use_amp=True,
    inp_scheduler=None,
    n_epochs=5,
    load_best_on_end=True,
    device="cuda",
):

    train_all_epoch_losses = []
    train_all_epoch_wers = []
    valid_all_epoch_losses = []
    valid_all_epoch_wers = []

    best_metric = np.inf
    best_model_state_dict = None

    inp_scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for epoch in range(1, n_epochs + 1):
        print(f"Starting Epoch {epoch}")
        print("Train phase")
        train_epoch_wer, train_epoch_loss = torch_loop(
            dataloader=train_torch_dataloader,
            inp_model=inp_model,
            inp_optimizer=inp_optimizer,
            # inp_scheduler=inp_scheduler,
            inp_scaler=inp_scaler,
            inp_processor=inp_processor,
            device=device,
            mode="train",
        )
        print(f"Train metrics. WER: {train_epoch_wer}. Loss: {train_epoch_loss}")
        print("Valid phase")
        valid_epoch_wer, valid_epoch_loss = torch_loop(
            dataloader=valid_torch_dataloader,
            inp_model=inp_model,
            inp_optimizer=inp_optimizer,
            inp_scaler=inp_scaler,
            inp_processor=inp_processor,
            device=device,
            mode="eval",
        )
        print(f"Valid metrics. WER: {valid_epoch_wer}. Loss: {valid_epoch_loss}")
        if valid_epoch_wer < best_metric:
            best_metric = valid_epoch_wer
            best_model_state_dict = deepcopy(inp_model.state_dict())

        train_all_epoch_losses.append(train_epoch_loss)
        train_all_epoch_wers.append(train_epoch_wer)

        valid_all_epoch_losses.append(valid_epoch_loss)
        valid_all_epoch_wers.append(valid_epoch_wer)

        if inp_scheduler is not None:
            inp_scheduler.step()

    if load_best_on_end:
        print(f"Loading best model with valid WER {best_metric}")
        inp_model.load_state_dict(best_model_state_dict)

    return (train_all_epoch_losses, train_all_epoch_wers, valid_all_epoch_losses, valid_all_epoch_wers)


def main():
    # Fix seed
    torch.manual_seed(42)
    torch.backends.cuda.deterministic = True
    # Load dataset
    common_voice_uk = load_dataset("mozilla-foundation/common_voice_11_0", "uk")
    # Drop useless columns
    common_voice_uk = common_voice_uk.remove_columns(
        ["client_id", "up_votes", "down_votes", "age", "gender", "accent", "locale", "segment"]
    )
    # Clean Text
    regex = re.compile(
        "[%s]"
        % re.escape(
            string.punctuation
            + "".join(
                [
                    "—",
                    "–",
                    # '’',
                    "“",
                    "”",
                    "…",
                    "«",
                    "»",
                ]
            )
        )
    )

    def remove_special_characters(batch):
        batch["sentence"] = regex.sub("", batch["sentence"]).lower()
        return batch

    common_voice_uk = common_voice_uk.map(remove_special_characters)
    # Compose Vocab
    def extract_all_chars(batch):
        all_text = " ".join(batch["sentence"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    vocabs = common_voice_uk.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=common_voice_uk.column_names["train"],
    )
    vocab_list = list(
        set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]) | set(vocabs["validation"]["vocab"][0])
    )
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    print(f"Vocab contains {len(vocab_dict)} chars")
    print("Vocab")
    print(vocab_dict)
    os.makedirs(EXP_FOLDER)
    with open(pjoin(EXP_FOLDER, "vocab.json"), "w") as vocab_file:
        json.dump(vocab_dict, vocab_file)
    # Copy train.py to logdir
    copyfile(__file__, pjoin(EXP_FOLDER, "train.py"))
    # Prepare Preprocessing
    tokenizer = Wav2Vec2CTCTokenizer(
        pjoin(EXP_FOLDER, "vocab.json"), unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    # Resample
    common_voice_uk = common_voice_uk.cast_column("audio", Audio(sampling_rate=16_000))
    # Process Dataset
    def prepare_dataset(batch):
        audio = batch["audio"]

        # batched output is "un-batched" to ensure mapping is correct
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]

        with processor.as_target_processor():
            batch["labels"] = processor(batch["sentence"]).input_ids
        return batch

    common_voice_uk = common_voice_uk.map(
        prepare_dataset, remove_columns=common_voice_uk.column_names["train"], num_proc=4
    )
    # Create Data Collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    # Setup Dataloaders
    train_torch_dataloader = torch.utils.data.DataLoader(
        common_voice_uk["train"],
        batch_size=16,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=data_collator,
    )
    valid_torch_dataloader = torch.utils.data.DataLoader(
        common_voice_uk["validation"],
        batch_size=16,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=data_collator,
    )
    # Setup Model
    model = Data2VecAudioForCTC.from_pretrained(
        BASE_MODEL_PATH,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=processor.tokenizer.vocab_size,
    ).to("cuda")
    # Setup Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        [
            {"params": model.lm_head.parameters(), "lr": 1e-3},
            {"params": model.data2vec_audio.encoder.parameters(), "lr": 1e-5},
        ],
        weight_decay=0,
    )
    n_epochs = 30
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=int(0.05 * len(train_torch_dataloader) * n_epochs),
    #     num_training_steps=len(train_torch_dataloader) * n_epochs,
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer, T_0=n_epochs, T_mult=1, eta_min=1e-7, last_epoch=-1
    )
    # Train
    (train_all_epoch_losses, train_all_epoch_wers, valid_all_epoch_losses, valid_all_epoch_wers) = run_train_valid(
        train_torch_dataloader=train_torch_dataloader,
        valid_torch_dataloader=valid_torch_dataloader,
        inp_model=model,
        inp_processor=processor,
        inp_optimizer=optimizer,
        use_amp=True,
        inp_scheduler=scheduler,
        n_epochs=n_epochs,
        load_best_on_end=True,
        device="cuda",
    )
    # Evaluate
    model.eval()

    def map_to_result(batch):
        with torch.no_grad():
            input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
            logits = model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        batch["pred_str"] = processor.batch_decode(pred_ids)[0]
        batch["text"] = processor.decode(batch["labels"], group_tokens=False)
        return batch

    # Do not use cache, because you may take previous results
    wer_metric = load_metric("wer")
    test_results = common_voice_uk["test"].map(
        map_to_result, remove_columns=common_voice_uk["test"].column_names, load_from_cache_file=False
    )
    test_wer_score = wer_metric.compute(predictions=test_results["pred_str"], references=test_results["text"])
    valid_results = common_voice_uk["validation"].map(
        map_to_result, remove_columns=common_voice_uk["validation"].column_names, load_from_cache_file=False
    )
    valid_wer_score = wer_metric.compute(predictions=valid_results["pred_str"], references=valid_results["text"])
    with open(pjoin(EXP_FOLDER, "metric.json"), "w") as vocab_file:
        json.dump(
            {
                "TEST_WER": str(test_wer_score),
                "VALID_WER": str(valid_wer_score),
                "train_all_epoch_losses": [str(el) for el in train_all_epoch_losses],
                "train_all_epoch_wers": [str(el) for el in train_all_epoch_wers],
                "valid_all_epoch_losses": [str(el) for el in valid_all_epoch_losses],
                "valid_all_epoch_wers": [str(el) for el in valid_all_epoch_wers],
            },
            vocab_file,
        )
    model.save_pretrained(EXP_FOLDER)
    processor.save_pretrained(EXP_FOLDER)
    print("Test WER: {:.3f}".format(test_wer_score))
    print("Valid WER: {:.3f}".format(valid_wer_score))


if __name__ == "__main__":
    main()
