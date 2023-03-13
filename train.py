import json
import os
import random
import re
import string
from dataclasses import dataclass
from os.path import join as pjoin
from shutil import copyfile
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from datasets import Audio, load_dataset, load_metric
from IPython.display import HTML, display
from transformers import (
    Data2VecAudioForCTC,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

EXP_FOLDER = "..."
BASE_MODEL_PATH = "..."


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


def main():
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
    # Create metric
    wer_metric = load_metric("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    # Upload PreTrained model and freeze
    model = Data2VecAudioForCTC.from_pretrained(
        BASE_MODEL_PATH,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=processor.tokenizer.vocab_size,
    )
    model.freeze_feature_encoder()
    # Configure Trainer
    training_args = TrainingArguments(
        output_dir=EXP_FOLDER,
        group_by_length=True,
        per_device_train_batch_size=32,
        evaluation_strategy="steps",
        num_train_epochs=30,
        fp16=True,
        gradient_checkpointing=True,
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=True,
    )
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=common_voice_uk["train"],
        eval_dataset=common_voice_uk["validation"],
        tokenizer=processor.feature_extractor,
    )
    # Train !!!!!!!!!!!!!!!!!!
    trainer.train()
    # Evaluate
    def map_to_result(batch):
        with torch.no_grad():
            input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
            logits = trainer.model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        batch["pred_str"] = processor.batch_decode(pred_ids)[0]
        batch["text"] = processor.decode(batch["labels"], group_tokens=False)
        return batch

    results = common_voice_uk["test"].map(map_to_result, remove_columns=common_voice_uk["test"].column_names)
    wer_score = wer_metric.compute(predictions=results["pred_str"], references=results["text"])
    with open(pjoin(EXP_FOLDER, "metric.json"), "w") as vocab_file:
        json.dump({"WER": wer_score}, vocab_file)
    print("Test WER: {:.3f}".format(wer_score))


if __name__ == "__main__":
    main()
