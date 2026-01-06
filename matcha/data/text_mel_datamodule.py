import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torchaudio as ta
from torch import Tensor
from torch.utils.data.dataloader import DataLoader

from matcha.text import text_to_sequence
from matcha.utils.audio import mel_spectrogram
from matcha.utils.model import fix_len_compatibility, normalize
from matcha.utils.utils import intersperse


def parse_filelist(filelist_path: str, split_char: str = "|") -> list[list[str]]:
    with open(filelist_path, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


class TextMelDataModule:
    """Data module for text-mel dataset without PyTorch Lightning dependency."""

    def __init__(
        self,
        name: str,
        train_filelist_path: str,
        valid_filelist_path: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        cleaners: list[str],
        add_blank: bool,
        n_spks: int,
        n_fft: int,
        n_feats: int,
        sample_rate: int,
        hop_length: int,
        win_length: int,
        f_min: float,
        f_max: float,
        data_statistics: Optional[dict[str, float]],
        seed: int,
        load_durations: bool,
    ) -> None:
        # Store hyperparameters as a simple namespace-like object
        self.hparams = type(
            "Hparams",
            (),
            {
                "name": name,
                "train_filelist_path": train_filelist_path,
                "valid_filelist_path": valid_filelist_path,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "cleaners": cleaners,
                "add_blank": add_blank,
                "n_spks": n_spks,
                "n_fft": n_fft,
                "n_feats": n_feats,
                "sample_rate": sample_rate,
                "hop_length": hop_length,
                "win_length": win_length,
                "f_min": f_min,
                "f_max": f_max,
                "data_statistics": data_statistics,
                "seed": seed,
                "load_durations": load_durations,
            },
        )()

        self.trainset: Optional[TextMelDataset] = None
        self.validset: Optional[TextMelDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.trainset`, `self.validset`."""
        # load and split datasets only if not loaded already

        self.trainset = TextMelDataset(
            self.hparams.train_filelist_path,
            self.hparams.n_spks,
            self.hparams.cleaners,
            self.hparams.add_blank,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
            self.hparams.load_durations,
        )
        self.validset = TextMelDataset(
            self.hparams.valid_filelist_path,
            self.hparams.n_spks,
            self.hparams.cleaners,
            self.hparams.add_blank,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
            self.hparams.load_durations,
        )

    def train_dataloader(self) -> DataLoader[dict[str, Any]]:
        assert self.trainset is not None, "Call setup() before train_dataloader()"
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks),
        )

    def val_dataloader(self) -> DataLoader[dict[str, Any]]:
        assert self.validset is not None, "Call setup() before val_dataloader()"
        return DataLoader(
            dataset=self.validset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks),
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up after fit or test."""
        pass  # pylint: disable=unnecessary-pass

    def state_dict(self) -> dict[str, Any]:
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Things to do when loading checkpoint."""
        pass  # pylint: disable=unnecessary-pass


class TextMelDataset(torch.utils.data.Dataset[dict[str, Any]]):
    def __init__(
        self,
        filelist_path: str,
        n_spks: int,
        cleaners: list[str],
        add_blank: bool = True,
        n_fft: int = 1024,
        n_mels: int = 80,
        sample_rate: int = 22050,
        hop_length: int = 256,
        win_length: int = 1024,
        f_min: float = 0.0,
        f_max: float = 8000,
        data_parameters: Optional[dict[str, float]] = None,
        seed: Optional[int] = None,
        load_durations: bool = False,
    ) -> None:
        self.filepaths_and_text: list[list[str]] = parse_filelist(filelist_path)
        self.n_spks: int = n_spks
        self.cleaners: list[str] = cleaners
        self.add_blank: bool = add_blank
        self.n_fft: int = n_fft
        self.n_mels: int = n_mels
        self.sample_rate: int = sample_rate
        self.hop_length: int = hop_length
        self.win_length: int = win_length
        self.f_min: float = f_min
        self.f_max: float = f_max
        self.load_durations: bool = load_durations
        self.data_parameters: dict[str, float] = (
            data_parameters
            if data_parameters is not None
            else {"mel_mean": 0.0, "mel_std": 1.0}
        )
        random.seed(seed)
        random.shuffle(self.filepaths_and_text)

    def get_datapoint(self, filepath_and_text: list[str]) -> dict[str, Any]:
        spk: Optional[int]
        if self.n_spks > 1:
            filepath, spk, text = (
                filepath_and_text[0],
                int(filepath_and_text[1]),
                filepath_and_text[2],
            )
        else:
            filepath, text = filepath_and_text[0], filepath_and_text[1]
            spk = None

        text_tensor, cleaned_text = self.get_text(text, add_blank=self.add_blank)
        mel = self.get_mel(filepath)

        durations = self.get_durations(filepath, text_tensor) if self.load_durations else None

        return {
            "x": text_tensor,
            "y": mel,
            "spk": spk,
            "filepath": filepath,
            "x_text": cleaned_text,
            "durations": durations,
        }

    def get_durations(self, filepath: str, text: Tensor) -> Tensor:
        filepath_path = Path(filepath)
        data_dir, name = filepath_path.parent.parent, filepath_path.stem
        dur_loc = data_dir / "durations" / f"{name}.npy"

        try:
            durs: Tensor = torch.from_numpy(np.load(dur_loc).astype(int))
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Tried loading the durations but durations didn't exist at {dur_loc}, make sure you've generate the durations first using: python matcha/utils/get_durations_from_trained_model.py \n"
            ) from e

        assert len(durs) == len(
            text
        ), f"Length of durations {len(durs)} and text {len(text)} do not match"

        return durs

    def get_mel(self, filepath: str) -> Tensor:
        audio, sr = ta.load(filepath)
        assert sr == self.sample_rate
        mel = mel_spectrogram(
            audio,
            self.n_fft,
            self.n_mels,
            self.sample_rate,
            self.hop_length,
            self.win_length,
            self.f_min,
            self.f_max,
            center=False,
        ).squeeze()
        mel = normalize(
            mel, self.data_parameters["mel_mean"], self.data_parameters["mel_std"]
        )
        return mel

    def get_text(self, text: str, add_blank: bool = True) -> tuple[Tensor, str]:
        text_norm, cleaned_text = text_to_sequence(text, self.cleaners)
        if self.add_blank:
            text_norm = intersperse(text_norm, 0)
        text_tensor = torch.IntTensor(text_norm)
        return text_tensor, cleaned_text

    def __getitem__(self, index: int) -> dict[str, Any]:
        datapoint = self.get_datapoint(self.filepaths_and_text[index])
        return datapoint

    def __len__(self) -> int:
        return len(self.filepaths_and_text)


class TextMelBatchCollate:
    def __init__(self, n_spks: int) -> None:
        self.n_spks: int = n_spks

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        B = len(batch)
        y_max_len = max(
            [item["y"].shape[-1] for item in batch]
        )  # pylint: disable=consider-using-generator
        y_max_length: int = int(fix_len_compatibility(y_max_len))
        x_max_length: int = max(
            [item["x"].shape[-1] for item in batch]
        )  # pylint: disable=consider-using-generator
        n_feats: int = int(batch[0]["y"].shape[-2])

        y = torch.zeros((B, n_feats, int(y_max_length)), dtype=torch.float32)
        x = torch.zeros((B, int(x_max_length)), dtype=torch.long)
        durations = torch.zeros((B, int(x_max_length)), dtype=torch.long)

        y_lengths: list[int] = []
        x_lengths: list[int] = []
        spks: list[Optional[int]] = []
        filepaths: list[str] = []
        x_texts: list[str] = []
        for i, item in enumerate(batch):
            y_, x_ = item["y"], item["x"]
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, : y_.shape[-1]] = y_
            x[i, : x_.shape[-1]] = x_
            spks.append(item["spk"])
            filepaths.append(item["filepath"])
            x_texts.append(item["x_text"])
            if item["durations"] is not None:
                durations[i, : item["durations"].shape[-1]] = item["durations"]

        y_lengths_tensor = torch.tensor(y_lengths, dtype=torch.long)
        x_lengths_tensor = torch.tensor(x_lengths, dtype=torch.long)
        spks_tensor: Optional[Tensor] = torch.tensor(spks, dtype=torch.long) if self.n_spks > 1 else None

        return {
            "x": x,
            "x_lengths": x_lengths_tensor,
            "y": y,
            "y_lengths": y_lengths_tensor,
            "spks": spks_tensor,
            "filepaths": filepaths,
            "x_texts": x_texts,
            "durations": durations if not torch.eq(durations, 0).all() else None,
        }
