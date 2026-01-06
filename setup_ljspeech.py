#!/usr/bin/env python3
"""
Automatic LJSpeech Dataset Download and Preprocessing Script

This script automates the entire process of:
1. Downloading the LJSpeech dataset
2. Extracting the archive
3. Creating train/validation filelists
4. Computing data statistics for normalization

Usage:
    python setup_ljspeech.py
    
    # Or with custom options:
    python setup_ljspeech.py --data-dir data --val-split 0.1 --skip-download
"""

import argparse
import os
import random
import sys
import tarfile
from pathlib import Path
from typing import Dict, Tuple
from urllib.request import urlretrieve

import torch
from tqdm.auto import tqdm

# Constants
LJSPEECH_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
LJSPEECH_ARCHIVE = "LJSpeech-1.1.tar.bz2"
LJSPEECH_DIR = "LJSpeech-1.1"


class DownloadProgressBar(tqdm):
    """Progress bar for download"""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_ljspeech(data_dir: Path, force: bool = False) -> Path:
    """
    Download LJSpeech dataset
    
    Args:
        data_dir: Directory to download the dataset to
        force: Force re-download even if file exists
        
    Returns:
        Path to the downloaded archive
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / LJSPEECH_ARCHIVE
    
    if archive_path.exists() and not force:
        print(f"✓ Archive already exists: {archive_path}")
        return archive_path
    
    print(f"Downloading LJSpeech dataset from {LJSPEECH_URL}")
    print(f"This may take a while (~2.6 GB)...")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
        urlretrieve(LJSPEECH_URL, archive_path, reporthook=t.update_to)
    
    print(f"✓ Downloaded to: {archive_path}")
    return archive_path


def extract_archive(archive_path: Path, data_dir: Path, force: bool = False) -> Path:
    """
    Extract the LJSpeech archive
    
    Args:
        archive_path: Path to the archive file
        data_dir: Directory to extract to
        force: Force re-extraction even if directory exists
        
    Returns:
        Path to the extracted directory
    """
    extract_dir = data_dir / LJSPEECH_DIR
    
    if extract_dir.exists() and not force:
        print(f"✓ Dataset already extracted: {extract_dir}")
        return extract_dir
    
    print(f"Extracting {archive_path}...")
    
    with tarfile.open(archive_path, 'r:bz2') as tar:
        # Get total size for progress bar
        members = tar.getmembers()
        with tqdm(total=len(members), desc="Extracting") as pbar:
            for member in members:
                tar.extract(member, path=data_dir)
                pbar.update(1)
    
    print(f"✓ Extracted to: {extract_dir}")
    return extract_dir


def create_filelists(
    ljspeech_dir: Path,
    filelists_dir: Path,
    val_split: float = 0.1,
    seed: int = 42,
    force: bool = False
) -> Tuple[Path, Path]:
    """
    Create train and validation filelists
    
    Args:
        ljspeech_dir: Path to LJSpeech directory
        filelists_dir: Directory to save filelists
        val_split: Validation split ratio (default: 0.1 = 10%)
        seed: Random seed for reproducibility
        force: Force regeneration even if files exist
        
    Returns:
        Tuple of (train_file_path, val_file_path)
    """
    filelists_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = filelists_dir / "ljs_audio_text_train_filelist.txt"
    val_file = filelists_dir / "ljs_audio_text_val_filelist.txt"
    
    if train_file.exists() and val_file.exists() and not force:
        print(f"✓ Filelists already exist:")
        print(f"    Train: {train_file}")
        print(f"    Val:   {val_file}")
        return train_file, val_file
    
    print("Creating filelists...")
    
    # Set random seed
    random.seed(seed)
    
    # Read metadata
    metadata_file = ljspeech_dir / "metadata.csv"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    entries = []
    with open(metadata_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 2:
                audio_id = parts[0]
                # Use the normalized text (third column if exists, otherwise second column)
                text = parts[2] if len(parts) >= 3 else parts[1]
                
                # Create full path to audio file
                audio_path = str(ljspeech_dir / "wavs" / f"{audio_id}.wav")
                
                # Verify audio file exists
                if not Path(audio_path).exists():
                    print(f"Warning: Audio file not found: {audio_path}")
                    continue
                
                # Format: path|text
                entries.append(f"{audio_path}|{text}\n")
    
    if not entries:
        raise ValueError("No valid entries found in metadata")
    
    # Shuffle entries
    random.shuffle(entries)
    
    # Split into train and validation
    split_idx = int(len(entries) * (1 - val_split))
    train_entries = entries[:split_idx]
    val_entries = entries[split_idx:]
    
    # Write train filelist
    with open(train_file, "w", encoding="utf-8") as f:
        f.writelines(train_entries)
    
    # Write validation filelist
    with open(val_file, "w", encoding="utf-8") as f:
        f.writelines(val_entries)
    
    print(f"✓ Created filelists:")
    print(f"    Train: {train_file} ({len(train_entries)} samples)")
    print(f"    Val:   {val_file} ({len(val_entries)} samples)")
    
    return train_file, val_file


def compute_statistics(
    batch_size: int = 64,
    num_workers: int = 4
) -> Dict[str, float]:
    """
    Compute mel spectrogram statistics for normalization
    
    Args:
        batch_size: Batch size for data loading
        num_workers: Number of worker processes
        
    Returns:
        Dictionary with mel_mean and mel_std
    """
    print("Computing data statistics...")
    print("This may take several minutes...")
    
    try:
        from hydra import compose, initialize
        from omegaconf import open_dict
        from matcha.data.text_mel_datamodule import TextMelDataModule
        import rootutils
        
        root_path = rootutils.find_root(search_from=__file__, indicator=".project-root")
        
        # Initialize Hydra
        with initialize(version_base="1.3", config_path="configs/data"):
            cfg = compose(config_name="ljspeech.yaml", return_hydra_config=True, overrides=[])
        
        with open_dict(cfg):
            if "hydra" in cfg:
                del cfg["hydra"]
            if "_target_" in cfg:
                del cfg["_target_"]
            cfg["data_statistics"] = None
            cfg["seed"] = 1234
            cfg["batch_size"] = batch_size
            cfg["num_workers"] = num_workers
            cfg["train_filelist_path"] = str(root_path / cfg["train_filelist_path"])
            cfg["valid_filelist_path"] = str(root_path / cfg["valid_filelist_path"])
            cfg["load_durations"] = False
        
        # Create datamodule
        datamodule = TextMelDataModule(**cfg)
        datamodule.setup()
        train_loader = datamodule.train_dataloader()
        
        # Compute statistics
        out_channels = cfg.get("n_fft", 1024) // 2 + 1
        
        total_mel_sum = 0
        total_mel_sq_sum = 0
        total_mel_len = 0
        
        for batch in tqdm(train_loader, desc="Computing statistics"):
            mels = batch["y"]
            mel_lengths = batch["y_lengths"]
            
            total_mel_len += torch.sum(mel_lengths)
            total_mel_sum += torch.sum(mels)
            total_mel_sq_sum += torch.sum(torch.pow(mels, 2))
        
        data_mean = total_mel_sum / (total_mel_len * out_channels)
        data_std = torch.sqrt((total_mel_sq_sum / (total_mel_len * out_channels)) - torch.pow(data_mean, 2))
        
        stats = {"mel_mean": data_mean.item(), "mel_std": data_std.item()}
        
        print(f"✓ Computed statistics:")
        print(f"    mel_mean: {stats['mel_mean']:.6f}")
        print(f"    mel_std:  {stats['mel_std']:.6f}")
        
        return stats
        
    except ImportError as e:
        print(f"Warning: Could not compute statistics: {e}")
        print("You'll need to run 'matcha-data-stats -i ljspeech.yaml' manually later")
        return {}


def update_config_file(config_path: Path, stats: Dict[str, float]):
    """
    Update the config file with computed statistics
    
    Args:
        config_path: Path to the config file
        stats: Dictionary with mel_mean and mel_std
    """
    if not stats or not config_path.exists():
        return
    
    print(f"Updating config file: {config_path}")
    
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    # Find and update the data_statistics section
    updated = False
    in_data_stats = False
    new_lines = []
    
    for line in lines:
        if "data_statistics:" in line:
            in_data_stats = True
            new_lines.append(line)
        elif in_data_stats and "mel_mean:" in line:
            new_lines.append(f"  mel_mean: {stats['mel_mean']:.6f}\n")
            updated = True
        elif in_data_stats and "mel_std:" in line:
            new_lines.append(f"  mel_std: {stats['mel_std']:.6f}\n")
        else:
            new_lines.append(line)
            if in_data_stats and line.strip() and not line.startswith(" "):
                in_data_stats = False
    
    if updated:
        with open(config_path, 'w') as f:
            f.writelines(new_lines)
        print(f"✓ Updated config file with statistics")
    else:
        print("Note: Could not automatically update config file")
        print("Please manually update configs/data/ljspeech.yaml with:")
        print(f"  mel_mean: {stats['mel_mean']:.6f}")
        print(f"  mel_std: {stats['mel_std']:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Automatically download and preprocess LJSpeech dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to download and extract dataset (default: data)"
    )
    parser.add_argument(
        "--filelists-dir",
        type=str,
        default="data/filelists",
        help="Directory to save filelists (default: data/filelists)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default: 42)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for computing statistics (default: 64)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for data loading (default: 4)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step (use existing archive)"
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extraction step (use existing directory)"
    )
    parser.add_argument(
        "--skip-statistics",
        action="store_true",
        help="Skip statistics computation"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download/re-extract/re-generate even if files exist"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    filelists_dir = Path(args.filelists_dir)
    
    print("=" * 70)
    print("LJSpeech Dataset Setup")
    print("=" * 70)
    
    try:
        # Step 1: Download
        if not args.skip_download:
            archive_path = download_ljspeech(data_dir, force=args.force)
        else:
            archive_path = data_dir / LJSPEECH_ARCHIVE
            print(f"Skipping download (using existing: {archive_path})")
        
        print()
        
        # Step 2: Extract
        if not args.skip_extract:
            ljspeech_dir = extract_archive(archive_path, data_dir, force=args.force)
        else:
            ljspeech_dir = data_dir / LJSPEECH_DIR
            print(f"Skipping extraction (using existing: {ljspeech_dir})")
        
        print()
        
        # Step 3: Create filelists
        train_file, val_file = create_filelists(
            ljspeech_dir,
            filelists_dir,
            val_split=args.val_split,
            seed=args.seed,
            force=args.force
        )
        
        print()
        
        # Step 4: Compute statistics
        if not args.skip_statistics:
            stats = compute_statistics(
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
            
            if stats:
                config_path = Path("configs/data/ljspeech.yaml")
                update_config_file(config_path, stats)
        else:
            print("Skipping statistics computation")
            stats = {}
        
        print()
        print("=" * 70)
        print("✓ Setup complete!")
        print("=" * 70)
        print()
        print("Next steps:")
        print("  1. Start training:")
        print("     python matcha/train.py experiment=ljspeech")
        print()
        print("  2. Or for minimum memory usage:")
        print("     python matcha/train.py experiment=ljspeech_min_memory")
        print()
        if not stats:
            print("  3. Don't forget to compute data statistics:")
            print("     matcha-data-stats -i ljspeech.yaml")
            print()
        
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during setup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
