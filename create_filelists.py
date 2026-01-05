"""
Create train and validation filelists for LJSpeech dataset
"""
import os
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Paths
data_dir = Path("data/LJSpeech-1.1")
metadata_file = data_dir / "metadata.csv"
filelists_dir = Path("data/filelists")
train_file = filelists_dir / "ljs_audio_text_train_filelist.txt"
val_file = filelists_dir / "ljs_audio_text_val_filelist.txt"

# Create filelists directory if it doesn't exist
filelists_dir.mkdir(parents=True, exist_ok=True)

# Read metadata and create file entries
entries = []
with open(metadata_file, encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("|")
        if len(parts) >= 2:
            audio_id = parts[0]
            # Use the normalized text (third column if exists, otherwise second column)
            text = parts[2] if len(parts) >= 3 else parts[1]
            
            # Create full path to audio file
            audio_path = str(data_dir / "wavs" / f"{audio_id}.wav")
            
            # Format: path|text
            entries.append(f"{audio_path}|{text}\n")

# Shuffle entries
random.shuffle(entries)

# Split into train (90%) and validation (10%)
split_idx = int(len(entries) * 0.9)
train_entries = entries[:split_idx]
val_entries = entries[split_idx:]

# Write train filelist
with open(train_file, "w", encoding="utf-8") as f:
    f.writelines(train_entries)

# Write validation filelist
with open(val_file, "w", encoding="utf-8") as f:
    f.writelines(val_entries)

print(f"Created filelists:")
print(f"  Train: {train_file} ({len(train_entries)} samples)")
print(f"  Val:   {val_file} ({len(val_entries)} samples)")
