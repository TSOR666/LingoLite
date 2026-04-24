"""Generate tiny example datasets for the quickstart/smoke pipeline.

Writes two files under ``examples/data/``:

- ``tiny_dataset.json``: ~40 bidirectional translation pairs across the six
  supported languages. Designed to be small enough that a tiny model can
  overfit in <2 minutes on CPU while still exercising multiple language pairs
  and directions.
- ``tiny_dataset_val.json``: a 4-example held-out slice drawn from the same
  distribution, used as the validation split for the smoke pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


TRAIN_PAIRS: List[Dict[str, str]] = [
    # English -> Spanish
    {"src_text": "Hello", "tgt_text": "Hola", "src_lang": "en", "tgt_lang": "es"},
    {"src_text": "Good morning", "tgt_text": "Buenos dias", "src_lang": "en", "tgt_lang": "es"},
    {"src_text": "How are you?", "tgt_text": "Como estas?", "src_lang": "en", "tgt_lang": "es"},
    {"src_text": "Thank you", "tgt_text": "Gracias", "src_lang": "en", "tgt_lang": "es"},
    {"src_text": "Please", "tgt_text": "Por favor", "src_lang": "en", "tgt_lang": "es"},
    {"src_text": "Goodbye", "tgt_text": "Adios", "src_lang": "en", "tgt_lang": "es"},
    {"src_text": "Good night", "tgt_text": "Buenas noches", "src_lang": "en", "tgt_lang": "es"},
    {"src_text": "See you soon", "tgt_text": "Hasta pronto", "src_lang": "en", "tgt_lang": "es"},
    {"src_text": "I love you", "tgt_text": "Te quiero", "src_lang": "en", "tgt_lang": "es"},
    {"src_text": "Yes", "tgt_text": "Si", "src_lang": "en", "tgt_lang": "es"},
    {"src_text": "No", "tgt_text": "No", "src_lang": "en", "tgt_lang": "es"},
    {"src_text": "What is your name?", "tgt_text": "Como te llamas?", "src_lang": "en", "tgt_lang": "es"},

    # Spanish -> English (reverse direction for a handful of the pairs above)
    {"src_text": "Hola", "tgt_text": "Hello", "src_lang": "es", "tgt_lang": "en"},
    {"src_text": "Buenos dias", "tgt_text": "Good morning", "src_lang": "es", "tgt_lang": "en"},
    {"src_text": "Gracias", "tgt_text": "Thank you", "src_lang": "es", "tgt_lang": "en"},
    {"src_text": "Adios", "tgt_text": "Goodbye", "src_lang": "es", "tgt_lang": "en"},

    # English -> French
    {"src_text": "Hello", "tgt_text": "Bonjour", "src_lang": "en", "tgt_lang": "fr"},
    {"src_text": "Good morning", "tgt_text": "Bonjour", "src_lang": "en", "tgt_lang": "fr"},
    {"src_text": "Thank you", "tgt_text": "Merci", "src_lang": "en", "tgt_lang": "fr"},
    {"src_text": "Goodbye", "tgt_text": "Au revoir", "src_lang": "en", "tgt_lang": "fr"},
    {"src_text": "Please", "tgt_text": "Sil vous plait", "src_lang": "en", "tgt_lang": "fr"},
    {"src_text": "Yes", "tgt_text": "Oui", "src_lang": "en", "tgt_lang": "fr"},
    {"src_text": "No", "tgt_text": "Non", "src_lang": "en", "tgt_lang": "fr"},

    # English -> German
    {"src_text": "Hello", "tgt_text": "Hallo", "src_lang": "en", "tgt_lang": "de"},
    {"src_text": "Thank you", "tgt_text": "Danke", "src_lang": "en", "tgt_lang": "de"},
    {"src_text": "Good morning", "tgt_text": "Guten Morgen", "src_lang": "en", "tgt_lang": "de"},
    {"src_text": "Goodbye", "tgt_text": "Auf Wiedersehen", "src_lang": "en", "tgt_lang": "de"},
    {"src_text": "Yes", "tgt_text": "Ja", "src_lang": "en", "tgt_lang": "de"},
    {"src_text": "No", "tgt_text": "Nein", "src_lang": "en", "tgt_lang": "de"},

    # English -> Italian
    {"src_text": "Hello", "tgt_text": "Ciao", "src_lang": "en", "tgt_lang": "it"},
    {"src_text": "Thank you", "tgt_text": "Grazie", "src_lang": "en", "tgt_lang": "it"},
    {"src_text": "Goodbye", "tgt_text": "Arrivederci", "src_lang": "en", "tgt_lang": "it"},
    {"src_text": "Yes", "tgt_text": "Si", "src_lang": "en", "tgt_lang": "it"},
    {"src_text": "No", "tgt_text": "No", "src_lang": "en", "tgt_lang": "it"},

    # English -> Danish
    {"src_text": "Hello", "tgt_text": "Hej", "src_lang": "en", "tgt_lang": "da"},
    {"src_text": "Thank you", "tgt_text": "Tak", "src_lang": "en", "tgt_lang": "da"},
    {"src_text": "Goodbye", "tgt_text": "Farvel", "src_lang": "en", "tgt_lang": "da"},
    {"src_text": "Yes", "tgt_text": "Ja", "src_lang": "en", "tgt_lang": "da"},
    {"src_text": "No", "tgt_text": "Nej", "src_lang": "en", "tgt_lang": "da"},
]

VAL_PAIRS: List[Dict[str, str]] = [
    {"src_text": "Hello", "tgt_text": "Hola", "src_lang": "en", "tgt_lang": "es"},
    {"src_text": "Thank you", "tgt_text": "Gracias", "src_lang": "en", "tgt_lang": "es"},
    {"src_text": "Goodbye", "tgt_text": "Au revoir", "src_lang": "en", "tgt_lang": "fr"},
    {"src_text": "Hello", "tgt_text": "Hallo", "src_lang": "en", "tgt_lang": "de"},
]


def main() -> None:
    out_dir = Path("examples/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "tiny_dataset.json"
    val_path = out_dir / "tiny_dataset_val.json"

    train_path.write_text(json.dumps(TRAIN_PAIRS, indent=2), encoding="utf-8")
    val_path.write_text(json.dumps(VAL_PAIRS, indent=2), encoding="utf-8")

    print(f"Wrote train dataset to {train_path} ({len(TRAIN_PAIRS)} pairs)")
    print(f"Wrote val   dataset to {val_path} ({len(VAL_PAIRS)} pairs)")


if __name__ == "__main__":
    main()
