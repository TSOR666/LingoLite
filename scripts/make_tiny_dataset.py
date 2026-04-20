"""Generate a tiny example dataset for quickstart training."""

import json
from pathlib import Path


def main() -> None:
    data = [
        {"src_text": "Hello", "tgt_text": "Hola", "src_lang": "en", "tgt_lang": "es"},
        {"src_text": "Good morning", "tgt_text": "Buenos dias", "src_lang": "en", "tgt_lang": "es"},
        {"src_text": "How are you?", "tgt_text": "Como estas?", "src_lang": "en", "tgt_lang": "es"},
        {"src_text": "Thank you", "tgt_text": "Gracias", "src_lang": "en", "tgt_lang": "es"},
        {"src_text": "Please", "tgt_text": "Por favor", "src_lang": "en", "tgt_lang": "es"},
        {"src_text": "Goodbye", "tgt_text": "Adios", "src_lang": "en", "tgt_lang": "es"},
        {"src_text": "Good night", "tgt_text": "Buenas noches", "src_lang": "en", "tgt_lang": "es"},
        {"src_text": "See you soon", "tgt_text": "Hasta pronto", "src_lang": "en", "tgt_lang": "es"},
    ]
    out = Path("examples/data/tiny_dataset.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote tiny dataset to {out}")


if __name__ == "__main__":
    main()
