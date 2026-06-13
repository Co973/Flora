import argparse
import json
import sys
from pathlib import Path

import timm
import torch
from PIL import Image

torch.serialization.add_safe_globals([argparse.Namespace])


MODEL_ROOT = Path(__file__).parent / "pretrained_models"
MODEL_DIRECTORY = (
    MODEL_ROOT
    / "vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all"
)
CHECKPOINT_PATH = MODEL_DIRECTORY / "model_best.pth.tar"
CLASS_MAPPING_PATH = MODEL_ROOT / "class_mapping.txt"
SPECIES_MAPPING_PATH = MODEL_ROOT / "species_id_to_name.json"


def load_class_ids():
    return [
        line.strip()
        for line in CLASS_MAPPING_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_species_mapping():
    return json.loads(SPECIES_MAPPING_PATH.read_text(encoding="utf-8"))


def load_model(class_count):
    model = timm.create_model(
        "vit_base_patch14_reg4_dinov2.lvd142m",
        pretrained=False,
        num_classes=class_count,
        checkpoint_path=str(CHECKPOINT_PATH),
    )
    model.eval()
    return model


def identify(image_path):
    class_ids = load_class_ids()
    species_mapping = load_species_mapping()
    model = load_model(len(class_ids))
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    with Image.open(image_path) as image:
        image_tensor = transform(image.convert("RGB")).unsqueeze(0)

    with torch.inference_mode():
        probabilities = model(image_tensor).softmax(dim=1)[0]
        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=min(25, len(class_ids)),
        )

    candidates = []
    for probability, class_index in zip(
        top_probabilities.tolist(),
        top_indices.tolist(),
    ):
        species_id = class_ids[class_index]
        species = species_mapping.get(species_id)
        if not species:
            continue
        candidates.append(
            {
                "scientificName": species["scientificName"],
                "commonNames": species.get("commonNames", []),
                "family": species.get("family", "Unknown"),
                "confidence": probability,
                "plantnetSpeciesId": species_id,
            }
        )
        if len(candidates) == 3:
            break

    if not candidates:
        raise RuntimeError(
            "The model's top predictions use retired species IDs with no current name mapping."
        )
    return candidates


def main():
    parser = argparse.ArgumentParser(description="PlantCLEF local inference")
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--organ",
        default="auto",
        choices=["auto", "leaf", "flower", "fruit", "bark"],
    )
    args = parser.parse_args()

    if not Path(args.image).is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    required_files = [CHECKPOINT_PATH, CLASS_MAPPING_PATH, SPECIES_MAPPING_PATH]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing local model files: {', '.join(missing)}")

    print(
        json.dumps(
            {
                "isPlant": True,
                "organ": args.organ,
                "candidates": identify(args.image),
            }
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
