import argparse
import pandas as pd
from tqdm import tqdm

import torch
import torchvision
from torch.utils.data import DataLoader

import open_clip

from prompts import build_zeroshot_prompts, TEMPLATES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="e.g. ViT-B-32")
    ap.add_argument("--pretrained", required=True, help="e.g. laion2b_s34b_b79k")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Model + preprocess
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(args.device).eval()

    # Dataset
    ds = torchvision.datasets.Food101(root="data", split=args.split, download=True, transform=preprocess)
    classnames = ds.classes

    # Build template-averaged text embeddings
    all_prompts = build_zeroshot_prompts(classnames)
    text_tokens = tokenizer(all_prompts).to(args.device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Average templates per class: (K*T, D) -> (K, T, D) -> (K, D)
    n_templates = len(TEMPLATES)
    K = len(classnames)
    text_features = text_features.view(K, n_templates, -1).mean(dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    rows = []

    with torch.no_grad():
        for batch_idx, (images, y_gold) in enumerate(tqdm(loader, desc=f"Weak labeling Food101/{args.split}")):
            images = images.to(args.device)
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = 100.0 * image_features @ text_features.T
            y_weak = torch.argmax(logits, dim=1).cpu().tolist()
            y_gold = y_gold.cpu().tolist()

            base = batch_idx * args.batch_size
            for i, (yg, yw) in enumerate(zip(y_gold, y_weak)):
                rows.append(
                    {
                        "index": base + i,
                        "split": args.split,
                        "y_gold": int(yg),
                        "y_weak": int(yw),
                        "model": args.model,
                        "pretrained": args.pretrained,
                    }
                )

    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False)
    print(f"Wrote {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()

