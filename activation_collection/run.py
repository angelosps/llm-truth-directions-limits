import argparse

from main import run
from model_instructions import INSTRUCTIONS

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--instruction", required=True, choices=sorted(INSTRUCTIONS))
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--token_mode", choices=["all", "last"], default="last")
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    run(
        model_name=args.model,
        instruction=args.instruction,
        dataset_path=args.dataset_path,
        save_dir=args.save_dir,
        token_mode=args.token_mode,
        batch_size=args.batch_size,
        seed=args.seed,
        tag=args.tag,
    )
