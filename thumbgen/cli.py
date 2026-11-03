"""Command-line entry point: `thumbgen ingest|serve|prompt`."""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="thumbgen")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="Build a kohya-ready dataset from CSV/JSON + images")
    p_ing.add_argument("--config", default="configs/dataset.yaml")
    p_ing.add_argument("--limit", type=int, default=None, help="Process only the first N rows")
    p_ing.add_argument("--no-caption", action="store_true", help="Skip WD14/Florence autocaptioning")
    p_ing.add_argument("--no-style", action="store_true", help="Skip CLIP style classification")
    p_ing.add_argument("--device", default=None, help="cuda | cpu (auto)")

    p_srv = sub.add_parser("serve", help="Run the generation web app")
    p_srv.add_argument("--host", default="0.0.0.0")
    p_srv.add_argument("--port", type=int, default=8000)
    p_srv.add_argument("--presets", default="configs/presets.yaml")

    p_pr = sub.add_parser("prompt", help="Print the expanded prompt for a preset (no GPU needed)")
    p_pr.add_argument("preset")
    p_pr.add_argument("--style", default="3d", choices=["3d", "anime"])
    p_pr.add_argument("--palette", default="orange_teal")
    p_pr.add_argument("--presets", default="configs/presets.yaml")
    p_pr.add_argument("--var", action="append", default=[], help="key=value slot override")

    args = parser.parse_args(argv)

    if args.cmd == "ingest":
        from thumbgen.data.ingest import run_ingest
        return run_ingest(args.config, limit=args.limit, caption=not args.no_caption,
                          classify_style=not args.no_style, device=args.device)
    if args.cmd == "serve":
        import uvicorn
        from thumbgen.app.server import create_app
        uvicorn.run(create_app(presets_path=args.presets), host=args.host, port=args.port)
        return 0
    if args.cmd == "prompt":
        from thumbgen.app.prompts import PromptBuilder
        pb = PromptBuilder.from_yaml(args.presets)
        slots = dict(kv.split("=", 1) for kv in args.var)
        pos, neg, meta = pb.build(args.preset, style=args.style, palette=args.palette, slots=slots)
        print("POSITIVE:\n" + pos + "\n\nNEGATIVE:\n" + neg + f"\n\nmeta: {meta}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
