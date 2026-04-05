"""
Run PathoLens on a real WSI file.

Usage:
    python run_real.py path/to/slide.tif
    python run_real.py path/to/slide.svs my_custom_id
"""

import sys
from pathlib import Path

from patholens.config import Config
from patholens.pipeline.inference import PathoLensPipeline

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_real.py <wsi_path> [slide_id]")
        sys.exit(1)

    wsi_path = Path(sys.argv[1])
    slide_id = sys.argv[2] if len(sys.argv) > 2 else wsi_path.stem

    if not wsi_path.exists():
        print(f"Error: file not found: {wsi_path}")
        sys.exit(1)

    print(f"Loading config ...")
    config = Config.load("configs/local-run.yaml")
    print(f"  device : {config.system.device}")
    print(f"  model  : {config.embedding.model_name}")

    print(f"\nRunning pipeline on: {wsi_path.name}")
    pipeline = PathoLensPipeline(config)
    result = pipeline.run(wsi_path=wsi_path, slide_id=slide_id)

    print("\n" + "="*50)
    print(f"  status           : {result.status}")
    print(f"  patches extracted: {result.num_patches}")
    print(f"  evidence coverage: {result.evidence_coverage:.0%}")
    print(f"  elapsed          : {result.elapsed_seconds:.1f}s")
    print(f"  heatmap          : {result.heatmap_path}")
    print(f"  report           : results/{slide_id}/diagnostic_report.json")
    print("="*50)

if __name__ == "__main__":
    main()
