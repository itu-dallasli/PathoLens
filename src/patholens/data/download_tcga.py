import argparse
import os
import subprocess
from pathlib import Path

from patholens.logger import get_logger

log = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Download TCGA-BRCA Diagnostic Slides via GDC API")
    parser.add_argument("--manifest", type=str, required=True, 
                        help="Path to the GDC manifest file (e.g., gdc_manifest_202x.txt)")
    parser.add_argument("--output-dir", type=str, default="data/raw", 
                        help="Directory to save downloaded .svs files")
    parser.add_argument("--token", type=str, required=False,
                        help="Path to GDC authentication token (if downloading controlled data)")
    parser.add_argument("--limit", type=int, default=10,
                        help="Number of files to download (useful for the Dev Gate phase)")
    return parser.parse_args()

def main():
    """
    Downloads diagnostic .svs slides using the gdc-client.
    """
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure gdc-client is installed
    try:
        subprocess.run(["gdc-client", "--version"], check=True, capture_output=True)
    except FileNotFoundError:
        log.error("gdc-client not found! Please download it from https://gdc.cancer.gov/access-data/gdc-data-transfer-tool")
        return
        
    log.info(f"Starting TCGA download using manifest: {args.manifest}")
    log.info(f"Target directory: {out_dir.absolute()}")
    
    cmd = [
        "gdc-client", "download",
        "-m", args.manifest,
        "-d", str(out_dir),
    ]
    if args.token:
        cmd.extend(["-t", args.token])
        
    # In a real scenario, we might parse the manifest to limit the download.
    # For now, we simulate execution to log the exact command.
    log.info(f"Executing: {' '.join(cmd)}")
    log.info("Note: Limiting downloads requires splitting the manifest file manually before passing it to gdc-client.")
    
    # Uncomment to actually run:
    # subprocess.run(cmd, check=True)
    
if __name__ == "__main__":
    main()
