#!/usr/bin/env python
"""
DFVEdit - Entry point script for video editing.

Usage:
    python scripts/run_edit.py --config dfvedit/configs/examples/bear_shape.yaml
    python scripts/run_edit.py --config dfvedit/configs/examples/bear_shape.yaml --model wanx
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dfvedit import ConfigLoader, DFVEditConfig
from dfvedit.core.runner import run_edit
from dfvedit.utils.logging import setup_logging, get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DFVEdit: Conditional Delta Flow Vector for Zero-shot Video Editing"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file (YAML)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name override (wanx, cogvideox)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    parser.add_argument(
        "--debug-tokens",
        action="store_true",
        help="Print token debug information",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = 10 if args.verbose else 20  # DEBUG=10, INFO=20
    setup_logging(level=log_level)
    logger = get_logger()

    # Load configuration
    logger.info(f"Loading config from {args.config}")
    cfg = ConfigLoader.load(args.config, model_name=args.model)

    # Apply CLI overrides
    if args.debug:
        cfg.debug = True
    if args.debug_tokens:
        cfg.debug_tokens = True

    # Store config path for copying to output
    cfg._config_path = args.config

    # Run editing
    elapsed = run_edit(cfg)

    print(f"\n{'='*60}")
    print(f"Editing complete!")
    print(f"Output: {cfg.output}")
    print(f"Time: {elapsed:.2f} seconds")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
