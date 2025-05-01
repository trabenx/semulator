#!/usr/bin/env python
import sys
import argparse
from pathlib import Path

# Add src directory to Python path if needed (e.g., when running main.py directly)
project_root = Path(__file__).resolve().parent
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
interfaces_path = project_root / 'interfaces'
if str(interfaces_path) not in sys.path:
     sys.path.insert(0, str(interfaces_path)) # Ensure interfaces package is importable

# Choose interface (CLI or WebUI)
# For now, default to CLI
from interfaces.cli import run_cli

if __name__ == "__main__":
    # Top-level parser to choose the interface mode
    parser = argparse.ArgumentParser(description="Synthetic SEM Image Generator Entry Point")
    parser.add_argument(
        'mode',
        nargs='?', # Makes the mode argument optional
        default='cli', # Default to CLI if no mode is specified
        choices=['cli', 'webui'],
        help="Mode to run the application: 'cli' (default) or 'webui'."
    )
    # Parse only the mode argument first, ignore others for now
    args, remaining_argv = parser.parse_known_args()

    # Pass remaining arguments to the specific interface runner
    sys.argv = [sys.argv[0]] + remaining_argv # Update sys.argv for the sub-parser

    if args.mode == 'cli':
        try:
            from interfaces.cli import run_cli
            run_cli() # run_cli will parse the remaining_argv using its own parser
        except ImportError:
            print("Error: Could not import CLI interface. Ensure 'interfaces/cli.py' exists.")
            sys.exit(1)
        except Exception as e:
            print(f"An error occurred running the CLI: {e}")
            sys.exit(1)

    elif args.mode == 'webui':
        try:
            # Ensure webui dependencies like Flask are installed if this mode is chosen
            from interfaces.webui.app import run_webui
            print("Starting Web UI...")
            run_webui() # run_webui doesn't typically parse CLI args in the same way
        except ImportError as e:
            print(f"Error: Could not import Web UI interface: {e}")
            print("Please ensure Flask is installed (`pip install Flask`) and 'interfaces/webui/app.py' exists.")
            sys.exit(1)
        except Exception as e:
             print(f"An error occurred running the WebUI: {e}")
             sys.exit(1)

    else:
        # Should not happen due to 'choices' in argparse, but good practice
        print(f"Error: Unknown mode '{args.mode}'. Use 'cli' or 'webui'.")
        parser.print_help()
        sys.exit(1)

