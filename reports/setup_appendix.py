#!/usr/bin/env python3
"""
Sentinel AI: Setup Thesis Appendix Folders.
Initializes the reports directory structure required for thesis materials.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"

FOLDERS = [
    "thesis_figures",
    "appendix_exports",
    "dashboard_screenshots",
    "api_examples",
    "backend_logs",
]

def main():
    print("[*] Initializing Sentinel AI Thesis Appendix Folders...")
    for folder in FOLDERS:
        folder_path = REPORTS_DIR / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"    - Created/Verified: {folder_path.relative_to(BASE_DIR)}")
    print("[+] Structure initialized successfully.\n")

if __name__ == "__main__":
    main()
