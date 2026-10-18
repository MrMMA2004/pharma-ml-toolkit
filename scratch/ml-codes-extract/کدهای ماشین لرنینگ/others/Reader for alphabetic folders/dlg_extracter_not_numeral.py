#!/usr/bin/env python3
"""
extract_rdlg.py

Usage:
    python extract_rdlg.py                # scans current working directory
    python extract_rdlg.py /path/to/parent_folder
    python extract_rdlg.py /path/to/parent_folder -o /path/to/output_folder --dry-run
"""
import os
import shutil
import argparse
import re
import sys
import traceback

def safe_name(name: str) -> str:
    # Replace characters that are unsafe in filenames with underscore.
    # Allow letters, numbers, space, dot, underscore, hyphen. Collapse runs of underscores.
    s = re.sub(r'[^A-Za-z0-9 ._\-]+', '_', name)
    s = re.sub(r'[_ ]+', '_', s).strip('_')
    return s or "folder"

def find_and_copy(base_dir: str, output_folder: str, dry_run: bool = False) -> (int, int):
    found = 0
    copied = 0
    base_dir = os.path.abspath(base_dir)
    output_folder = os.path.abspath(output_folder)

    print(f"Scanning: {base_dir}")
    print(f"Output folder: {output_folder}")
    print(f"Dry run: {dry_run}")
    print("-" * 60)

    for root, dirs, files in os.walk(base_dir):
        # skip the output folder (in case it's inside base_dir)
        if os.path.abspath(root) == output_folder:
            continue

        for fname in files:
            # case-insensitive match for r.dlg
            if fname.lower().endswith('.dlg'):
                found += 1
                source_path = os.path.join(root, fname)
                folder_name = os.path.basename(root) or "root"
                safe = safe_name(folder_name)
                new_filename = f"{safe}.dlg"
                dest_path = os.path.join(output_folder, new_filename)

                # avoid overwriting: add numeric suffix if necessary
                counter = 1
                final_dest = dest_path
                while os.path.exists(final_dest):
                    final_dest = os.path.join(output_folder, f"{safe}_{counter}.dlg")
                    counter += 1

                print(f"Found: {source_path}")
                print(f" => Will copy to: {final_dest}")

                if not dry_run:
                    try:
                        shutil.copy2(source_path, final_dest)
                        copied += 1
                    except Exception as e:
                        print(f"ERROR copying '{source_path}' -> '{final_dest}': {e}")
                        traceback.print_exc()
                print("-" * 40)

    print(f"SUMMARY: Found {found} r.dlg files. Copied {copied} files.")
    return found, copied

def main():
    p = argparse.ArgumentParser(description="Collect r.dlg files from subfolders and copy them to one folder, renamed by folder name.")
    p.add_argument('base_dir', nargs='?', default=os.getcwd(), help="Directory to scan (default: current working directory).")
    p.add_argument('-o', '--output', default=None, help="Output folder (default: <base_dir>/extracted_dialogs).")
    p.add_argument('--dry-run', action='store_true', help="Don't copy, just show what would be done.")
    args = p.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    output_folder = args.output or os.path.join(base_dir, 'extracted_dialogs')
    try:
        os.makedirs(output_folder, exist_ok=True)
    except Exception as e:
        print(f"Could not create output folder '{output_folder}': {e}")
        sys.exit(1)

    find_and_copy(base_dir, output_folder, dry_run=args.dry_run)

if __name__ == '__main__':
    main()
