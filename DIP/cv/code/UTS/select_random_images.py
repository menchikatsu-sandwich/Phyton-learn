
from pathlib import Path
import argparse
import random
import shutil
import subprocess
import sys

IMAGE_EXTS = {"jpg","jpeg","png","bmp","tif","tiff","webp","heic"}


def find_images(src: Path, recursive: bool=False):
    if recursive:
        files = [p for p in src.rglob("*") if p.is_file() and p.suffix.lower().lstrip('.') in IMAGE_EXTS]
    else:
        files = [p for p in src.glob("*") if p.is_file() and p.suffix.lower().lstrip('.') in IMAGE_EXTS]
    return files


def copy_files(selected, dest: Path, overwrite: bool=False):
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for p in selected:
        target = dest / p.name
        if target.exists() and not overwrite:
            # if exists, append counter to avoid overwrite
            stem = p.stem
            suf = p.suffix
            i = 1
            while True:
                newname = f"{stem}_{i}{suf}"
                target = dest / newname
                if not target.exists():
                    break
                i += 1
        shutil.copy2(p, target)
        copied.append(target.resolve())
    return copied


def copy_to_clipboard(text: str):
    # Works on Windows (uses clip)
    try:
        p = subprocess.run(['clip'], input=text.encode('utf-8'), check=True)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description='Pilih N gambar acak dari folder dan salin ke folder tujuan')
    ap.add_argument('src', help='Folder sumber (absolute path)')
    ap.add_argument('-n', '--number', type=int, default=10, help='Jumlah gambar yang dipilih (default 10)')
    ap.add_argument('-d', '--dest', help='Folder tujuan (absolute path). Jika tidak diberikan akan dibuat di <src>_rboflow_sample')
    ap.add_argument('--recursive', action='store_true', help='Cari juga di subfolder')
    ap.add_argument('--no-copy', action='store_true', help='Jangan salin file, hanya cetak path yang dipilih')
    ap.add_argument('--clipboard', action='store_true', help='Salin daftar path hasil ke clipboard (Windows: `clip`)')
    ap.add_argument('--overwrite', action='store_true', help='Izinkan overwrite file tujuan jika nama sama')
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists() or not src.is_dir():
        print(f"Error: sumber '{src}' tidak ditemukan atau bukan folder", file=sys.stderr)
        sys.exit(1)

    files = find_images(src, recursive=args.recursive)
    if not files:
        print(f"Tidak ada file gambar di {src}")
        sys.exit(1)

    total = len(files)
    k = min(args.number, total)
    selected = random.sample(files, k) if total >= k else files[:]

    # default dest
    if args.dest:
        dest = Path(args.dest)
    else:
        dest = src.parent / f"{src.name}_rboflow_sample"

    if args.no_copy:
        out_paths = [str(p.resolve()) for p in selected]
        print("Selected files (not copied):")
        for p in out_paths:
            print(p)
        if args.clipboard:
            joined = "\r\n".join(out_paths)
            ok = copy_to_clipboard(joined)
            print("Paths copied to clipboard" if ok else "Gagal menyalin ke clipboard")
        return

    copied = copy_files(selected, dest, overwrite=args.overwrite)
    print(f"Copied {len(copied)} files to: {dest.resolve()}")
    for p in copied:
        print(p)

    if args.clipboard:
        joined = "\r\n".join(str(p) for p in copied)
        ok = copy_to_clipboard(joined)
        print("Copied destination paths to clipboard" if ok else "Gagal menyalin ke clipboard")


if __name__ == '__main__':
    main()
