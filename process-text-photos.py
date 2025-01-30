#!/usr/bin/env python3

from pathlib import Path
import subprocess
import sys
import time

class Timer:
    def __init__(self) -> None:
        self.start = time.time()

    def measure_next(self) -> float:
        t = time.time()
        m = t - self.start
        self.start = t
        return m

def convert(source: Path, destination: Path) -> None:
    print('*' * 80)
    print(f'Converting: {source}')
    t = Timer()
    subprocess.run(['photo-cleanup/target/release/photo-cleanup', str(source), str(destination)])
    print("convert: ", t.measure_next())

def is_image(f: Path) -> bool:
    return f.suffix.lower() in ['.jpg', '.jpeg', '.png']

def convert_all(dir: Path) -> None:
    for f in dir.glob('*'):
        if f.is_dir():
            convert_all(f)
            continue
        if f.is_file() and not f.name.startswith('fix.') and is_image(f):
            fixed_file = f.parent / ('fix.' + f.name)
            if fixed_file.exists():
                continue
            convert(f, fixed_file)

def main(argv: list[str]) -> None:
    assert len(argv) == 1
    convert_all(Path(argv[0]))

if __name__ == '__main__':
    main(sys.argv[1:])
