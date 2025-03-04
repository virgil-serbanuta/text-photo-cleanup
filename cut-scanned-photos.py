#!/usr/bin/env python3

from pathlib import Path
import subprocess
import sys

from recursive_photos import process_all, Timer

def run_cutter(source: Path, destination1: Path, destination2: Path) -> None:
    print('*' * 80)
    print(f'Cutting: {source}')
    t = Timer()
    subprocess.run(['photo-cut/target/release/photo-cut', str(source), str(destination1), str(destination2)])
    print("convert: ", t.measure_next())

def cut(source:Path):
    cut_file_left = source.parent / ('cut.1.' + source.name)
    cut_file_right = source.parent / ('cut.2.' + source.name)
    run_cutter(source, cut_file_left, cut_file_right)


def main(argv: list[str]) -> None:
    assert len(argv) == 1
    process_all(Path(argv[0]), cut)

if __name__ == '__main__':
    main(sys.argv[1:])
