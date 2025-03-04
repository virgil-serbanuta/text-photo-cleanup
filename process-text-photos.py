#!/usr/bin/env python3

from pathlib import Path
import subprocess
import sys

from recursive_photos import process_all, Timer

def run_convert(source: Path, destination: Path) -> None:
    print('*' * 80)
    print(f'Converting: {source}')
    t = Timer()
    subprocess.run(['photo-cleanup/target/release/photo-cleanup', str(source), str(destination)])
    print("convert: ", t.measure_next())

def convert(source:Path) -> None:
    fixed_file = source.parent / ('fix.' + source.name)
    if fixed_file.exists():
        return
    run_convert(source, fixed_file)

def main(argv: list[str]) -> None:
    assert len(argv) == 1
    process_all(Path(argv[0]), convert)

if __name__ == '__main__':
    main(sys.argv[1:])
