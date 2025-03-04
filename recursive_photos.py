import time
from collections.abc import Callable
from pathlib import Path

class Timer:
    def __init__(self) -> None:
        self.start = time.time()

    def measure_next(self) -> float:
        t = time.time()
        m = t - self.start
        self.start = t
        return m

def is_image(f: Path) -> bool:
    return f.suffix.lower() in ['.jpg', '.jpeg', '.png']

def process_all(dir: Path, process:Callable[[Path], None]) -> None:
    for f in dir.glob('*'):
        if f.is_dir():
            process_all(f, process)
            continue
        if f.is_file() and not f.name.startswith('fix.') and is_image(f):
            process(f)
