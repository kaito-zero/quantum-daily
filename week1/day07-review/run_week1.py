"""
run_week1.py: 1주 차 코드를 한 번에 실행해서 모두 잘 돌아가는지 확인한다 (Day 7 복습)

실행
    python week1/day07-review/run_week1.py

하는 일
  Day 1~6 코드와 Day 6 테스트를 차례로 실행한다.
  각 파일은 자기 폴더 안에서 실행한다. Day 6의 day06.py가 같은 폴더의 qubit_sim.py를 import하기 때문이다.
  끝나면 통과/실패 표를 찍는다. 하나라도 실패하면 종료 코드 1로 끝난다.
"""

import subprocess
import sys
import time
from pathlib import Path

WEEK1 = Path(__file__).resolve().parent.parent      # .../week1

# (폴더, 실행할 파일) 목록. 순서대로 실행한다.
SCRIPTS = [
    ("day01-first-circuit", "hello.py"),
    ("day02-complex-numbers", "day02.py"),
    ("day03-state-vectors", "day03.py"),
    ("day04-measurement", "day04.py"),
    ("day05-unitary", "day05.py"),
    ("day06-gates-simulator", "day06.py"),
    ("day06-gates-simulator", "test_qubit_sim.py"),
]


def run(folder, name):
    """파일 하나를 실행하고 (성공 여부, 걸린 초, 마지막 출력 한 줄)을 돌려준다."""
    path = WEEK1 / folder / name
    if not path.exists():
        return False, 0.0, "파일이 없다"
    start = time.time()
    result = subprocess.run(
        [sys.executable, name],          # 지금 이 파이썬(가상환경)으로 실행한다
        cwd=path.parent,                 # 그 파일이 있는 폴더에서 실행한다
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    seconds = time.time() - start
    lines = (result.stdout or result.stderr).strip().splitlines()
    last = lines[-1].strip() if lines else ""
    if result.returncode != 0:            # 에러가 났으면 에러의 마지막 줄을 보여 준다
        err = result.stderr.strip().splitlines()
        last = err[-1].strip() if err else last
    return result.returncode == 0, seconds, last


if __name__ == "__main__":
    failed = 0
    print(f"{'파일':<45} {'결과':<6} {'시간':>6}  마지막 출력")
    for folder, name in SCRIPTS:
        ok, seconds, last = run(folder, name)
        failed += not ok
        print(f"{folder + '/' + name:<45} {'통과' if ok else '실패':<6} {seconds:5.1f}s  {last[:60]}")
    print(f"\n{len(SCRIPTS) - failed}/{len(SCRIPTS)} 통과")
    sys.exit(1 if failed else 0)
