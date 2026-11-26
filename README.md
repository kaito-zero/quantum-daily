# quantum-daily

웹 개발자가 매일 양자컴퓨팅 개념 하나를 공부하고, 직접 돌려 본 코드와 노트를 남기는 저장소입니다.
같은 내용을 게임으로 만든 [Qubit Quest](https://kaito-zero.github.io/qubit-quest/)도 함께 만들고 있습니다.

## 실행 환경

- Python 3.11 이상
- Qiskit 2.x, qiskit-aer, NumPy

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   /  macOS·Linux: source .venv/bin/activate
pip install qiskit qiskit-aer numpy
```

## 1주 차: 큐비트 하나

| 날짜 | 주제 | 코드 | 한 줄 요약 |
| --- | --- | --- | --- |
| Day 1 (11/16) | 큐비트와 측정 | [hello.py](week1/day01-first-circuit/hello.py) | H를 걸고 1,000번 측정하면 0과 1이 반반 나온다 |
| Day 2 (11/17) | 복소수: 크기와 위상 | [day02.py](week1/day02-complex-numbers/day02.py) | e^(iθ)를 곱하면 크기는 그대로, 각도만 돈다 |
| Day 3 (11/18) | 상태 벡터와 디랙 표기법 | [day03.py](week1/day03-state-vectors/day03.py) | 상태는 진폭 두 개의 벡터이고, 크기 제곱의 합은 1이다 |
| Day 4 (11/19) | 측정과 보른 규칙 | [day04.py](week1/day04-measurement/day04.py) | 확률은 진폭 크기의 제곱이고, 측정하면 상태가 바뀐다 |
| Day 5 (11/20) | 행렬과 유니터리 | [day05.py](week1/day05-unitary/day05.py) | 게이트는 U†U = I인 행렬이다 |
| Day 6 (11/21) | 단일 큐비트 게이트 정리 | [day06.py](week1/day06-gates-simulator/day06.py), [qubit_sim.py](week1/day06-gates-simulator/qubit_sim.py) | X, Y, Z, H, S, T와 NumPy 1큐비트 시뮬레이터 |
| Day 7 (11/22) | 1주 복습 | [run_week1.py](week1/day07-review/run_week1.py) | 이번 주 코드를 한 번에 돌려 보고 백지에 설명해 보기 |

## 2주 차: 여러 큐비트와 얽힘

| 날짜 | 주제 | 코드 | 한 줄 요약 |
| --- | --- | --- | --- |
| Day 8 (11/23) | 텐서곱과 두 큐비트 | [day08.py](week2/day08-tensor-product/day08.py) | 두 큐비트는 진폭 4개. Qiskit은 오른쪽 글자가 0번 큐비트다 |
| Day 9 (11/24) | 2큐비트 게이트: CNOT, CZ, SWAP | [day09.py](week2/day09-two-qubit-gates/day09.py) | 제어가 1일 때만 타깃에 게이트. SWAP은 CNOT 세 개 |

각 폴더의 `notes.md`에 개념 정리, 질문과 답, 실행 결과가 있습니다.

## 이번 주 코드 한 번에 확인하기

```bash
python week1/day07-review/run_week1.py
```

Day 1~6 코드와 Day 6 테스트를 차례로 실행하고, 어느 것이 실패했는지 알려 줍니다.

## 이 저장소에 대해

- 코드와 노트는 AI(Claude)와 함께 작성했습니다. 개념은 직접 공부하고, 코드는 모두 실제로 실행해 확인했습니다.
- 참고: [IBM Quantum Learning: Basics of quantum information](https://quantum.cloud.ibm.com/learning/courses/basics-of-quantum-information)
