"""
Day 6: 단일 큐비트 게이트 정리 (X, Y, Z, H, S, T) + NumPy 1큐비트 시뮬레이터

목표
  큐비트 하나에 거는 게이트 여섯 개를 한 번에 정리하고,
  NumPy만으로 만든 시뮬레이터(qubit_sim.py)가 Qiskit과 같은 결과를 내는지 확인한다.

흐름
  1. 여섯 게이트의 행렬을 본다. 새로 나온 것은 Y와 T다.
  2. 각 게이트가 |0>, |1>, |+>를 어떻게 바꾸는지 표로 본다.
  3. 게이트 사이의 관계: 두 번 걸면 원래대로(X, Y, Z, H), T를 두 번 = S, S를 두 번 = Z, HZH = X.
  4. 여섯 게이트 모두 유니터리다 (Day 5의 U†U = I).
  5. Z, S, T는 둘째 진폭의 각도만 돌린다. 바로 측정하면 차이가 안 보이지만, H 사이에 넣으면 확률로 드러난다.
  6. 직접 만든 시뮬레이터로 같은 실험을 하고, Qiskit 결과와 비교한다.

같이 있는 파일
  qubit_sim.py        시뮬레이터 본체
  test_qubit_sim.py   시뮬레이터가 Qiskit과 같은지 확인하는 테스트 (python test_qubit_sim.py 로 실행)

예상 결과
  5번과 6번의 측정 횟수는 실행할 때마다 조금씩 달라진다. 나머지는 항상 같다.
"""

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from qiskit_aer import AerSimulator

from qubit_sim import GATES, OneQubitSim


def num(x):
    """숫자 하나를 0.707, -1, 0.5+0.5j처럼 짧게 적는다. 소수 셋째 자리까지만."""
    x = complex(x)
    re = round(x.real, 3) + 0.0   # + 0.0 은 -0.0 을 0.0 으로 바꾼다
    im = round(x.imag, 3) + 0.0
    if im == 0:
        return f"{re:g}"
    if re == 0:
        return f"{im:g}j"
    return f"{re:g}{im:+g}j"


def show(a):
    """벡터는 [1, 0], 행렬은 [[0, 1], [1, 0]]처럼 적어 주는 도우미 (Day 5와 같다)."""
    a = np.asarray(a)
    if a.ndim == 1:
        return "[" + ", ".join(num(x) for x in a) + "]"
    return "[" + ", ".join(show(row) for row in a) + "]"


def same(a, b):
    """두 배열이 (아주 작은 계산 오차를 빼고) 같은가."""
    return bool(np.allclose(a, b))


X, Y, Z, H, S, T = (GATES[k] for k in "xyzhst")
I = np.eye(2)

# ---------------------------------------------------------------
# 1. 여섯 게이트의 행렬
# ---------------------------------------------------------------
print("1. 여섯 게이트")
for name in "XYZHST":
    print(f"   {name} = {show(GATES[name.lower()])}")
# X = [[0, 1], [1, 0]]                Y = [[0, -1j], [1j, 0]]
# Z = [[1, 0], [0, -1]]               H = [[0.707, 0.707], [0.707, -0.707]]
# S = [[1, 0], [0, 1j]]               T = [[1, 0], [0, 0.707+0.707j]]
# T의 0.707+0.707j는 크기가 1이고 각도가 45도인 복소수 e^(iπ/4)다 (Day 2).

# ---------------------------------------------------------------
# 2. 각 게이트가 상태를 어떻게 바꾸나
# ---------------------------------------------------------------
ket0 = np.array([1, 0])
ket1 = np.array([0, 1])
plus = np.array([1, 1]) / np.sqrt(2)
print("2. 게이트 |0>에  |1>에  |+>에")
for name in "XYZHST":
    U = GATES[name.lower()]
    print(f"   {name}: {show(U @ ket0):<16} {show(U @ ket1):<18} {show(U @ plus)}")
# X: [0, 1]  [1, 0]  [0.707, 0.707]          0과 1이 바뀐다. |+>는 그대로
# Y: [0, 1j] [-1j, 0] [-0.707j, 0.707j]      바뀌면서 j가 붙는다
# Z: [1, 0]  [0, -1]  [0.707, -0.707]        |1> 쪽 부호만 바뀐다. |+>가 |->가 된다
# S, T도 |0>은 그대로 두고 |1> 쪽에만 j, 0.707+0.707j를 곱한다

# ---------------------------------------------------------------
# 3. 게이트 사이의 관계
# ---------------------------------------------------------------
print("3. X·X = I:", same(X @ X, I), "/ Y·Y = I:", same(Y @ Y, I),
      "/ Z·Z = I:", same(Z @ Z, I), "/ H·H = I:", same(H @ H, I))        # 모두 True. 두 번 걸면 원래대로
print("   T·T = S:", same(T @ T, S), "/ S·S = Z:", same(S @ S, Z))       # True, True
print("   H·Z·H = X:", same(H @ Z @ H, X), "/ H·X·H = Z:", same(H @ X @ H, Z))  # True, True
print("   Y = j·X·Z:", same(Y, 1j * X @ Z))                                # True. Y는 Z를 건 뒤 X를 걸고 j를 곱한 것
print("   T를 8번 = I:", same(np.linalg.matrix_power(T, 8), I))            # True. 45도 x 8 = 360도
# 각도로 보면 T = 45도, S = 90도, Z = 180도다. 그래서 T 두 번 = S, S 두 번 = Z.

# ---------------------------------------------------------------
# 4. 모두 유니터리인가
# ---------------------------------------------------------------
for name in ["x", "y", "z", "h", "s", "t", "sdg", "tdg"]:
    U = GATES[name]
    print(f"4. {name:>3}: U†U = I 인가 {same(U.conj().T @ U, I)}")        # 모두 True

# ---------------------------------------------------------------
# 5. 위상 게이트(Z, S, T)는 H 사이에 넣어야 차이가 보인다
# ---------------------------------------------------------------
sim = AerSimulator()
for name, angle in [("z", 180), ("s", 90), ("t", 45)]:
    direct = Statevector.from_label("+").evolve(Operator(GATES[name])).probabilities()   # |+>에 걸고 바로 측정
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    getattr(qc, name)(0)               # qc.z(0), qc.s(0), qc.t(0) 중 하나
    qc.h(0)
    qc.measure(0, 0)
    counts = sim.run(qc, shots=1000).result().get_counts()
    expect = np.cos(np.radians(angle) / 2) ** 2           # Day 2의 공식: 0이 나올 확률 = cos²(θ/2)
    print(f"5. {name.upper()}({angle}도) 바로 측정: {show(direct)} / H 사이: {counts} (0이 {expect:.1%} 근처)")
# Z: 바로 측정 [0.5, 0.5] / H 사이 {'1': 1000}          (0이 0.0%)
# S: 바로 측정 [0.5, 0.5] / H 사이 0과 1이 반반         (0이 50.0%)
# T: 바로 측정 [0.5, 0.5] / H 사이 0이 850번쯤          (0이 85.4%)

# ---------------------------------------------------------------
# 6. 직접 만든 시뮬레이터 vs Qiskit
# ---------------------------------------------------------------
gates = ["h", "t", "s", "y", "h", "tdg"]                 # 아무렇게나 고른 게이트 순서
mine = OneQubitSim().run(gates).state

qc = QuantumCircuit(1)
for name in gates:
    getattr(qc, name)(0)
theirs = Statevector(qc).data
print("6. 내 시뮬레이터:", show(mine))
print("   Qiskit       :", show(theirs))
print("   같은가:", same(mine, theirs))                            # True

ours = OneQubitSim().run(["h", "t", "h"])
print("   H, T, H 확률 (내 것):", show(ours.probabilities()))      # [0.854, 0.146]
print("   1,000번 측정 (내 것):", ours.sample(1000))               # 예: {'0': 861, '1': 139}

one = OneQubitSim().apply("h")
first = one.measure()
print("   |+>를 측정:", first, "-> 다시 측정:", one.measure(), one.measure(), one.measure())   # 첫 결과가 계속 반복된다
