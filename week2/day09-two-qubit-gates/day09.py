"""
Day 9: 2큐비트 게이트 CNOT, CZ, SWAP과 '제어' 연산

목표
  큐비트 두 개에 함께 거는 게이트 세 개를 4x4 행렬로 직접 쓰고, 회로로 동작을 확인한다.

흐름
  1. CNOT(제어 0번, 타깃 1번)의 4x4 행렬을 직접 쓰고 Qiskit의 qc.cx(0, 1)과 비교한다.
  2. 진리표: 네 가지 입력(00, 01, 10, 11)을 넣어 출력을 회로로 확인한다.
     제어 큐비트가 1일 때만 타깃 큐비트가 뒤집힌다.
  3. 방향이 중요하다. qc.cx(1, 0)은 다른 행렬이다. 교과서의 CNOT 행렬은 Qiskit에서는 cx(1, 0)이다.
  4. CZ: |11>일 때만 부호를 뒤집는다. 제어와 타깃을 바꿔도 같은 게이트다. H로 감싸면 CNOT이 된다.
  5. SWAP: 두 큐비트를 맞바꾼다. CNOT 세 개로 만들 수 있다는 것을 Operator로 검증한다.
  6. 네 게이트 모두 유니터리다.

예상 결과
  측정이 없어서 실행할 때마다 결과가 같다.
"""

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector


def num(x):
    """숫자 하나를 0.5, -1처럼 짧게 적는다. 소수 셋째 자리까지만."""
    x = complex(x)
    re = round(x.real, 3) + 0.0   # + 0.0 은 -0.0 을 0.0 으로 바꾼다
    im = round(x.imag, 3) + 0.0
    if im == 0:
        return f"{re:g}"
    if re == 0:
        return f"{im:g}j"
    return f"{re:g}{im:+g}j"


def show(a):
    """벡터는 [1, 0, 0, 0], 행렬은 줄마다 [..]로 적는다."""
    a = np.asarray(a)
    if a.ndim == 1:
        return "[" + ", ".join(num(x) for x in a) + "]"
    return "[" + ", ".join(show(row) for row in a) + "]"


def ket(v):
    """두 큐비트 상태를 0.707|00> + 0.707|11> 처럼 적는다 (Day 8과 같다)."""
    parts = [f"{num(a)}|{k:02b}>" for k, a in enumerate(np.asarray(v)) if abs(a) > 1e-9]
    return " + ".join(parts).replace("+ -", "- ")


def label(v):
    """기저 상태 하나뿐이면 그 이름('01' 등)을 돌려준다."""
    return f"{int(np.argmax(np.abs(v))):02b}"


same = lambda a, b: bool(np.allclose(a, b))

# ---------------------------------------------------------------
# 1. CNOT(제어 0번, 타깃 1번)의 행렬
# ---------------------------------------------------------------
# 칸 순서는 |00>, |01>, |10>, |11> (Day 8). 오른쪽 글자가 0번 큐비트다.
# 제어(0번, 오른쪽 글자)가 1인 칸은 |01>과 |11>. 이 둘에서 타깃(1번, 왼쪽 글자)을 뒤집는다 = 두 칸을 맞바꾼다.
CX01 = np.array([[1, 0, 0, 0],
                 [0, 0, 0, 1],      # |01> 자리에는 |11>의 진폭이 온다
                 [0, 0, 1, 0],
                 [0, 1, 0, 0]])     # |11> 자리에는 |01>의 진폭이 온다
qc = QuantumCircuit(2)
qc.cx(0, 1)                         # 첫 번째 숫자가 제어, 두 번째 숫자가 타깃
print("1. 직접 쓴 CNOT(0->1):", show(CX01))
print("   qc.cx(0, 1)의 행렬과 같은가:", same(Operator(qc).data, CX01))              # True

# ---------------------------------------------------------------
# 2. 진리표: 네 가지 입력을 회로로 확인
# ---------------------------------------------------------------
print("2. 입력 -> 출력 (제어 0번, 타깃 1번)")
for k in range(4):
    bits = f"{k:02b}"                           # '00', '01', '10', '11'
    qc = QuantumCircuit(2)
    if bits[1] == "1":                          # 오른쪽 글자 = 0번 큐비트
        qc.x(0)
    if bits[0] == "1":                          # 왼쪽 글자 = 1번 큐비트
        qc.x(1)
    qc.cx(0, 1)
    out = Statevector(qc).data
    print(f"   {bits} -> {label(out)}")
# 00 -> 00, 01 -> 11, 10 -> 10, 11 -> 01
# 제어(오른쪽 글자)가 1인 01과 11만 왼쪽 글자가 뒤집혔다.

# ---------------------------------------------------------------
# 3. 방향이 중요하다
# ---------------------------------------------------------------
qc = QuantumCircuit(2)
qc.cx(1, 0)                                     # 제어 1번, 타깃 0번
CX10 = Operator(qc).data.real
print("3. qc.cx(1, 0):", show(CX10))           # [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
print("   cx(0, 1)과 같은가:", same(CX10, CX01))  # False
textbook = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]])             # 책에서 흔히 보는 CNOT 행렬 (왼쪽 큐비트가 제어)
print("   교과서 CNOT 행렬은 cx(1, 0)인가:", same(textbook, CX10))   # True
# 책은 보통 왼쪽 큐비트를 첫 번째(제어)로 쓴다. Qiskit은 오른쪽이 0번이라 그 행렬이 cx(1, 0)이 된다.

# ---------------------------------------------------------------
# 4. CZ
# ---------------------------------------------------------------
CZ = np.diag([1, 1, 1, -1])                     # |11>일 때만 부호를 뒤집는다
a = QuantumCircuit(2)
a.cz(0, 1)
b = QuantumCircuit(2)
b.cz(1, 0)
print("4. CZ =", show(CZ))
print("   cz(0, 1) = cz(1, 0) = 직접 쓴 CZ:", same(Operator(a).data, CZ), same(Operator(b).data, CZ))   # True True
# 제어와 타깃을 바꿔도 같다. 둘 다 1일 때만 -1을 곱하니까 어느 쪽이 제어인지 구별이 없다.

qc = QuantumCircuit(2)
qc.h(0)
qc.h(1)
qc.cz(0, 1)
print("   H, H, CZ 다음 상태:", ket(Statevector(qc).data))   # 0.5|00> + 0.5|01> + 0.5|10> - 0.5|11>
# 확률은 넷 다 25% 그대로다. 바뀐 것은 |11>의 부호(위상)뿐이다. Day 6의 Z와 같은 이야기다.

wrapped = QuantumCircuit(2)
wrapped.h(1)
wrapped.cz(0, 1)
wrapped.h(1)                                    # 타깃(1번)을 H로 감싼다
print("   H(1번)·CZ·H(1번) = cx(0, 1):", same(Operator(wrapped).data, CX01))   # True
# Day 6의 H·Z·H = X를 두 큐비트로 늘린 것이다. 제어가 1일 때만 타깃에 Z가 걸리니, H로 감싸면 제어가 1일 때만 X.

# ---------------------------------------------------------------
# 5. SWAP과 CNOT 세 개
# ---------------------------------------------------------------
SWAP = np.array([[1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]])                 # |01>과 |10>의 진폭을 맞바꾼다
qc = QuantumCircuit(2)
qc.x(0)                                         # |01>에서 시작
qc.swap(0, 1)
print("5. |01>에 SWAP:", ket(Statevector(qc).data))   # 1|10>

three = QuantumCircuit(2)
three.cx(0, 1)
three.cx(1, 0)
three.cx(0, 1)
sw = QuantumCircuit(2)
sw.swap(0, 1)
print("   CNOT 세 개의 행렬:", show(Operator(three).data.real))
print("   SWAP과 같은가 (Operator.equiv):", Operator(three).equiv(Operator(sw)))   # True
print("   직접 곱해도 같은가:", same(CX01 @ CX10 @ CX01, SWAP))                  # True
# 식에서는 나중에 건 게이트가 왼쪽에 온다 (Day 5). cx(0,1), cx(1,0), cx(0,1)은 앞뒤가 대칭이라 식도 같은 모양이다.

# ---------------------------------------------------------------
# 6. 네 게이트 모두 유니터리
# ---------------------------------------------------------------
for name, U in [("CX01", CX01), ("CX10", CX10), ("CZ", CZ), ("SWAP", SWAP)]:
    print(f"6. {name}: U†U = I 인가 {same(U.conj().T @ U, np.eye(4))}")   # 모두 True
