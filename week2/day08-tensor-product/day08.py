"""
Day 8: 텐서곱과 두 큐비트

목표
  큐비트 두 개의 상태를 텐서곱(np.kron)으로 만들고, Qiskit의 큐비트 순서(오른쪽이 0번)를 직접 확인한다.

흐름
  1. 큐비트 두 개의 상태는 진폭 네 개짜리 벡터다. 순서는 |00>, |01>, |10>, |11>.
  2. 텐서곱: 큐비트 하나짜리 상태 두 개를 곱해서 두 큐비트 상태를 만든다. [a, b] ⊗ [c, d] = [ac, ad, bc, bd].
  3. 텐서곱은 순서가 중요하다. |0> ⊗ |1> 과 |1> ⊗ |0> 은 다른 상태다.
  4. Qiskit은 오른쪽 글자가 0번 큐비트다. 0번 큐비트에만 X를 걸면 '01'이 나온다.
     그래서 Qiskit의 상태 벡터는 (1번 큐비트) ⊗ (0번 큐비트) 순서로 계산한다.
  5. 게이트도 텐서곱으로 4x4 행렬을 만든다. 0번 큐비트에 H = I ⊗ H, 1번 큐비트에 H = H ⊗ I.
  6. 직접 만든 행렬과 상태가 Qiskit의 Operator, Statevector와 같은지 비교하고, 1,000번 측정해 본다.

예상 결과
  6번의 측정 횟수만 실행할 때마다 조금씩 달라진다.
"""

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from qiskit_aer import AerSimulator


def num(x):
    """숫자 하나를 0.5, -0.707, 1j처럼 짧게 적는다. 소수 셋째 자리까지만."""
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
    """두 큐비트 상태를 0.707|00> + 0.707|11> 처럼 적는다. 0인 칸은 뺀다."""
    parts = []
    for k, amp in enumerate(np.asarray(v)):
        if abs(amp) > 1e-9:
            parts.append(f"{num(amp)}|{k:02b}>")       # k:02b 는 0~3을 '00'~'11' 두 자리 이진수로 바꾼다
    return " + ".join(parts).replace("+ -", "- ")


same = lambda a, b: bool(np.allclose(a, b))         # 두 배열이 (계산 오차를 빼고) 같은가

ket0 = np.array([1, 0])
ket1 = np.array([0, 1])
plus = np.array([1, 1]) / np.sqrt(2)
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

# ---------------------------------------------------------------
# 1. 두 큐비트 상태는 진폭 네 개
# ---------------------------------------------------------------
print("1. 칸 번호와 이름:", [f"{k} -> |{k:02b}>" for k in range(4)])   # 0 -> |00>, 1 -> |01>, 2 -> |10>, 3 -> |11>
state = np.array([0.5, 0.5, 0.5, 0.5])                   # 네 칸이 모두 0.5
print("   [0.5, 0.5, 0.5, 0.5]의 확률:", show(np.abs(state) ** 2), "합:", np.sum(np.abs(state) ** 2))   # 각 0.25, 합 1.0
# 규칙은 Day 3, 4와 같다. 진폭 크기의 제곱이 확률이고, 확률의 합은 1이다. 칸이 네 개로 늘었을 뿐이다.

# ---------------------------------------------------------------
# 2. 텐서곱: [a, b] ⊗ [c, d] = [a*c, a*d, b*c, b*d]
# ---------------------------------------------------------------
def kron_by_hand(u, v):
    """텐서곱을 손으로 계산한다. 왼쪽 숫자 하나하나에 오른쪽 벡터 전체를 곱해서 이어 붙인다."""
    a, b = u
    c, d = v
    return np.array([a * c, a * d, b * c, b * d])


print("2. |0> ⊗ |1> 손으로:", show(kron_by_hand(ket0, ket1)), "/ np.kron:", show(np.kron(ket0, ket1)))   # [0, 1, 0, 0] = |01>
print("   |1> ⊗ |0> 손으로:", show(kron_by_hand(ket1, ket0)), "/ np.kron:", show(np.kron(ket1, ket0)))   # [0, 0, 1, 0] = |10>
both_plus = np.kron(plus, plus)
print("   |+> ⊗ |+> =", show(both_plus), "=", ket(both_plus))   # [0.5, 0.5, 0.5, 0.5]. 네 결과가 모두 25%
mix = np.kron(plus, ket0)
print("   |+> ⊗ |0> =", show(mix), "=", ket(mix))                 # [0.707, 0, 0.707, 0] = 0.707|00> + 0.707|10>

# ---------------------------------------------------------------
# 3. 순서가 중요하다
# ---------------------------------------------------------------
print("3. |+> ⊗ |0> 와 |0> ⊗ |+> 가 같은가:", same(np.kron(plus, ket0), np.kron(ket0, plus)))   # False
print("   |0> ⊗ |+> =", ket(np.kron(ket0, plus)))                                              # 0.707|00> + 0.707|01>
# 왼쪽에 놓인 큐비트가 이름의 왼쪽 글자를 정한다. |+> ⊗ |0>은 왼쪽 글자가 0과 1로 갈리고, |0> ⊗ |+>는 오른쪽 글자가 갈린다.

# ---------------------------------------------------------------
# 4. Qiskit의 큐비트 순서: 오른쪽 글자가 0번 큐비트
# ---------------------------------------------------------------
qc = QuantumCircuit(2)
qc.x(0)                                                    # 0번 큐비트만 1로 바꾼다
sv = Statevector(qc)
print("4. 0번 큐비트에 X:", show(sv.data), "->", {str(k): float(p) for k, p in sv.probabilities_dict().items()})   # [0, 1, 0, 0] -> {'01': 1.0}
# 0번 큐비트를 바꿨는데 이름은 '01'이다. 오른쪽 글자가 0번 큐비트라서다.
print("   np.kron(|1>, |0>) 와 같은가:", same(sv.data, np.kron(ket1, ket0)),     # False
      "/ np.kron(|0>, |1>) 와 같은가:", same(sv.data, np.kron(ket0, ket1)))      # True
# 그래서 Qiskit 상태를 np.kron으로 만들 때는 (1번 큐비트) ⊗ (0번 큐비트), 즉 큰 번호부터 적는다.

qc = QuantumCircuit(2)
qc.h(1)                                                    # 1번 큐비트에만 H
mine = np.kron(plus, ket0)                                 # 1번 = |+>, 0번 = |0>
print("   1번 큐비트에 H:", ket(Statevector(qc).data), "/ 직접 만든 것과 같은가:", same(Statevector(qc).data, mine))   # 0.707|00> + 0.707|10> / True

# ---------------------------------------------------------------
# 5. 게이트도 텐서곱: 0번 큐비트에 H = I ⊗ H
# ---------------------------------------------------------------
H_on_0 = np.kron(I, H)                                     # 1번에는 아무것도 안 하고(I), 0번에 H
H_on_1 = np.kron(H, I)
print("5. I ⊗ H =", show(H_on_0))                          # 4x4 행렬
qc = QuantumCircuit(2)
qc.h(0)
print("   Qiskit의 '0번에 H' 행렬과 같은가: I⊗H", same(Operator(qc).data, H_on_0), "/ H⊗I", same(Operator(qc).data, H_on_1))   # True / False

start = np.kron(ket0, ket0)                                # |00>
after = np.kron(H, H) @ start                              # 두 큐비트 모두에 H
print("   H ⊗ H 를 |00>에 곱하기:", show(after))           # [0.5, 0.5, 0.5, 0.5]
print("   H ⊗ H 는 유니터리인가:", same(np.kron(H, H).conj().T @ np.kron(H, H), np.eye(4)))   # True. 4x4에서도 U†U = I

# ---------------------------------------------------------------
# 6. 1,000번 측정해 보기
# ---------------------------------------------------------------
qc = QuantumCircuit(2, 2)
qc.h(1)
qc.x(0)
qc.measure([0, 1], [0, 1])                                 # 0번 큐비트 -> 0번 칸, 1번 큐비트 -> 1번 칸
counts = AerSimulator().run(qc, shots=1000).result().get_counts()
print("6. 1번에 H, 0번에 X:", counts)                       # 예: {'01': 497, '11': 503}
# 결과 이름도 오른쪽이 0번 칸이다. 0번 큐비트는 항상 1이라 오른쪽 글자는 늘 1, 왼쪽 글자는 반반이다.
expect = np.abs(np.kron(plus, ket1)) ** 2
print("   직접 계산한 확률:", {f"{k:02b}": round(float(p), 3) for k, p in enumerate(expect) if p > 0})   # {'01': 0.5, '11': 0.5}
