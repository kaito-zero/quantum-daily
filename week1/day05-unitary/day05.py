"""
Day 5: 행렬과 유니터리

목표
  게이트의 정체가 행렬이라는 것을 직접 곱해 보며 확인하고,
  게이트가 되려면 왜 유니터리(U†U = I)여야 하는지 숫자로 확인한다.

흐름
  1. 게이트는 숫자 네 개가 들어 있는 2x2 행렬이다. X, Z, H를 NumPy 배열로 만든다.
  2. 게이트를 건다 = 행렬을 상태 벡터에 곱한다.
     한 줄씩 손으로 계산한 값과 @ 연산자로 계산한 값을 비교한다.
  3. 게이트를 연달아 건다 = 행렬끼리 곱한다. 식에서는 나중에 건 게이트가 왼쪽에 온다.
     순서를 바꾸면 결과가 달라질 수 있다.
  4. 켤레 전치(†, 대거): 행과 열을 바꾸고(전치), 성분마다 j의 부호를 뒤집는다(켤레).
  5. 유니터리 검사: U†U = I 인 행렬이 유니터리다. X, Z, H, S는 유니터리지만 아무 행렬이나 그런 것은 아니다.
  6. 게이트가 유니터리여야 하는 이유: 유니터리 행렬은 확률의 합을 1로 유지한다.
     그리고 U†를 걸면 원래 상태로 되돌릴 수 있다.
  7. 직접 만든 행렬과 계산 결과가 Qiskit의 Operator와 같은지 비교한다.

예상 결과
  측정이 없어서 실행할 때마다 결과가 같다. 주석에 적은 값이 그대로 나온다.
"""

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector


def num(x):
    """숫자 하나를 0.707, -1, 2j처럼 짧게 적는다. 소수 셋째 자리까지만 적는다."""
    x = complex(x)
    re = round(x.real, 3) + 0.0   # + 0.0 은 -0.0 을 0.0 으로 바꾼다
    im = round(x.imag, 3) + 0.0
    if im == 0:
        return f"{re:g}"
    if re == 0:
        return f"{im:g}j"
    return f"{re:g}{im:+g}j"


def show(a):
    """벡터는 [1, 0], 행렬은 [[0, 1], [1, 0]]처럼 보기 좋게 적어 주는 도우미."""
    a = np.asarray(a)
    if a.ndim == 1:                                          # 벡터: 숫자를 한 줄로 적는다
        return "[" + ", ".join(num(x) for x in a) + "]"
    return "[" + ", ".join(show(row) for row in a) + "]"     # 행렬: 가로줄(행)마다 벡터처럼 적는다


# ---------------------------------------------------------------
# 1. 게이트는 행렬이다
# ---------------------------------------------------------------
X = np.array([[0, 1],
              [1, 0]])                      # 0과 1을 맞바꾼다
Z = np.array([[1, 0],
              [0, -1]])                     # 둘째 진폭의 부호만 뒤집는다
H = np.array([[1, 1],
              [1, -1]]) / np.sqrt(2)        # 중첩을 만든다. 1/√2 = 0.707
I = np.eye(2)                               # 단위행렬. 곱해도 아무것도 바뀌지 않는다
print("1. X =", show(X))                    # [[0, 1], [1, 0]]
print("   Z =", show(Z))                    # [[1, 0], [0, -1]]
print("   H =", show(H))                    # [[0.707, 0.707], [0.707, -0.707]]
print("   I =", show(I))                    # [[1, 0], [0, 1]]

# ---------------------------------------------------------------
# 2. 게이트를 건다 = 행렬을 상태 벡터에 곱한다
# ---------------------------------------------------------------
def apply_by_hand(U, v):
    """행렬 x 벡터를 한 줄씩 직접 계산한다. U @ v 가 하는 일을 풀어 쓴 것이다."""
    top = U[0][0] * v[0] + U[0][1] * v[1]       # 결과의 첫째 숫자: 행렬 첫째 줄과 벡터를 짝지어 곱한 뒤 더한다
    bottom = U[1][0] * v[0] + U[1][1] * v[1]    # 결과의 둘째 숫자: 행렬 둘째 줄로 같은 일을 한다
    return np.array([top, bottom])


ket0 = np.array([1, 0])                     # |0>
ket1 = np.array([0, 1])                     # |1>
plus = np.array([1, 1]) / np.sqrt(2)        # |+> = [0.707, 0.707]
minus = np.array([1, -1]) / np.sqrt(2)      # |-> = [0.707, -0.707]

# 손으로 계산한 값과 @ 연산자(행렬 곱)로 계산한 값이 같은지 나란히 찍는다.
print("2. X|0> 손으로:", show(apply_by_hand(X, ket0)), "/ @ 로:", show(X @ ket0))    # [0, 1] = |1>
print("   H|0> 손으로:", show(apply_by_hand(H, ket0)), "/ @ 로:", show(H @ ket0))    # [0.707, 0.707] = |+>
print("   Z|+> 손으로:", show(apply_by_hand(Z, plus)), "/ @ 로:", show(Z @ plus))    # [0.707, -0.707] = |->
# X|0>을 풀어 쓰면: 첫째 줄 0 x 1 + 1 x 0 = 0, 둘째 줄 1 x 1 + 0 x 0 = 1. 그래서 [0, 1].

# ---------------------------------------------------------------
# 3. 게이트를 연달아 건다 = 행렬끼리 곱한다
# ---------------------------------------------------------------
print("3. H @ H =", show(H @ H))                # [[1, 0], [0, 1]] = I. H를 두 번 걸면 원래대로 돌아온다
print("   H @ Z @ H =", show(H @ Z @ H))        # [[0, 1], [1, 0]] = X. H, Z, H를 차례로 걸면 X와 같다

# 회로에서 X를 먼저, H를 나중에 걸면 상태 v는 H @ (X @ v) 가 된다.
# 괄호를 옮기면 (H @ X) @ v 이다. 그래서 식에서는 나중에 건 게이트가 왼쪽에 온다.
x_then_h = H @ X       # 회로: X 먼저, H 나중
h_then_x = X @ H       # 회로: H 먼저, X 나중
print("   X 먼저, H 나중:", show(x_then_h), "-> |0>에 걸면", show(x_then_h @ ket0))    # [0.707, -0.707] = |->
print("   H 먼저, X 나중:", show(h_then_x), "-> |0>에 걸면", show(h_then_x @ ket0))    # [0.707, 0.707] = |+>
print("   두 행렬이 같은가:", np.allclose(x_then_h, h_then_x))                        # False. 순서가 중요하다

# ---------------------------------------------------------------
# 4. 켤레 전치(†)
# ---------------------------------------------------------------
def dagger(U):
    """켤레 전치. 성분마다 j의 부호를 뒤집고(.conj()), 행과 열을 바꾼다(.T)."""
    return np.asarray(U).conj().T


M = np.array([[1, 2j],
              [3, 4]])                           # 게이트가 아니라 연습용 행렬
print("4. M =", show(M))                         # [[1, 2j], [3, 4]]
print("   전치 M.T =", show(M.T))                 # [[1, 3], [2j, 4]]. 2j와 3이 자리를 바꿨다
print("   켤레 M.conj() =", show(M.conj()))       # [[1, -2j], [3, 4]]. 2j가 -2j가 됐다
print("   켤레 전치 M† =", show(dagger(M)))        # [[1, 3], [-2j, 4]]. 둘 다 한 것

S = np.array([[1, 0],
              [0, 1j]])                          # S 게이트. 둘째 진폭에 1j를 곱한다(Day 2에서 본 90도 회전)
print("   S =", show(S), "-> S† =", show(dagger(S)))     # [[1, 0], [0, -1j]]
print("   X† = X:", np.allclose(dagger(X), X),
      "/ Z† = Z:", np.allclose(dagger(Z), Z),
      "/ H† = H:", np.allclose(dagger(H), H))            # 모두 True
# X, Z, H는 성분이 모두 실수이고 대각선을 기준으로 대칭이라서, 켤레 전치를 해도 자기 자신이다.

# ---------------------------------------------------------------
# 5. 유니터리 검사: U†U = I
# ---------------------------------------------------------------
def is_unitary(U):
    """U†U 가 단위행렬 I 와 같으면 유니터리다."""
    return bool(np.allclose(dagger(U) @ U, np.eye(2)))


A = np.array([[2, 0],
              [0, 1]])                           # 첫째 진폭을 2배로 만드는 행렬. 게이트가 될 수 있을까?
print("5. H†H =", show(dagger(H) @ H))           # [[1, 0], [0, 1]] = I
print("   S†S =", show(dagger(S) @ S))           # [[1, 0], [0, 1]] = I
print("   A†A =", show(dagger(A) @ A))           # [[4, 0], [0, 1]]. I가 아니다
for name, U in [("X", X), ("Z", Z), ("H", H), ("S", S), ("A", A)]:
    print(f"   {name}는 유니터리인가: {is_unitary(U)}")     # X, Z, H, S는 True. A는 False

# ---------------------------------------------------------------
# 6. 게이트가 유니터리여야 하는 이유
# ---------------------------------------------------------------
def total_prob(state):
    """확률의 합. Day 4의 보른 규칙대로 진폭 크기의 제곱을 모두 더한다."""
    return round(float(np.sum(np.abs(state) ** 2)), 3)


state = np.array([0.6, 0.8j])                    # 0이 36%, 1이 64%. 합은 1
print("6. 처음 상태:", show(state), "-> 확률의 합", total_prob(state))            # 1.0
print("   H를 건 뒤:", show(H @ state), "-> 확률의 합", total_prob(H @ state))    # 1.0
print("   S를 건 뒤:", show(S @ state), "-> 확률의 합", total_prob(S @ state))    # 1.0
print("   A를 건 뒤:", show(A @ state), "-> 확률의 합", total_prob(A @ state))    # 2.08. 확률의 합이 1을 넘는다
# 유니터리 행렬은 벡터의 길이를 바꾸지 않는다. 그래서 어떤 게이트를 걸어도 확률의 합이 1로 남는다.
# A처럼 유니터리가 아닌 행렬을 걸면 결과가 더 이상 상태 벡터가 아니다. 그래서 게이트가 될 수 없다.

back = dagger(H) @ (H @ state)                   # H를 건 뒤 H†를 건다
print("   H를 걸고 H†를 건 상태:", show(back))                                    # [0.6, 0.8j]. 처음 상태 그대로
# U†U = I 라는 것은 U를 건 뒤 U†를 걸면 아무것도 안 한 것과 같다는 뜻이다.
# 그래서 양자 게이트는 항상 되돌릴 수 있다. Day 4에서 본 측정은 게이트가 아니라서 되돌릴 수 없다.

# ---------------------------------------------------------------
# 7. Qiskit Operator와 비교하기
# ---------------------------------------------------------------
print("7. 직접 만든 행렬이 Qiskit의 행렬과 같은가")
for name, U in [("X", X), ("Z", Z), ("H", H), ("S", S)]:
    op = Operator.from_label(name)               # Qiskit이 가지고 있는 게이트 행렬
    print(f"   {name}: {bool(np.allclose(op.data, U))} / Qiskit의 유니터리 검사: {op.is_unitary()}")    # 모두 True
print("   A를 Operator로 만들어 검사:", Operator(A).is_unitary())                 # False

# 상태 벡터에 곱한 결과도 비교한다.
ours = H @ ket0                                          # 직접 곱한 결과
theirs = Statevector(ket0).evolve(Operator(H))           # Qiskit으로 |0>에 H를 건 결과
print("   H|0> 직접:", show(ours), "/ Qiskit:", show(theirs.data),
      "/ 같은가:", bool(np.allclose(ours, theirs.data)))                         # True

# 회로 전체를 행렬 하나로 바꿔서, 3번에서 곱한 순서가 맞았는지 확인한다.
qc = QuantumCircuit(1)
qc.x(0)                # X 먼저
qc.h(0)                # H 나중
circuit_matrix = Operator(qc).data                       # 회로 전체에 해당하는 행렬
print("   회로(X 먼저, H 나중)의 행렬:", show(circuit_matrix))                    # [[0.707, 0.707], [-0.707, 0.707]]
print("   H @ X 와 같은가:", bool(np.allclose(circuit_matrix, H @ X)),
      "/ X @ H 와 같은가:", bool(np.allclose(circuit_matrix, X @ H)))            # True / False
