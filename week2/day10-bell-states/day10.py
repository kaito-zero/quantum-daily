"""
Day 10: 곱 상태와 얽힌 상태, 벨 상태 4개

목표
  H + CNOT으로 벨 상태를 만들고, 측정하면 00과 11만 나오는 것을 확인한다.
  나머지 세 벨 상태도 만들고, 곱 상태와 얽힌 상태를 숫자로 구별해 본다.

흐름
  1. H(0번) + CNOT(0->1)로 첫 번째 벨 상태 |Φ+> = 0.707|00> + 0.707|11>을 만든다.
  2. 1,000번 측정하면 00과 11만 나온다. 두 큐비트의 결과가 항상 같다.
  3. 시작 상태를 X로 바꾸면 나머지 세 벨 상태 |Φ->, |Ψ+>, |Ψ->가 나온다.
  4. 곱 상태 검사: [a, b, c, d]가 두 큐비트 상태의 텐서곱이면 a*d - b*c = 0 이다.
     벨 상태는 이 값이 0이 아니다. 텐서곱으로 쪼갤 수 없는 상태가 얽힌 상태다.
  5. 한 큐비트만 보면 반반이다. 두 큐비트를 같이 봐야 "항상 같다"가 보인다.
  6. 벨 상태 네 개의 측정 결과를 히스토그램 그림(bell_histogram.png)으로 저장한다.

예상 결과
  측정 횟수는 실행할 때마다 조금씩 달라진다. 나머지는 항상 같다.
"""

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


def num(x):
    """숫자 하나를 0.707, -0.5처럼 짧게 적는다. 소수 셋째 자리까지만."""
    x = complex(x)
    re = round(x.real, 3) + 0.0
    im = round(x.imag, 3) + 0.0
    if im == 0:
        return f"{re:g}"
    if re == 0:
        return f"{im:g}j"
    return f"{re:g}{im:+g}j"


def ket(v):
    """두 큐비트 상태를 0.707|00> + 0.707|11> 처럼 적는다 (Day 8과 같다)."""
    parts = [f"{num(a)}|{k:02b}>" for k, a in enumerate(np.asarray(v)) if abs(a) > 1e-9]
    return " + ".join(parts).replace("+ -", "- ")


def bell(first_x0=False, first_x1=False):
    """벨 상태를 만드는 회로. X를 먼저 걸지 말지에 따라 네 가지가 나온다."""
    qc = QuantumCircuit(2)
    if first_x0:
        qc.x(0)              # 0번 큐비트를 1로 바꾸고 시작 -> 위상이 바뀐다 (Φ- 또는 Ψ-)
    if first_x1:
        qc.x(1)              # 1번 큐비트를 1로 바꾸고 시작 -> 짝이 바뀐다 (Ψ+ 또는 Ψ-)
    qc.h(0)                  # 0번 큐비트를 반반으로
    qc.cx(0, 1)              # 0번이 1일 때만 1번을 뒤집는다 -> 두 큐비트가 묶인다
    return qc


def entangled(v):
    """[a, b, c, d]가 곱 상태인지 검사한다. a*d - b*c 가 0이 아니면 얽힌 상태다."""
    a, b, c, d = v
    return abs(a * d - b * c) > 1e-9


sim = AerSimulator()

# ---------------------------------------------------------------
# 1. 첫 번째 벨 상태 |Φ+>
# ---------------------------------------------------------------
qc = bell()
phi_plus = Statevector(qc).data
print("1. H(0번), CNOT(0->1):", ket(phi_plus))          # 0.707|00> + 0.707|11>
# H 뒤에는 0.707|00> + 0.707|01>. CNOT이 |01>을 |11>로 바꿔서 |00>과 |11>만 남는다.

# ---------------------------------------------------------------
# 2. 1,000번 측정
# ---------------------------------------------------------------
m = qc.copy()
m.measure_all()                                        # 두 큐비트를 모두 측정한다
counts = sim.run(m, shots=1000).result().get_counts()
print("2. 1,000번 측정:", counts)                      # 예: {'00': 503, '11': 497}. 01과 10은 한 번도 안 나온다
# 오른쪽 글자(0번)와 왼쪽 글자(1번)가 항상 같다. 0번이 0이면 1번도 0, 0번이 1이면 1번도 1.

# ---------------------------------------------------------------
# 3. 벨 상태 네 개
# ---------------------------------------------------------------
recipes = {
    "Φ+": (False, False),     # (|00> + |11>) / √2
    "Φ-": (True, False),      # (|00> - |11>) / √2
    "Ψ+": (False, True),      # (|01> + |10>) / √2
    "Ψ-": (True, True),       # (|01> - |10>) / √2 (전체에 -1이 붙어 나온다)
}
states = {}
for name, (x0, x1) in recipes.items():
    states[name] = Statevector(bell(x0, x1)).data
    print(f"3. {name}: {ket(states[name])}")
# Φ+ : 0.707|00> + 0.707|11>
# Φ- : 0.707|00> - 0.707|11>
# Ψ+ : 0.707|01> + 0.707|10>
# Ψ- : -0.707|01> + 0.707|10>   전체에 붙은 -1은 측정으로 구별되지 않는다 (Day 6의 전체 위상)
# Φ 두 개는 결과가 항상 같고(00, 11), Ψ 두 개는 항상 반대다(01, 10).

# 네 벨 상태는 서로 겹치지 않는다(내적이 0). 그래서 측정으로 서로 구별할 수 있는 한 묶음(기저)이다.
names = list(states)
overlaps = [abs(np.vdot(states[p], states[q])) for p in names for q in names if p < q]
print("   서로 다른 벨 상태끼리 내적이 모두 0인가:", bool(np.allclose(overlaps, 0)))   # True

# ---------------------------------------------------------------
# 4. 곱 상태와 얽힌 상태
# ---------------------------------------------------------------
plus = np.array([1, 1]) / np.sqrt(2)
product = np.kron(plus, plus)                          # Day 8의 |+> ⊗ |+>
print("4. |+>⊗|+> =", ket(product), "-> a*d - b*c =", num(product[0] * product[3] - product[1] * product[2]),
      "/ 얽힘:", entangled(product))                    # 0 / False
for name, v in states.items():
    print(f"   {name}: a*d - b*c = {num(v[0] * v[3] - v[1] * v[2])} / 얽힘: {entangled(v)}")   # 0.5 또는 -0.5 / True
# 곱 상태 [p, q] ⊗ [r, s] = [pr, ps, qr, qs] 이면 a*d - b*c = pr*qs - ps*qr = 0 이 항상 성립한다.
# 그래서 이 값이 0이 아니면 어떤 두 상태의 텐서곱으로도 만들 수 없다. 그것이 얽힘이다.

# ---------------------------------------------------------------
# 5. 한 큐비트만 보면 반반
# ---------------------------------------------------------------
sv = Statevector(phi_plus)
print("5. 0번 큐비트만 본 확률:", [round(float(p), 3) for p in sv.probabilities([0])],   # [0.5, 0.5]
      "/ 1번 큐비트만:", [round(float(p), 3) for p in sv.probabilities([1])])              # [0.5, 0.5]
# 각 큐비트만 보면 그냥 동전 던지기다. "둘이 항상 같다"는 정보는 두 큐비트를 같이 볼 때만 보인다.

# ---------------------------------------------------------------
# 6. 히스토그램 그림 저장 (matplotlib이 있을 때만)
# ---------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")                              # 창을 띄우지 않고 파일로만 저장한다
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    for ax, (name, (x0, x1)) in zip(axes, recipes.items()):
        m = bell(x0, x1)
        m.measure_all()
        c = sim.run(m, shots=1000).result().get_counts()
        labels = ["00", "01", "10", "11"]
        ax.bar(labels, [c.get(k, 0) for k in labels], color="#2b6f77")
        ax.set_title(f"|{name}>")
        ax.set_ylim(0, 600)
    axes[0].set_ylabel("counts (1,000 shots)")       # 그림 글자는 영어로 (한글 글꼴이 없으면 네모로 깨진다)
    fig.suptitle("Bell states: measurement results")
    fig.tight_layout()
    fig.savefig("bell_histogram.png", dpi=150)
    print("6. bell_histogram.png 저장")
except ImportError:
    print("6. matplotlib이 없어서 그림은 건너뜀 (pip install matplotlib)")
