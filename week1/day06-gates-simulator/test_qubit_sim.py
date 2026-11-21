"""
test_qubit_sim.py: 직접 만든 1큐비트 시뮬레이터가 Qiskit과 같은 답을 내는지 확인하는 테스트

실행 방법 (둘 중 아무거나)
    python test_qubit_sim.py      # 추가 설치 없이 실행. 테스트마다 ok / FAIL을 찍는다
    python -m pytest              # pytest가 깔려 있으면 이것도 된다

테스트 함수는 모두 test_ 로 시작한다. 안의 assert가 하나라도 틀리면 그 테스트는 실패한다.
"""

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from qubit_sim import GATES, OneQubitSim

NAMES = ["x", "y", "z", "h", "s", "t", "sdg", "tdg"]


def qiskit_state(names):
    """같은 게이트 순서를 Qiskit 회로로 만들어 상태 벡터를 구한다. 비교 기준(정답)이다."""
    qc = QuantumCircuit(1)
    for name in names:
        getattr(qc, name)(0)          # "h" -> qc.h(0) 처럼 이름으로 메서드를 찾아 부른다
    return Statevector(qc).data


# ---------------------------------------------------------------
# 1. 게이트 행렬 자체
# ---------------------------------------------------------------
def test_every_gate_is_unitary():
    """여섯 게이트와 S†, T† 모두 U†U = I 이어야 한다 (Day 5)."""
    for name in NAMES:
        U = GATES[name]
        assert np.allclose(U.conj().T @ U, np.eye(2)), name


def test_gate_relations():
    """각도로 보면 T = 45도, S = 90도, Z = 180도. 그리고 HZH = X."""
    X, Y, Z, H, S, T = (GATES[k] for k in "xyzhst")
    assert np.allclose(T @ T, S)
    assert np.allclose(S @ S, Z)
    assert np.allclose(H @ Z @ H, X)
    assert np.allclose(Y, 1j * X @ Z)
    assert np.allclose(GATES["sdg"] @ S, np.eye(2))     # S†가 S를 되돌린다


# ---------------------------------------------------------------
# 2. Qiskit과 상태 벡터 비교
# ---------------------------------------------------------------
def test_single_gates_match_qiskit():
    """게이트 하나씩: |0>에 걸었을 때와 |1>에 걸었을 때 모두 Qiskit과 같아야 한다."""
    for name in NAMES:
        assert np.allclose(OneQubitSim().run([name]).state, qiskit_state([name])), name
        assert np.allclose(OneQubitSim().run(["x", name]).state, qiskit_state(["x", name])), name


def test_random_sequences_match_qiskit():
    """게이트를 1~12개 무작위로 이어 붙인 회로 200개가 모두 Qiskit과 같아야 한다."""
    rng = np.random.default_rng(2026)                   # 시드를 고정해서 매번 같은 200개로 시험한다
    for _ in range(200):
        names = list(rng.choice(NAMES, size=rng.integers(1, 13)))
        mine = OneQubitSim().run(names).state
        assert np.allclose(mine, qiskit_state(names)), names
        assert np.isclose(np.sum(np.abs(mine) ** 2), 1)   # 확률의 합은 언제나 1


def test_unknown_gate_is_rejected():
    """모르는 게이트 이름은 조용히 넘어가지 말고 에러를 내야 한다."""
    try:
        OneQubitSim().apply("q")
    except ValueError:
        return
    raise AssertionError("ValueError가 나야 한다")


# ---------------------------------------------------------------
# 3. 측정
# ---------------------------------------------------------------
def test_measure_collapses_the_state():
    """측정 뒤 상태는 |0> 또는 |1>이고, 다시 측정하면 같은 값이 나와야 한다 (Day 4)."""
    for seed in range(20):
        sim = OneQubitSim(seed=seed).apply("h")
        first = sim.measure()
        assert np.allclose(sim.state, [1, 0] if first == 0 else [0, 1])
        assert all(sim.measure() == first for _ in range(5))


def test_sampling_matches_probability_and_qiskit():
    """H, T, H를 20,000번 측정하면 0의 비율이 cos²(22.5도) = 0.854 근처여야 한다. Qiskit도 마찬가지."""
    shots = 20000
    expect = np.cos(np.pi / 8) ** 2
    mine = OneQubitSim(seed=11).run(["h", "t", "h"]).sample(shots)
    assert mine["0"] + mine["1"] == shots

    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.t(0)
    qc.h(0)
    qc.measure(0, 0)
    theirs = AerSimulator(seed_simulator=11).run(qc, shots=shots).result().get_counts()

    # 20,000번이면 비율의 흔들림(표준편차)은 0.0025 정도다. 0.015는 그 6배라 넉넉하다.
    assert abs(mine["0"] / shots - expect) < 0.015
    assert abs(theirs.get("0", 0) / shots - expect) < 0.015


def test_same_seed_gives_same_counts():
    """시드가 같으면 랜덤 결과도 같아야 한다. 그래야 테스트를 다시 돌려도 결과가 같다."""
    a = OneQubitSim(seed=5).apply("h").sample(1000)
    b = OneQubitSim(seed=5).apply("h").sample(1000)
    assert a == b


def test_sample_does_not_change_the_state():
    """sample()은 '여러 번 새로 준비해서 측정'이라 상태를 바꾸면 안 된다. measure()와 다른 점이다."""
    sim = OneQubitSim(seed=1).apply("h")
    before = sim.state.copy()
    sim.sample(500)
    assert np.allclose(sim.state, before)


# ---------------------------------------------------------------
# pytest 없이 실행할 때 쓰는 아주 작은 실행기
# ---------------------------------------------------------------
if __name__ == "__main__":
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("ok  ", name)
        except Exception as e:                      # assert가 틀리면 AssertionError가 여기로 온다
            failed += 1
            print("FAIL", name, "->", repr(e))
    print(f"\n{len(tests) - failed}/{len(tests)} 통과")
    raise SystemExit(1 if failed else 0)
