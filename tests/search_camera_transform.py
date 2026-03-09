import numpy as np
from itertools import permutations, product

R = np.array(
    [
        [0.01596141, 0.44885520, -0.89346194],
        [0.99911254, 0.02767543, 0.03175234],
        [0.03897914, -0.89317584, -0.44801512],
    ],
    dtype=np.float64,
)

# "mostly correct" quaternion (wxyz)
q_target = np.array([0.017026, 0.197039, 0.00342095, -0.980242], dtype=np.float64)
q_target = q_target / np.linalg.norm(q_target)


def rotmat_to_quat_wxyz(M: np.ndarray) -> np.ndarray:
    # Robust conversion; assumes M is a proper rotation (det ~ 1)
    m00, m01, m02 = M[0, 0], M[0, 1], M[0, 2]
    m10, m11, m12 = M[1, 0], M[1, 1], M[1, 2]
    m20, m21, m22 = M[2, 0], M[2, 1], M[2, 2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif m00 > m11 and m00 > m22:
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = abs(np.dot(q1, q2))
    dot = max(min(dot, 1.0), -1.0)
    return 2.0 * np.arccos(dot)


# Build orthonormal basis transforms (axis perm + sign flip)
# Allow both det=+1 and det=-1 in the basis, but only keep proper rotations in result
basis = []
for perm in permutations([0, 1, 2]):
    P = np.eye(3)[:, perm]
    for signs in product([1, -1], repeat=3):
        S = np.diag(signs)
        basis.append(P @ S)

best = []
for L in basis:
    for RR in basis:
        for use_T in (False, True):
            M = L @ R @ RR
            if use_T:
                M = M.T
            if np.linalg.det(M) <= 0.0:
                continue
            q = rotmat_to_quat_wxyz(M)
            ang = quat_angle(q, q_target)
            best.append((ang, L, RR, use_T, q))

best.sort(key=lambda x: x[0])

print("Target q (wxyz):", q_target)
print("Top 5 candidates (angle deg):")
for ang, L, RR, use_T, q in best[:5]:
    print("- angle=%.2f deg use_T=%s" % (np.degrees(ang), use_T))
    print("  q=", q)
    print("  L=\n", L)
    print("  RR=\n", RR)

# Also print the current (no transform) for reference
q_current = rotmat_to_quat_wxyz(R)
ang_current = quat_angle(q_current, q_target)
print("\nCurrent R -> q:", q_current)
print("Current angle deg:", np.degrees(ang_current))
