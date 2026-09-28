import numpy as np


def sample_random_rotation_matrix():
    """
    Returns a random 3x3 rotation matrix (uniformly sampled from SO(3)).
    """
    # Sample a random quaternion and convert it to a rotation matrix
    q = np.random.randn(4)
    q /= np.linalg.norm(q)

    q1, q2, q3, q4 = q

    # Quaternion to rotation matrix
    R = np.array(
        [
            [1 - 2 * (q3**2 + q4**2), 2 * (q2 * q3 - q1 * q4), 2 * (q2 * q4 + q1 * q3)],
            [2 * (q2 * q3 + q1 * q4), 1 - 2 * (q2**2 + q4**2), 2 * (q3 * q4 - q1 * q2)],
            [2 * (q2 * q4 - q1 * q3), 2 * (q3 * q4 + q1 * q2), 1 - 2 * (q2**2 + q3**2)],
        ],
        dtype=np.float32,
    )

    return R
