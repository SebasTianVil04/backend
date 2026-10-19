import numpy as np

UMBRAL_DOS_MANOS = 0.3


def normalizar_una_mano(secuencia: np.ndarray) -> np.ndarray:
    seq = np.asarray(secuencia, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] != 126:
        return seq

    p0 = np.any(seq[:, :63] != 0, axis=1)
    p1 = np.any(seq[:, 63:] != 0, axis=1)
    con_mano = int((p0 | p1).sum())
    if con_mano == 0:
        return seq

    simultaneos = int((p0 & p1).sum())
    if simultaneos / con_mano > UMBRAL_DOS_MANOS:
        return seq

    dominante = 0 if p0.sum() >= p1.sum() else 1
    resultado = np.zeros_like(seq)

    for i in range(len(seq)):
        if not (p0[i] or p1[i]):
            continue
        if p0[i] and p1[i]:
            usar = dominante
        else:
            usar = 0 if p0[i] else 1
        mano = seq[i, usar * 63:(usar + 1) * 63].reshape(21, 3).copy()
        if usar == 1:
            mano[:, 0] = -mano[:, 0]
        resultado[i, :63] = mano.flatten()

    return resultado