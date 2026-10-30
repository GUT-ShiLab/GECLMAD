"""Create DMCFMDA's native Peryton input files from the audited archive."""

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "G" / "data" / "peryton" / "data_peryton.npz"
CANONICAL = ROOT / "baseline" / "peryton_global_positive_5fold_v1" / "fold_assignment_disease_microbe.txt"
TARGET = Path(__file__).resolve().parent / "data" / "Peryton"


def main():
    archive = np.load(SOURCE)
    # The shared archive stores (disease, microbe, label); DMCFMDA natively
    # expects (microbe, disease, label).
    associations = archive["adj"][:, [1, 0, 2]]
    canonical = np.loadtxt(CANONICAL, dtype=np.int64)
    if not np.array_equal(canonical[:, :2], archive["adj"][:, :2] + 1):
        raise ValueError("The canonical fold file does not match the Peryton association ordering.")
    fold_ids = canonical[:, 2]
    positive = associations[:, 2] == 1
    TARGET.mkdir(parents=True, exist_ok=True)
    np.savetxt(TARGET / "km.txt", archive["microbefeat"], delimiter=",")
    np.savetxt(TARGET / "ID.txt", archive["diseasefeat"])
    np.savetxt(TARGET / "knowinteraction.txt", associations[positive, :2] + 1, fmt="%d")
    np.savetxt(TARGET / "fold_assignments.txt", np.column_stack((associations[:, :2], fold_ids)), fmt="%d")
    np.savetxt(TARGET / "candidate_labels.txt", associations, fmt="%d")
    print(f"Wrote {TARGET}: {archive['microbefeat'].shape[0]} microbes, {archive['diseasefeat'].shape[0]} diseases, {positive.sum()} positives.")


if __name__ == "__main__":
    main()
