# Serpent assembly point-reactor check

Place the fixed-state `*_res.m` files here. The supplied local files cover
2, 4, 7, 14 and 23 energy groups, universe `0`, Serpent 2.2.0, and zero leakage.
Raw Serpent files are kept local and unchanged; selected validation reports
are tracked separately.

From the repository root, use the existing Nix and project-local uv environment:

```sh
nix develop path:. --command uv sync --locked --extra test
nix develop path:. --command .venv/bin/python examples/serpent-xs/point-reactor/run.py

# halo AMD validation; uses the ROCm shell and CuPy dependencies already
# documented in the repository README
nix develop path:.#rocm --command uv sync --locked --extra test --extra rocm
nix develop path:.#rocm --command .venv/bin/python examples/serpent-xs/point-reactor/run.py --device gpu --require-rocm
```

The GPU command runs every CPU reference first, then actual GPU solves with
fusion off and on. It exits unsuccessfully on unavailable GPU, malformed input,
finite leakage, nonconvergence, or failed numerical checks. For another input
directory or report location, set `--input-dir` and `--output-dir`.

Reports default to ignored `results/serpent-point-reactor/`: a batch
`summary.json`, one JSON per input, and one CSV per input with normalized
group spectra. Reports retain input SHA256 hashes and basenames, avoiding host
paths. The reviewed halo evidence is in
[benchmark-results/serpent-point-reactor-halo](../../../benchmark-results/serpent-point-reactor-halo).

Run the self-contained parser/physics tests and local dataset checks with:

```sh
nix develop path:. --command .venv/bin/python -m pytest -q tests/verification/test_serpent_xs.py
```

Local dataset tests skip explicitly when their files are absent. See the
[interface documentation](../../../docs/serpent_xs.md) for the mapping,
tolerances, uncertainty conventions and scope limits. This is an infinite
homogeneous multigroup eigenproblem, with a reflective one-cell diffusion
representation; it validates k and normalized spectrum rather than spatial
assembly flux, leakage or point kinetics.
