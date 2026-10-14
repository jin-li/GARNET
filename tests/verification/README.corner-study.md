The corner control in `test_mesh_corner_order.py` compares the existing CPU
`MeshDiffusionEigenSolver` with an independent conforming triangular P1 FEM.
Both solve a homogeneous one-group problem with D=1 cm, absorption=0.02 /cm,
production=0.04 /cm and lateral Robin coefficient gamma=0.5. The FV geometry
has one 0.123 cm z cell with reflecting top and bottom. The planar rectangle
is 12×12 cm; the L shape removes its upper-right 6×6 cm quadrant, leaving
108 cm² and one 270° reentrant corner. Both perimeters are 48 cm.

The inspected FV boundary term is S*gamma*D/(distance*gamma+D), corresponding
to D*normal_derivative(u)+gamma*u=0. The independent rectangle reference uses
q*tan(q*L/2)=gamma/D and k=production/[absorption+D*(qx²+qy²)]. The FEM uses
consistent triangle mass and exact Robin boundary edge integrals. Tests check
area, perimeter, volume and boundary x² integrals, gradient energy, reflecting
limits, analytic rectangle convergence, and independent L-shape refinement.

Run from this worktree using the existing parent scientific environment:

```sh
nix develop /home/lijin/Documents/GitHub/NDgpu --offline --command env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=. /home/lijin/Documents/GitHub/NDgpu/.venv/bin/python -m pytest -q tests/verification/test_mesh_corner_order.py
```

The module CLI emits immutable JSON to an explicitly chosen new study path:

```sh
nix develop /home/lijin/Documents/GitHub/NDgpu --offline --command env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=. /home/lijin/Documents/GitHub/NDgpu/.venv/bin/python tests/verification/test_mesh_corner_order.py --fem-max 256 --output /home/lijin/Documents/GitHub/NDgpu/.scratch/isonees2027/results/verification/corner-control-01/final-control.json
```

FV levels are 8/16/32/64 cells per full square side. FEM levels are
16/32/64/128, optionally 256 when reference refinement uncertainty warrants it.
The JSON records all eigenvalues, k values, normalized eigenpair residuals,
observed orders, source/lockfile/interpreter hashes, software versions and thread
counts. It rejects source changes during the calculation and existing output
paths. Keep generated results outside the software repository.

For the L shape the finest FEM k is a finite-resolution reference. Its last
refinement change is an uncertainty diagnostic, not a rigorous remaining-error
bound; orders against it may shift when the reference is refined. Successive
FV differences also give an independent order diagnostic. Do not fit a
continuum limit or infer the cause of heterogeneous core convergence from this
homogeneous geometry control. This test does not reproduce core materials,
interfaces, reflector prescriptions or eigenvector singular amplitudes.
