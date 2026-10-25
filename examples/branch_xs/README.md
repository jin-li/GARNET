# Static branch XS examples

Run `python examples/branch_xs/thermal_star.py` from a source checkout.
The manufactured two-node problem builds an explicitly unvalidated five-point
thermal star, evaluates a withheld coordinate, constructs a fresh CPU NEM
operator and checks its eigenvalue against the homogeneous analytical limit.
It uses no reactor files, Serpent installation or nuclear-data libraries.

The synthetic native descriptor identifies manufactured data; it does not
claim a transport calculation. Real data require the evidence and ADF
qualification described in the [User Manual](../../docs/user/branch-xs.md).
