"""O15: adaptive BDF on the exact-reference ramp of test_kinetics_reference.
python ramp_record.py <code root> <tests dir> <label> <out.jsonl>"""
import json, sys, time
root, tests, label, out = sys.argv[1:5]
sys.path.insert(0, root); sys.path.insert(1, tests)
import numpy as np
import garnet
assert garnet.__file__.startswith(root), garnet.__file__
from test_kinetics_reference import _solve, _ramp_problem, RAMP_REFERENCE
variants = [("bdf5", True)] if label != "after" else [("bdf1", False), ("bdf2", False), ("bdf3", False), ("bdf4", False), ("bdf5", False), ("bdf5", True)]
for rtol in (1e-3, 1e-4, 1e-5, 1e-7):
    for scheme, auto in variants:
        t0 = time.perf_counter()
        r = _solve(_ramp_problem(), dt=1e-4, time_scheme=scheme, step_solver="monolithic",
                   multigroup_kwargs={"rtol": 1e-13}, adaptive_bdf=dict(rtol=rtol, automatic_order=auto))
        rec = dict(code=label, scheme=scheme, automatic_order=auto, rtol=rtol, steps=len(r.times) - 1,
                   rejected=int(r.rejected_steps), orders=list(map(int, r.time_orders)),
                   final_power_error=float(abs(r.power[-1] / RAMP_REFERENCE - 1)), seconds=time.perf_counter() - t0)
        open(out, "a").write(json.dumps(rec) + "\n")
        print(label, scheme, "auto" if auto else "fixed", rtol, rec["steps"], rec["rejected"], f"{rec['final_power_error']:.2e}", flush=True)
