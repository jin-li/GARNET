"""O15: LRA-2D adaptive BDF5 automatic order, before/after.
python lra.py <code root> <rtols> <out.jsonl>"""
import json, sys, time
sys.path.insert(0, sys.argv[1])
import numpy as np
import garnet
assert garnet.__file__.startswith(sys.argv[1]), garnet.__file__
from garnet.benchmarks.lra import run_lra2d_cpu
REF = dict(P3=112.79205994447805, P_peak=5973.68656837582, t_peak=1.3022553538349138, T3=1354.0820081467843)
for rtol in [float(x) for x in sys.argv[2].split(",")]:
    t0 = time.perf_counter()
    r = run_lra2d_cpu(refine=1, implicit_feedback=True, thermal_zones="assembly", dt=1e-6, bdf_order=5,
                      adaptive_bdf=dict(rtol=rtol, automatic_order=True, min_dt=1e-7, max_dt=0.1,
                                        rejection_strategy="half", reject_max_factor=0.5))
    sec = time.perf_counter() - t0
    tr = r.transient
    P = np.asarray(r.average_power_w_cm3); i = int(np.argmax(P))
    orders = np.bincount(tr.time_orders, minlength=7)[1:6].tolist()
    o = np.asarray(tr.time_orders)
    rec = dict(code="before" if "before" in sys.argv[1] else "after", rtol=rtol, steps=len(tr.times) - 1,
               rejected=int(tr.rejected_steps), orders=orders, switches=int(np.sum(o[1:] != o[:-1])),
               peak_err=float(P[i] / REF["P_peak"] - 1), t_peak_err=float(tr.times[i] - REF["t_peak"]),
               P3_err=float(P[-1] / REF["P3"] - 1), T3_err=float(r.average_temperature_k[-1] / REF["T3"] - 1),
               seconds=sec)
    open(sys.argv[3], "a").write(json.dumps(rec) + "\n")
    print(f"{rec['code']:6s} rtol {rtol:g}: steps {rec['steps']} rej {rec['rejected']} orders {orders} switches {rec['switches']} "
          f"peak {rec['peak_err']:+.2e} tpk {rec['t_peak_err']:+.1e} P3 {rec['P3_err']:+.1e} T3 {rec['T3_err']:+.1e} {sec:.1f}s", flush=True)
