#!/usr/bin/env python3
"""
Solver x integrator test in Newton.  ONE scene: a stack of N cubes standing on the floor.
Question: how many solver iterations does each (solver, integrator) combination need to keep the stack standing?

Run once per combination:
    python newton_stack.py --solver newton --integrator euler
    python newton_stack.py --solver newton --integrator implicitfast
    python newton_stack.py --solver cg     --integrator euler
    python newton_stack.py --solver cg     --integrator implicitfast
Each run writes to ./results/ :
    <combo>_sweep.csv        one row per (N cubes, iteration cap): did it stand, how far the top cube sagged
    <combo>_timeseries.csv   top-cube sag over time: the window run (N=--n, cap=--iters) and the same stack with cap 30
and opens a window with that run.  Add --record to save it as <combo>_n<N>_cap<iters>.mp4 (needs: pip install imageio imageio-ffmpeg).   --no-viewer = data only,  --mjc-cpu = run on the CPU.

Docs: https://newton-physics.github.io/newton/stable/api/_generated/newton.solvers.SolverMuJoCo.html
"""
import argparse, csv, os, sys, tempfile, time
import numpy as np
import warp as wp
import newton
import newton.examples

HALF, MU, DT = 0.05, 0.8, 0.005            # cube half-size [m], friction, timestep [s]
NS = [3, 5, 7, 10, 15]                            # stack heights to test
ITERS = [1, 2, 3, 5, 8, 12, 20, 30, 50]        # solver iteration caps to test
T = 3.0                                     # seconds simulated per run


def build_stack(n):
    b = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(density=1.0 / (2 * HALF) ** 3, mu=MU)       # 1 kg per cube
    for i in range(n):
        body = b.add_body(xform=wp.transform(p=wp.vec3(0, 0, HALF + 2 * HALF * i + 0.0005 * i), q=wp.quat_identity()))
        b.add_shape_box(body, hx=HALF, hy=HALF, hz=HALF, cfg=cfg)
    b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=MU))
    return b.finalize(device=DEVICE)


def make_solver(model, iters):
    # the two arguments this experiment is about: solver= and integrator=
    return newton.solvers.SolverMuJoCo(model, solver=ARGS.solver, integrator=ARGS.integrator, iterations=iters,
                                       njmax=2048, nconmax=512, use_mujoco_cpu=ARGS.mjc_cpu)


def quiet():
    """Silence the 'iteration limit reached' warnings (expected: we cap the iterations on purpose)."""
    class Q:
        def __enter__(s):
            sys.stdout.flush(); s.f = tempfile.TemporaryFile(); s.o = os.dup(1), os.dup(2)
            os.dup2(s.f.fileno(), 1); os.dup2(s.f.fileno(), 2); return s
        def __exit__(s, *a):
            if DEVICE.is_cuda: wp.synchronize_device(DEVICE)
            sys.stdout.flush(); os.dup2(s.o[0], 1); os.dup2(s.o[1], 2)
            s.f.seek(0); s.text = s.f.read().decode(errors="replace"); s.f.close()
    return Q()


def simulate(n, iters):
    """Run one stack; return (top-cube sag [mm], max sideways slide [mm], finite?, 'nefc overflow' printed?)."""
    model = build_stack(n)
    solver = make_solver(model, iters)
    s0, s1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    q0 = s0.body_q.numpy()
    z0, sag, slide, finite = q0[-1, 2], 0.0, 0.0, True
    with quiet() as cap:
        for k in range(int(T / DT)):
            s0.clear_forces()
            solver.step(s0, s1, control, None, DT)
            s0, s1 = s1, s0
            q = s0.body_q.numpy()
            if not np.all(np.isfinite(q)):
                finite = False
                break
            sag = z0 - q[-1, 2]
            slide = max(slide, np.abs(q[:, :2] - q0[:, :2]).max())
    return 1e3 * sag, 1e3 * slide, finite, "nefc overflow" in cap.text


def main():
    combo = f"{ARGS.solver}_{ARGS.integrator}"
    os.makedirs(ARGS.out, exist_ok=True)
    print(f"{combo}   device={DEVICE}")

    rows = []
    for n in NS:
        line = ""
        for it in ITERS:
            sag, slide, finite, overflow = simulate(n, it)
            stands = finite and abs(sag) < 20 and slide < 20            # < 2 cm sag and < 2 cm sideways slide
            rows.append(dict(solver=ARGS.solver, integrator=ARGS.integrator, n_cubes=n, iterations=it,
                             stands=stands, top_sag_mm=sag if finite else "", max_slide_mm=slide if finite else "",
                             nefc_overflow=overflow))
            line += ("  +" if stands else "  .") + ("!" if overflow else " ")
        print(f"  N={n}  iterations {ITERS}\n        {line}      (+ stands, . fell)")
    with open(f"{ARGS.out}/{combo}_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    if any(r["nefc_overflow"] for r in rows):
        print("  '!' = constraint buffer overflowed: those runs are INVALID")

    # --- the window run: one stack, real time, logged every step -------------------------------------------
    viewer = None if ARGS.no_viewer else newton.examples.init()[0]
    model = build_stack(ARGS.n)
    solver = make_solver(model, ARGS.iters)
    s0, s1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    if viewer:
        viewer.set_model(model)
    z0 = s0.body_q.numpy()[-1, 2]
    ts, t_wall = [], time.time()
    writer = None
    if viewer and ARGS.record:                    # save a video file: every 3rd step is one frame, 3*DT seconds apart
        import imageio.v2 as imageio              # pip install imageio imageio-ffmpeg
        video = f"{ARGS.out}/{combo}_n{ARGS.n}_cap{ARGS.iters}.mp4"
        writer = imageio.get_writer(video, fps=round(ARGS.speed / (3 * DT)), macro_block_size=1)
    for k in range(int(T / DT)):
        with quiet():
            s0.clear_forces(); solver.step(s0, s1, control, None, DT); s0, s1 = s1, s0
        ts.append(dict(solver=ARGS.solver, integrator=ARGS.integrator, n_cubes=ARGS.n, iterations=ARGS.iters, t=(k + 1) * DT,
                       top_sag_mm=1e3 * (z0 - float(s0.body_q.numpy()[-1, 2]))))
        if viewer and k % 3 == 0:
            viewer.begin_frame((k + 1) * DT); viewer.log_state(s0); viewer.end_frame()
            if writer:
                frame = viewer.get_frame().numpy()          # RGB pixels of the window, shape (height, width, 3)
                frame = frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]   # the mp4 encoder needs even sizes
                writer.append_data(frame)
            else:
                time.sleep(max(0.0, (k + 1) * DT / ARGS.speed - (time.time() - t_wall)))
    if writer:
        for _ in range(int(0.5 * ARGS.speed / (3 * DT)) + 1):   # hold the last frame for half a second
            writer.append_data(frame)
        writer.close()
        print(f"saved video {video}")
    # second timeseries, headless, with plenty of iterations: the fair comparison (all four combinations should stand)
    m2 = build_stack(ARGS.n)
    sv2 = make_solver(m2, 30)
    a0, a1, c2 = m2.state(), m2.state(), m2.control()
    newton.eval_fk(m2, m2.joint_q, m2.joint_qd, a0)
    z2 = a0.body_q.numpy()[-1, 2]
    for k in range(int(T / DT)):
        with quiet():
            a0.clear_forces(); sv2.step(a0, a1, c2, None, DT); a0, a1 = a1, a0
        ts.append(dict(solver=ARGS.solver, integrator=ARGS.integrator, n_cubes=ARGS.n, iterations=30, t=(k + 1) * DT,
                       top_sag_mm=1e3 * (z2 - float(a0.body_q.numpy()[-1, 2]))))
    with open(f"{ARGS.out}/{combo}_timeseries.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ts[0].keys()); w.writeheader(); w.writerows(ts)
    print(f"wrote {ARGS.out}/{combo}_sweep.csv and {combo}_timeseries.csv")
    while viewer and viewer.is_running():         # keep the last frame on screen until you close the window
        viewer.begin_frame(T); viewer.log_state(s0); viewer.end_frame(); time.sleep(1 / 30)


if __name__ == "__main__":
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--solver", choices=["newton", "cg"], required=True)
    p.add_argument("--integrator", choices=["euler", "implicitfast"], required=True)
    p.add_argument("--out", default="results")
    p.add_argument("--n", type=int, default=5, help="cubes in the window run")
    p.add_argument("--iters", type=int, default=2, help="iteration cap in the window run")
    p.add_argument("--speed", type=float, default=1.0, help="window playback speed (0.25 = slow motion)")
    p.add_argument("--record", action="store_true", help="save the window run as an .mp4 in --out")
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--mjc-cpu", action="store_true")
    ARGS, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest            # hand Newton's own parser only ITS flags (it accepts abbreviations like --n)
    DEVICE = wp.get_device("cpu") if ARGS.mjc_cpu else wp.get_device()
    main()