from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .protocols import Controller, Plant, Scenario
from .types import Array, IOSpec, SimConfig, Trajectory, as_vector


@dataclass(frozen=True)
class SimResult:
    traj: Trajectory
    io: IOSpec


def simulate_closed_loop(
    *,
    plant: Plant,
    controller: Controller,
    scenario: Scenario,
    cfg: SimConfig,
) -> SimResult:
    """
    Generic closed-loop simulation.
    This function should NOT be edited when you add new plants/controllers.
    """

    # ---- validate I/O compatibility ----
    if plant.io.ref_dim != controller.io.ref_dim:
        raise ValueError(f"ref_dim mismatch: plant {plant.io.ref_dim} vs controller {controller.io.ref_dim}")
    if plant.io.obs_dim != controller.io.obs_dim:
        raise ValueError(f"obs_dim mismatch: plant {plant.io.obs_dim} vs controller {controller.io.obs_dim}")
    if plant.io.act_dim != controller.io.act_dim:
        raise ValueError(f"act_dim mismatch: plant {plant.io.act_dim} vs controller {controller.io.act_dim}")

    dt = float(plant.dt)
    if dt <= 0:
        raise ValueError(f"Invalid plant.dt={dt}")

    # steps
    if cfg.max_steps is not None:
        n_steps = int(cfg.max_steps)
    else:
        n_steps = int(np.floor(cfg.t_final / dt)) + 1
    if n_steps <= 1:
        raise ValueError("Simulation too short: increase t_final or max_steps")

    # seed
    seed = cfg.seed
    if seed is not None:
        np.random.seed(seed)

    # ---- reset everything ----
    plant.reset(seed=seed)
    controller.reset()
    scenario.reset(seed=seed)

    traj = Trajectory.empty(n_steps=n_steps, io=plant.io)

    # ---- loop ----
    for k in range(n_steps):
        t = k * dt

        # reference and current signals
        r = as_vector(scenario.reference(t), plant.io.ref_dim, "reference r")
        y = as_vector(plant.output(), plant.io.out_dim, "plant output y")

        # measurement -> observation
        obs = as_vector(scenario.measurement(y, t), plant.io.obs_dim, "observation obs")

        # controller proposes action
        u_raw = as_vector(controller.step(r=r, obs=obs, t=t), plant.io.act_dim, "control u_raw")

        # additive input disturbance
        du = as_vector(scenario.disturbance_u(t), plant.io.act_dim, "disturbance_u du")
        u_disturbed = u_raw + du

        # saturation/constraints
        u = as_vector(scenario.saturate(u_disturbed, plant.u_bounds), plant.io.act_dim, "control u")

        # log
        traj.t[k] = t
        traj.r[k] = r
        traj.y[k] = y
        traj.obs[k] = obs
        traj.u_raw[k] = u_raw
        traj.u[k] = u

        # advance plant
        plant.step(u)

    traj.info["seed"] = seed
    traj.info["dt"] = dt
    traj.info["t_final"] = cfg.t_final

    return SimResult(traj=traj, io=plant.io)