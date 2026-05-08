from __future__ import annotations

from inverted_pendulum_rl_wind_common import run_rl_wind_demo


if __name__ == "__main__":
    run_rl_wind_demo(
        kind="pidfeat",
        figure_name="closed_loop_linearized_inverted_pendulum_rl_pid_wind.png",
    )
