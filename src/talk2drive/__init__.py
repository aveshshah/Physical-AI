"""Talk-to-Drive: VLM-guided, energy-aware NMPC driving in NVIDIA Isaac Sim.

The core (robot model, references, controllers, runner) has no Isaac Sim dependency, so it can be
developed and tested anywhere. ``talk2drive.isaac`` contains the Isaac Sim adapters.
"""

from .references import TRAJECTORIES, Trajectory, circle, figure_eight, make_trajectory, slalom
from .robot import AckermannParams, DiffDriveParams, KinematicBicycle, KinematicDiffDrive
from .runner import compute_metrics, plot_runs, run_tracking, save_run

__all__ = [
    "AckermannParams",
    "KinematicBicycle",
    "DiffDriveParams",
    "KinematicDiffDrive",
    "Trajectory",
    "TRAJECTORIES",
    "make_trajectory",
    "figure_eight",
    "slalom",
    "circle",
    "run_tracking",
    "compute_metrics",
    "save_run",
    "plot_runs",
]
