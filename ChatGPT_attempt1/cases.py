"""Teaching cases, in SI units. No Streamlit dependency."""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Fluid:
    label: str
    rho: float = 1000.0
    mu: float = 0.1
    H: float = 0.005
    U: float = 0.02

    @property
    def nu(self):
        return self.mu / self.rho

    @property
    def characteristic_time(self):
        return self.H * self.H / self.nu


BASELINE = Fluid("Baseline")
EXPERIMENTS = {
    "Double density": {
        "fluid": replace(BASELINE, label="Twice the density", rho=2000.0),
        "answer": "Slower",
        "explanation": "Doubling density at fixed dynamic viscosity halves kinematic viscosity. The fluid has more inertia per volume, so the same viscous coupling changes its velocity more slowly. Settling takes twice as long.",
    },
    "Double dynamic viscosity": {
        "fluid": replace(BASELINE, label="Twice the dynamic viscosity", mu=0.2),
        "answer": "Faster",
        "explanation": "Doubling dynamic viscosity at fixed density doubles kinematic viscosity. Stronger viscous coupling transfers motion between layers faster. Settling takes half as long, even though maintaining a given velocity gradient requires more shear stress.",
    },
    "Double both density and viscosity": {
        "fluid": replace(BASELINE, label="Twice both properties", rho=2000.0, mu=0.2),
        "answer": "Same rate",
        "explanation": "The ratio μ/ρ stays the same. With identical plate motion and gap, the velocity histories coincide. Greater dynamic viscosity still requires greater shear stress for the same velocity gradient; kinematic viscosity does not replace dynamic viscosity.",
    },
    "Double the gap": {
        "fluid": replace(BASELINE, label="Twice the gap", H=0.010),
        "answer": "Slower",
        "explanation": "The characteristic time is H²/ν. Doubling the gap multiplies that time by four. Compare relative progress toward each fluid's own steady profile: the wider gap also changes the final velocity gradient.",
    },
}


def shared_duration(fluids):
    return 0.6 * max(fluid.characteristic_time for fluid in fluids)
