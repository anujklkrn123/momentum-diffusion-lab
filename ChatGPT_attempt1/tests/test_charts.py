"""Physics-oriented checks for comparison scales and diagram geometry."""
import itertools
import unittest

import numpy as np

from cases import BASELINE, EXPERIMENTS, Fluid, shared_duration
from charts import arrows_figure, cylinder_diagram, profile_figure
from physics import _solve_couette_with_metadata


class ChartTests(unittest.TestCase):
    def make_trajectories(self, fluids):
        times = np.array([0.0, shared_duration(fluids)])
        return [
            _solve_couette_with_metadata(f.rho, f.mu, f.H, f.U, times)
            for f in fluids
        ]

    def test_doubled_gap_uses_common_vertical_extent(self):
        fluids = [BASELINE, EXPERIMENTS["Double the gap"]["fluid"]]
        trajectories = self.make_trajectories(fluids)
        arrows = arrows_figure(fluids, trajectories, 1)
        self.assertEqual(arrows.layout.yaxis.range, arrows.layout.yaxis2.range)
        self.assertEqual(arrows.layout.xaxis.range, arrows.layout.xaxis2.range)
        self.assertGreater(arrows.layout.yaxis.range[1], 10.0)
        profile = profile_figure(fluids, trajectories, 1)
        self.assertGreater(profile.layout.yaxis.range[1], 10.0)
        self.assertEqual(len(profile.data), 4)
        self.assertEqual([trace.line.dash for trace in profile.data], [None, "dash", None, "dash"])
        self.assertAlmostEqual(max(profile.data[0].y), 5.0)
        self.assertAlmostEqual(max(profile.data[2].y), 10.0)

    def test_velocity_arrows_stay_horizontal_at_fixed_layers(self):
        fluids = [BASELINE, EXPERIMENTS["Double density"]["fluid"]]
        trajectories = self.make_trajectories(fluids)
        for frame in (0, 1):
            with self.subTest(frame=frame):
                chart = arrows_figure(fluids, trajectories, frame)
                velocity_arrows = [a for a in chart.layout.annotations if a.showarrow]
                self.assertTrue(velocity_arrows)
                for arrow in velocity_arrows:
                    self.assertEqual(arrow.y, arrow.ay)
                    self.assertGreaterEqual(arrow.x, arrow.ax)
                    self.assertEqual(arrow.yref, arrow.ayref)
                    self.assertEqual(arrow.xref, arrow.axref)

    def test_cylinder_diagram_has_equal_geometric_axis_scaling(self):
        chart = cylinder_diagram()
        self.assertEqual(chart.layout.yaxis.scaleanchor, "x")
        self.assertEqual(chart.layout.yaxis.scaleratio, 1)
        circles = [shape for shape in chart.layout.shapes if shape.type == "circle"]
        self.assertEqual(len(circles), 2)
        for circle in circles:
            self.assertAlmostEqual(circle.x1 - circle.x0, circle.y1 - circle.y0)

    def test_all_exploration_extremes_remain_conservative(self):
        for rho, mu, gap, speed in itertools.product(
            (500.0, 1500.0), (0.05, 1.0), (0.002, 0.010), (0.005, 0.050)
        ):
            fluid = Fluid("Bounds", rho, mu, gap, speed)
            self.assertLessEqual(fluid.U * fluid.H / fluid.nu, 15.0 + 1e-12)
        for experiment in EXPERIMENTS.values():
            fluid = experiment["fluid"]
            self.assertLessEqual(fluid.U * fluid.H / fluid.nu, 15.0)


if __name__ == "__main__":
    unittest.main()
