"""Streamlit integration checks; these do not replace browser layout review."""
import html
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from streamlit.testing.v1 import AppTest


PROJECT = Path(__file__).resolve().parents[1]
PAGES = [
    "Start here", "Predict and observe", "Free exploration", "Physics explained",
    "ChemE connections", "AI process and evidence", "Presentation", "References",
]


class AppIntegrationTests(unittest.TestCase):
    def make_app(self, path=None):
        return AppTest.from_file(str(path or PROJECT / "app.py"), default_timeout=20).run()

    def assert_no_exception(self, app):
        self.assertEqual([item.value for item in app.exception], [])

    def go(self, app, page):
        app.radio(key="page").set_value(page).run()
        self.assert_no_exception(app)

    def test_all_eight_pages_load(self):
        app = self.make_app()
        self.assert_no_exception(app)
        self.assertEqual(app.radio(key="page").options, PAGES)
        for page in PAGES:
            with self.subTest(page=page):
                self.go(app, page)
                self.assertTrue(app.title)
        self.assertTrue(any("reference" in item.value.lower() for item in app.markdown))

    def test_prediction_is_required_and_all_four_cases_render(self):
        answers = {
            "Double density": "Slower",
            "Double dynamic viscosity": "Faster",
            "Double both density and viscosity": "Same rate",
            "Double the gap": "Slower",
        }
        app = self.make_app()
        self.go(app, "Predict and observe")
        for experiment, answer in answers.items():
            with self.subTest(experiment=experiment):
                app.selectbox(key="experiment").set_value(experiment).run()
                self.assert_no_exception(app)
                self.assertEqual(len(app.get("plotly_chart")), 0)
                reveal = next(button for button in app.button if button.label == "Reveal and run experiment")
                self.assertTrue(reveal.disabled)
                app.radio(key=f"prediction_{experiment}").set_value(answer).run()
                reveal = next(button for button in app.button if button.label == "Reveal and run experiment")
                self.assertFalse(reveal.disabled)
                reveal.click().run()
                self.assert_no_exception(app)
                self.assertEqual(len(app.get("plotly_chart")), 2)
                self.assertEqual(len(app.select_slider), 1)
                self.assertTrue(app.success)
                self.assertEqual(app.session_state["time_frame"], 0)
                self.assertFalse(app.session_state["playing"])

    def test_wrong_prediction_explains_without_blocking_demo(self):
        app = self.make_app()
        self.go(app, "Predict and observe")
        app.radio(key="prediction_Double density").set_value("Faster").run()
        next(button for button in app.button if button.label == "Reveal and run experiment").click().run()
        self.assert_no_exception(app)
        self.assertFalse(app.success)
        self.assertTrue(any("slower" in item.value for item in app.info))
        self.assertEqual(len(app.get("plotly_chart")), 2)

    def test_play_pause_scrub_reset_and_end_restart(self):
        app = self.make_app()
        self.go(app, "Free exploration")
        self.assertEqual(len(app.get("plotly_chart")), 2)
        app.button(key="play_pause").click().run()
        self.assert_no_exception(app)
        self.assertTrue(app.session_state["playing"])
        self.assertEqual(app.button(key="play_pause").label, "Pause")
        app.button(key="play_pause").click().run()
        self.assertFalse(app.session_state["playing"])
        app.select_slider(key="time_frame").set_value(60).run()
        self.assert_no_exception(app)
        self.assertEqual(app.session_state["time_frame"], 60)
        self.assertFalse(app.session_state["playing"])
        self.assertTrue(any("Elapsed time:" in item.value and "0 s" not in item.value for item in app.markdown))
        app.button(key="play_pause").click().run()
        app.select_slider(key="time_frame").set_value(100).run()
        self.assertFalse(app.session_state["playing"])
        app.button(key="reset_time").click().run()
        self.assertEqual(app.session_state["time_frame"], 0)
        self.assertFalse(app.session_state["playing"])
        app.select_slider(key="time_frame").set_value(120).run()
        app.button(key="play_pause").click().run()
        self.assert_no_exception(app)
        self.assertEqual(app.session_state["time_frame"], 0)
        self.assertTrue(app.session_state["playing"])

    def test_each_physical_input_change_resets_and_pauses(self):
        app = self.make_app()
        self.go(app, "Free exploration")
        changes = {"density": 1200.0, "viscosity": 0.2, "gap": 8.0, "speed": 0.03}
        for key, value in changes.items():
            with self.subTest(key=key):
                app.select_slider(key="time_frame").set_value(60).run()
                app.button(key="play_pause").click().run()
                app.slider(key=key).set_value(value).run()
                self.assert_no_exception(app)
                self.assertEqual(app.session_state["time_frame"], 0)
                self.assertFalse(app.session_state["playing"])
                self.assertEqual(len(app.get("plotly_chart")), 2)

    def test_clock_advances_without_mutating_live_scrubber_and_stops_at_end(self):
        app = self.make_app()
        self.go(app, "Free exploration")
        app.button(key="play_pause").click().run()
        starting_frame = app.session_state["display_frame"]
        scrubber_frame = app.session_state["time_frame"]
        for increment in range(1, 4):
            # Force an elapsed tick deterministically without sleeping or changing
            # the input widget. The clock advances its separate display state.
            app.session_state["last_tick"] = 0.0
            app.run()
            self.assert_no_exception(app)
            self.assertEqual(app.session_state["display_frame"], starting_frame + increment)
            self.assertEqual(app.session_state["time_frame"], scrubber_frame)
            self.assertTrue(app.session_state["playing"])
        app.button(key="play_pause").click().run()
        self.assertFalse(app.session_state["playing"])
        self.assertEqual(app.session_state["time_frame"], app.session_state["display_frame"])
        app.select_slider(key="time_frame").set_value(30).run()
        self.assertEqual(app.session_state["display_frame"], 30)
        self.assertFalse(app.session_state["playing"])
        app.button(key="play_pause").click().run()
        app.session_state["display_frame"] = 119
        app.session_state["last_tick"] = 0.0
        app.run()
        self.assert_no_exception(app)
        self.assertEqual(app.session_state["display_frame"], 120)
        self.assertEqual(app.session_state["time_frame"], 120)
        self.assertFalse(app.session_state["playing"])
        self.assertFalse(app.session_state["sync_scrubber"])
        self.assertEqual(app.button(key="play_pause").label, "Play")

    def test_case_change_and_navigation_reset_player(self):
        app = self.make_app()
        self.go(app, "Predict and observe")
        app.radio(key="prediction_Double density").set_value("Slower").run()
        next(button for button in app.button if button.label == "Reveal and run experiment").click().run()
        app.select_slider(key="time_frame").set_value(50).run()
        app.button(key="play_pause").click().run()
        app.selectbox(key="experiment").set_value("Double the gap").run()
        self.assert_no_exception(app)
        self.assertEqual(app.session_state["time_frame"], 0)
        self.assertFalse(app.session_state["playing"])
        self.go(app, "Free exploration")
        app.select_slider(key="time_frame").set_value(50).run()
        app.button(key="play_pause").click().run()
        self.go(app, "Physics explained")
        self.assertFalse(app.session_state["playing"])
        self.go(app, "Free exploration")
        self.assertEqual(app.session_state["time_frame"], 0)

    def test_pending_presentation_is_honest(self):
        app = self.make_app()
        self.go(app, "Presentation")
        self.assertTrue(any("Pending" in item.value for item in app.info))
        self.assertTrue(any("Pending" in item.value for item in app.markdown))
        self.assertEqual(len(app.get("video")), 0)

    def test_transcript_fixture_preserves_exact_display_text(self):
        # Use a temporary project copy, so no real evidence or transcript is edited.
        fixture = "\r\n  0:00\r\n  μ is dynamic viscosity.\r\n\r\n0:03\r\n<u> & [literal text]\r\n\r\n"
        with tempfile.TemporaryDirectory() as folder:
            copy = Path(folder) / "project"
            shutil.copytree(PROJECT, copy, ignore=shutil.ignore_patterns("__pycache__", "tests", ".git"))
            transcript_file = copy / "content" / "transcript.txt"
            transcript_file.write_bytes(fixture.encode("utf-8"))
            presentation_file = copy / "content" / "presentation.json"
            config = json.loads(presentation_file.read_text(encoding="utf-8"))
            config["youtube_url"] = "https://www.youtube.com/watch?v=fixture"
            presentation_file.write_text(json.dumps(config), encoding="utf-8")
            app = self.make_app(copy / "app.py")
            self.go(app, "Presentation")
            # Accept literal text or escaped preformatted HTML with no alteration.
            literal_bodies = [item.value for item in app.text]
            html_bodies = [item.proto.body for item in app.get("html")]
            self.assertTrue(
                fixture in literal_bodies or any(html.escape(fixture) in body for body in html_bodies),
                "Transcript display changed supplied whitespace, line breaks, or literal characters.",
            )
            self.assertEqual(transcript_file.read_bytes(), fixture.encode("utf-8"))
            self.assertEqual(len(app.get("download_button")), 1)
            self.assertEqual(len(app.get("video")), 1)


if __name__ == "__main__":
    unittest.main()
