"""The Optuna search must start from the sweep, keep its space and options, and stop by its rules."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import optuna

from scripts import optuna_era5_stgan as script

optuna.logging.set_verbosity(optuna.logging.WARNING)
ROOT = Path(__file__).resolve().parents[1]
METRIC = "validation/pca_mmd_rolling_mean"


def sweep_runs(count=12):
    runs = []
    for index in range(count):
        config = {"generator_learning_rate": 2e-4 + 1e-4 * index, "discriminator_lr_ratio": 0.5 + 0.2 * index,
                  "generator_reconstruction_weight": 60.0 + 50 * index,
                  "discriminator_generator_update_ratio": ("1:1", "2:1", "1:2")[index % 3], "seed": 20}
        runs.append({"name": f"sweep-{index}", "id": f"id{index}", "state": "finished", "config": config,
                     "summary": {METRIC: 0.003 + 0.0002 * index}})
    runs[3]["state"] = "failed"
    runs[4]["summary"] = {METRIC: float("nan")}  # finished with no usable objective
    runs[5]["config"] = {**runs[5]["config"], "seed": 21}  # another seed: left out
    runs[6]["state"] = "running"
    return runs


class OptunaSearchTests(unittest.TestCase):
    def setUp(self):
        self.space, self.fixed, self.metric = script.read_sweep_config(ROOT / script.SWEEP_CONFIG)

    def test_space_and_fixed_options_of_the_sweep(self):
        self.assertEqual(sorted(self.space), ["discriminator_generator_update_ratio", "discriminator_lr_ratio",
                                              "generator_learning_rate", "generator_reconstruction_weight"])
        self.assertTrue(self.space["generator_learning_rate"].log)
        self.assertEqual(self.fixed["epochs"], 6)
        self.assertNotIn("seed", self.fixed)
        self.assertEqual(self.metric, METRIC)

    def test_import_of_the_sweep(self):
        study = optuna.create_study(direction="minimize")
        counts = script.import_sweep(study, sweep_runs(), self.space, self.metric, 20)
        self.assertEqual(counts, {"finished": 8, "failed": 2, "skipped": 2})
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        self.assertAlmostEqual(min(t.value for t in complete), 0.003)
        self.assertEqual({t.user_attrs["source"] for t in study.trials}, {"sweep"})
        with self.assertRaises(ValueError):
            script.import_sweep(study, sweep_runs(), self.space, self.metric, 20)  # only once

    def test_stop_rules(self):
        def trial(value, source="optuna", failed=False):
            state = optuna.trial.TrialState.FAIL if failed else optuna.trial.TrialState.COMPLETE
            return optuna.trial.create_trial(params={}, distributions={}, value=None if failed else value,
                                             state=state, user_attrs={"source": source})
        old = [trial(0.0040, "sweep")]
        trials = old + [trial(0.0030), trial(0.0029), trial(0.0050), trial(0.0, failed=True)]
        for number, item in enumerate(trials):
            item.number = number
        self.assertIsNone(script.stop_reason(trials, 10, 4, 0.0004))  # 0.0030 lowered it by 0.001: streak 3
        trials.append(trial(0.0028))  # lowers by 0.0002 only: fourth in a row
        trials[-1].number = len(trials) - 1
        self.assertIn("4 new trials in a row", script.stop_reason(trials, 10, 4, 0.0004))
        self.assertIn("5 new trials", script.stop_reason(trials, 5, 10, 0.0004))

    def test_preview_leaves_the_study_alone(self):
        study = optuna.create_study(direction="minimize")
        script.import_sweep(study, sweep_runs(16), self.space, self.metric, 20)  # 12 finished: the process is fitted
        before = len(study.trials)
        proposal, beliefs = script.preview(study.trials, self.space, 0)
        self.assertEqual(len(study.trials), before)
        self.assertEqual(sorted(proposal), sorted(self.space))
        self.assertEqual(script.preview(study.trials, self.space, 0)[0], proposal)  # the same proposal again
        self.assertEqual(sorted(beliefs["length_scales"]), sorted(self.space))
        self.assertGreater(beliefs["noise_sd"], 0)
        self.assertEqual(beliefs["best"][:2], ("sweep-0", 0.003))
        few = optuna.create_study(direction="minimize")
        script.import_sweep(few, sweep_runs(), self.space, self.metric, 20)  # 8 finished: still drawn at random
        self.assertIsNone(script.preview(few.trials, self.space, 0)[1])

    def test_search_proposes_trains_and_stops(self):
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.GPSampler(seed=0, deterministic_objective=False))
        script.import_sweep(study, sweep_runs(), self.space, self.metric, 20)
        stale = study.ask(fixed_distributions=self.space)  # left running by an interrupted search
        stale.set_user_attr("source", "optuna")
        written = []

        def train(config_path, trial_dir, name):
            config = __import__("json").loads(Path(config_path).read_text(encoding="utf-8"))
            written.append((name, config))
            return None if len(written) == 2 else 0.002 + config["generator_learning_rate"]

        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            reason = script.run_search(study, self.space, self.fixed, 20, Path(temp), 3, 10, 0.0004, train)
        self.assertIn("3 new trials", reason)
        self.assertEqual(study.trials[stale.number].state, optuna.trial.TrialState.FAIL)
        self.assertTrue(study.trials[stale.number].user_attrs["interrupted"])  # not one of the 3 new trials
        new = [t for t in study.trials if t.user_attrs.get("source") == "optuna" and not t.user_attrs.get("interrupted")]
        self.assertEqual([t.state.name for t in new], ["COMPLETE", "FAIL", "COMPLETE"])
        name, config = written[0]
        self.assertTrue(name.startswith(f"optuna-{study.study_name}-"))
        self.assertEqual(config["epochs"], 6)  # the fixed options of the sweep
        self.assertIn(config["discriminator_generator_update_ratio"], ("1:1", "2:1", "1:2"))
        self.assertTrue(1e-4 <= config["generator_learning_rate"] <= 3e-3)


if __name__ == "__main__":
    unittest.main()
