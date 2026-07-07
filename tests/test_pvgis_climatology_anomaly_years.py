from __future__ import annotations

import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import pandas as pd

from scripts import run_pvgis_climatology_anomaly as single
from scripts import run_pvgis_climatology_anomaly_years as years
from physiq_pv.data import pvgis_anomaly_scores


class PvgisClimatologyAnomalyYearsTests(unittest.TestCase):
    def test_effective_climatology_end_year_is_past_only(self) -> None:
        self.assertEqual(single.effective_climatology_end_year(2016, 2005, 2023), 2015)
        self.assertEqual(single.effective_climatology_end_year(2019, 2005, 2018), 2018)
        self.assertEqual(single.effective_climatology_end_year(2019, 2005, 2009), 2009)

        with self.assertRaisesRegex(ValueError, "no climatology years available"):
            single.effective_climatology_end_year(2005, 2005, 2023)

    def test_paper_faithful_import_path_exists(self) -> None:
        self.assertEqual(
            pvgis_anomaly_scores.DEFAULT_VARIABLES,
            single.DEFAULT_VARIABLES,
        )

    def test_default_output_names_are_past_only(self) -> None:
        p2016 = years.year_out_dir("outputs", 2016, 2005, 2015, 15, 0.975)
        p2018 = years.year_out_dir("outputs", 2018, 2005, 2017, 15, 0.975)
        agg = years.default_aggregate_dir("outputs", [2016, 2017, 2018], 2005, 2018, 15, 0.975)

        self.assertTrue(
            str(p2016).replace("\\", "/").endswith(
                "pvgis_anomaly_2016_2005_2015_w15_q0975"
            )
        )
        self.assertTrue(
            str(p2018).replace("\\", "/").endswith(
                "pvgis_anomaly_2018_2005_2017_w15_q0975"
            )
        )
        self.assertTrue(
            str(agg).replace("\\", "/").endswith(
                "pvgis_anomaly_train_2016_2018_2005_past_w15_q0975"
            )
        )

    def test_legacy_flags_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            common = [
                "--year",
                "2019",
                "--pvgis-path",
                str(tmp_path / "piedmont_pvgis_2019.nc"),
                "--pvgis-climatology-dir",
                str(tmp_path),
                "--climatology-start-year",
                "2005",
                "--climatology-end-year",
                "2023",
            ]
            argv = [
                "run_pvgis_climatology_anomaly.py",
                *common,
                "--include-target-year-in-climatology",
            ]
            with mock.patch.object(sys, "argv", argv):
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    single.parse_args()

            years_common = [
                "--years",
                "2016,2017",
                "--pvgis-dir",
                str(tmp_path),
                "--climatology-start-year",
                "2005",
                "--climatology-end-year",
                "2023",
            ]
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                years.parse_args([*years_common, "--rolling-past-climatology"])
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                years.parse_args([*years_common, "--include-target-year-in-climatology"])

    def test_generate_years_passes_effective_past_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pvgis_dir = tmp_path / "pvgis"
            pvgis_dir.mkdir()
            for year in (2016, 2017):
                (pvgis_dir / f"piedmont_pvgis_{year}.nc").write_text(
                    "placeholder",
                    encoding="utf-8",
                )

            calls: list[tuple[int, int, str]] = []

            def fake_run_single_year(**kwargs: object) -> None:
                calls.append(
                    (
                        int(kwargs["year"]),
                        int(kwargs["climatology_end_year"]),
                        str(kwargs["out_dir"]),
                    )
                )
                out_dir = Path(str(kwargs["out_dir"]))
                out_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    {"timestamp": [f"{kwargs['year']}-01-01"], "label": ["x"]}
                ).to_csv(out_dir / years.SCORES_FILENAME, index=False)

            args = years.parse_args(
                [
                    "--years",
                    "2016,2017",
                    "--pvgis-dir",
                    str(pvgis_dir),
                    "--climatology-start-year",
                    "2005",
                    "--climatology-end-year",
                    "2023",
                    "--out-root",
                    str(tmp_path / "outputs"),
                    "--no-aggregate",
                ]
            )

            with mock.patch.object(years, "run_single_year", fake_run_single_year):
                with redirect_stdout(StringIO()):
                    scores = years.generate_years(args)

            self.assertEqual(calls[0][0:2], (2016, 2015))
            self.assertEqual(calls[1][0:2], (2017, 2016))
            self.assertIn("pvgis_anomaly_2016_2005_2015", calls[0][2].replace("\\", "/"))
            self.assertIn("pvgis_anomaly_2017_2005_2016", calls[1][2].replace("\\", "/"))
            self.assertEqual(sorted(scores), [2016, 2017])

    def test_aggregate_annual_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            p2016 = tmp_path / "2016.csv"
            p2017 = tmp_path / "2017.csv"
            pd.DataFrame({"timestamp": ["2016-01-01"], "label": ["a"]}).to_csv(
                p2016,
                index=False,
            )
            pd.DataFrame({"timestamp": ["2017-01-01"], "label": ["b"]}).to_csv(
                p2017,
                index=False,
            )

            out_csv = tmp_path / "aggregate" / years.SCORES_FILENAME
            combined = years.aggregate_annual_scores({2017: p2017, 2016: p2016}, out_csv)

            self.assertTrue(out_csv.exists())
            self.assertEqual(combined["source_anomaly_year"].tolist(), [2016, 2017])
            self.assertEqual(combined["label"].tolist(), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
