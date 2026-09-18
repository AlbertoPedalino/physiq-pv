"""Synthetic-only ERA5, F=15 STGAN and streaming event graph tests."""
import json
import contextlib
import io
from contextlib import closing
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
import xarray as xr

from physiq_pv.era5.cube import CubeGrid, disk_percentile, morphology, spatial_clusters, score_cube_chunks
from physiq_pv.era5.events import EventConfig, process_events, event_trajectory
from physiq_pv.era5.data import monthly_blocks, prepare_era5
from physiq_pv.era5.features import FEATURE_NAMES, SINGLE, PRESSURE
from physiq_pv.era5.visualization import create_synthetic_demo, plot_frame, plot_event
from physiq_pv.anomaly_detection.stgan import STGAN, STGANWindowDataset, build_spatial_grid, fit_and_score_stgan


def grid(h=9,w=13):
    rows,cols = np.indices((h,w))
    return CubeGrid(45-np.arange(h)*.5,7+np.arange(w)*.5,rows.ravel(),cols.ravel())


class ERA5CubeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def query(self,directory,sql):
        with closing(sqlite3.connect(directory/"events.sqlite")) as db:
            return db.execute(sql).fetchall()

    def test_f15_windows_generator_discriminator_and_score_pipeline(self):
        torch.manual_seed(20)
        layout = grid(3,3)
        lat,lon = layout.latitudes[layout.rows],layout.longitudes[layout.cols]
        spatial = build_spatial_grid(lat,lon,grid_crs="EPSG:4326",angular_spacing=.5,audit_knn=False)
        data = np.random.default_rng(20).normal(size=(64,9,15)).astype(np.float32)
        times = pd.date_range("2004-12-24",periods=64,freq="3h").as_unit("ns")
        dataset = STGANWindowDataset(data,times,spatial,feature_minimum=np.zeros(15),
                                    feature_scale=np.ones(15),recent_steps=1,trend_steps=56,stride=1)
        recent,trend,mask,calendar,observed,*_ = dataset.fetch_batch([0,4,9])
        self.assertEqual(tuple(recent.shape),(3,1,15,3,3))
        self.assertEqual(tuple(trend.shape),(3,56,15))
        self.assertEqual(tuple(observed.shape),(3,15,3,3))
        model = STGAN(n_features=15,hidden_size=4,n_layers=1,cnn_channels=2,cnn_layers=1)
        prediction,real,fake,errors = model.components(recent,trend,mask,calendar,observed)
        self.assertEqual(tuple(prediction.shape),(3,15,3,3))
        self.assertEqual(tuple(real.shape),(3,1))
        self.assertEqual(tuple(fake.shape),(3,1))
        prediction.square().mean().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        # Actual tiny synthetic optimization, not training on ERA5 observations.
        times = pd.date_range("2004-12-24",periods=66,freq="3h").as_unit("ns")
        options=dict(train_timestamps=times[:60],test_timestamps=times[60:],
            location_names=tuple(map(str,range(9))),feature_names=FEATURE_NAMES,
            latitudes=lat,longitudes=lon,epochs=1,batch_size=16,hidden_size=4,n_layers=1,
            cnn_channels=2,cnn_layers=1,trend_steps=56,device="cpu",grid_crs="EPSG:4326",
            timestep_hours=3,angular_grid_spacing=.5,grid_audit_knn=False,
            dataset_name="era5",cache_normalized=False,score_storage="memmap",score_dir=self.root/"scores")
        values = np.random.default_rng(3).normal(size=(66,9,15)).astype(np.float32)
        with fit_and_score_stgan(values[:60],values[60:],**options) as result:
            self.assertEqual(result.test_scores.shape,(6,9))
            self.assertEqual(result.test_feature_scores.shape,(6,9,15))
            self.assertTrue(np.isfinite(result.test_scores).all())
            self.assertEqual(result.metadata["trend_hours"],168)
            g,d = result.test_generator_scores,result.test_discriminator_scores
            expected = (g-g.min())/(g.max()-g.min())+(d-d.min())/(d.max()-d.min())
            np.testing.assert_allclose(result.test_scores,expected,rtol=1e-6,atol=1e-6)
        self.assertTrue((self.root/"scores/test_scores.npy").is_file())

    def test_cube_mapping_permutation_missing_and_chunks(self):
        full = grid(3,4)
        indices = np.array([10,0,5,3,8])
        selected = CubeGrid(full.latitudes,full.longitudes,full.rows[indices],full.cols[indices])
        from_coords = CubeGrid.from_locations(full.latitudes[selected.rows],full.longitudes[selected.cols],
                                             area=[45,7,44,8.5])
        np.testing.assert_array_equal(from_coords.rows,selected.rows)
        scores = np.arange(35,dtype=np.float32).reshape(7,5)
        chunks = list(score_cube_chunks(scores,selected,3))
        cube = np.concatenate([b for _,b in chunks])
        np.testing.assert_array_equal(cube[:,selected.rows,selected.cols],scores)
        self.assertTrue(np.isnan(cube[:,~selected.valid_mask]).all())
        with self.assertRaises(ValueError):
            CubeGrid(full.latitudes,full.longitudes,[0,0],[0,0])

    def test_exact_global_percentile_all_requested_thresholds(self):
        scores = np.random.default_rng(42).normal(size=(37,19)).astype(np.float32)
        scores[0,:3] = np.nan
        for top in (2.5,1,.5,.1):
            actual = disk_percentile(scores,100-top,self.root,3)
            expected = np.percentile(scores[np.isfinite(scores)].astype(float),100-top)
            self.assertAlmostEqual(actual,expected,places=12)
        self.assertEqual(list(self.root.iterdir()),[])

    def test_morphology_holes_noise_nan_and_eight_connectivity(self):
        frame = np.zeros((11,11),np.float32)
        frame[3:8,3:8] = 2
        frame[1,1] = 2
        raw,opened,closed = morphology(frame,1)
        self.assertTrue(raw[1,1])
        self.assertFalse(opened[1,1])
        self.assertEqual(int(opened.sum()),25)
        frame[5,5] = 0
        _,_,closed = morphology(frame,1,opening_iterations=0)
        self.assertTrue(closed[5,5])
        frame[5,5] = np.nan
        _,_,closed = morphology(frame,1,opening_iterations=0)
        self.assertFalse(closed[5,5])
        binary = np.eye(3,dtype=bool)
        _,clusters = spatial_clusters(binary.astype(float),binary,grid(3,3))
        self.assertEqual(len(clusters),1)
        self.assertEqual(clusters[0]["cells"],3)
        self.assertGreater(clusters[0]["area_km2"],0)

    def test_persistence_expansion_contraction_split_merge_chunk_equivalence(self):
        layout = grid()
        frames = np.zeros((5,*layout.shape),np.float32)
        frames[0,3:6,2:5] = 2
        frames[0,3:6,8:11] = 3
        frames[1,3:6,2:11] = 2.5  # merge
        frames[2] = frames[0]  # split
        frames[3,4,3] = 4  # contraction of one branch; other disappears
        frames[4,3:6,2:5] = 2  # expansion
        times = pd.date_range("2005-01-01",periods=5,freq="3h")
        config = EventConfig(absolute_threshold=1,opening_iterations=0,closing_iterations=0,link_policy="overlap")
        results=[]
        for chunk in (1,3,20):
            output=self.root/f"chunk{chunk}"
            metadata=process_events(frames.reshape(5,-1),times,layout,output,replace(config,chunk_size=chunk))
            self.assertEqual(metadata["n_events"],1)
            results.append(self.query(output,"SELECT * FROM clusters ORDER BY cluster_id"))
        self.assertEqual(results[0],results[1])
        self.assertEqual(results[1],results[2])
        relations = {r[0] for r in self.query(self.root/"chunk1","SELECT relation FROM links")}
        self.assertTrue({"split","merge","continue"}<=relations)
        duration = self.query(self.root/"chunk1","SELECT elapsed_hours,duration_hours FROM events")[0]
        self.assertEqual(duration,(12,15))
        trajectory=event_trajectory(self.root/"chunk1",1)
        self.assertEqual(trajectory.cells.tolist(),[18,27,18,1,9])

    def test_motion_centroid_and_temporal_gap(self):
        layout=grid()
        frames=np.zeros((2,*layout.shape),np.float32)
        frames[0,4,4]=2
        frames[1,4,5]=2
        times=pd.date_range("2005-01-01",periods=2,freq="3h")
        base=EventConfig(absolute_threshold=1,opening_iterations=0,closing_iterations=0)
        for policy,expected in (("overlap",2),("dilated_overlap",1),("centroid",1),("any",1)):
            result=process_events(frames.reshape(2,-1),times,layout,self.root/policy,replace(base,link_policy=policy))
            self.assertEqual(result["n_events"],expected)
        result=process_events(frames.reshape(2,-1),times[[0]].append(times[[1]]+pd.Timedelta(hours=3)),
                              layout,self.root/"gap",base)
        self.assertEqual(result["n_events"],2)

    def test_empty_anomalies_and_strict_percentile_ties(self):
        layout=grid(3,3)
        scores=np.ones((2,9),np.float32)
        result=process_events(scores,pd.date_range("2005",periods=2,freq="3h"),layout,self.root/"ties")
        self.assertEqual(result["n_events"],0)
        self.assertEqual(self.query(self.root/"ties","SELECT COUNT(*) FROM clusters"),[(0,)])

    def test_monthly_netcdf_adapter_preserves_f15_and_accumulations(self):
        times=pd.date_range("1980-01-01",periods=31*8,freq="3h").as_unit("ns")
        expected={}
        for pressure,variables in ((False,SINGLE),(True,PRESSURE)):
            prefix="era5_850" if pressure else "era5_single"
            directory=self.root/("pressure_850" if pressure else "single_levels")/"1980"
            directory.mkdir(parents=True)
            coords={"valid_time":times,"latitude":[30.5,30],"longitude":[0,.5]}
            dims=["valid_time","latitude","longitude"]
            shape=(248,2,2)
            if pressure:
                coords["pressure_level"]=[850]
                dims.insert(1,"pressure_level")
                shape=(248,1,2,2)
            data={}
            for name,aliases in variables.items():
                values=np.arange(np.prod(shape),dtype=np.float32).reshape(shape)+FEATURE_NAMES.index(name)*10
                data[aliases[0]]=(dims,values)
                expected[name]=values.reshape(248,2,2)
            xr.Dataset(data,coords=coords).to_netcdf(directory/f"{prefix}_1980_01.nc")
        blocks=list(monthly_blocks(self.root,1980,1,area=(30.5,0,30,.5),chunk_size=17))
        actual=np.concatenate([b for _,b in blocks])
        self.assertEqual(actual.shape,(248,2,2,15))
        for i,name in enumerate(FEATURE_NAMES):
            np.testing.assert_array_equal(actual[...,i],expected[name])

    def test_preparation_split_static_mask_and_reject_transient_missing(self):
        def blocks(root,year,month,**kwargs):
            times=pd.date_range(f"{year}-{month:02d}-01",periods=pd.Period(f"{year}-{month:02d}").days_in_month*8,freq="3h").as_unit("ns")
            values=np.ones((len(times),2,2,15),np.float32)
            values[:,0,0,7]=np.nan
            yield times,values
        with patch("physiq_pv.era5.data.month_files",return_value=[]),patch("physiq_pv.era5.data.monthly_blocks",side_effect=blocks):
            cubes,layout,metadata=prepare_era5(self.root,self.root/"cache",start_year=1980,train_end_year=1980,
                score_end_year=1981,area=(30.5,0,30,.5))
            try:
                self.assertEqual(cubes.train.shape,(366*8,3,15))
                self.assertEqual(cubes.test.shape,(365*8,3,15))
                self.assertEqual(cubes.test_timestamps[0],pd.Timestamp("1981-01-01"))
                self.assertEqual(metadata["n_excluded_cells"],1)
                self.assertFalse(layout.valid_mask[0,0])
            finally:
                cubes.close()
        def changing(*args,**kwargs):
            for times,values in blocks(*args,**kwargs):
                values[1,1,1,0]=np.nan
                yield times,values
        with patch("physiq_pv.era5.data.month_files",return_value=[]),patch("physiq_pv.era5.data.monthly_blocks",side_effect=changing):
            with self.assertRaisesRegex(ValueError,"Time-varying"):
                prepare_era5(self.root,self.root/"invalid",start_year=1980,train_end_year=1980,
                    score_end_year=1981,area=(30.5,0,30,.5))

    def test_visualization_synthetic_demo(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        output=create_synthetic_demo(self.root/"demo")
        fig,clusters=plot_frame(output,4)
        self.assertFalse(clusters.empty)
        fig.savefig(self.root/"frame.png")
        plt.close(fig)
        fig,trajectory=plot_event(output,int(clusters.event_id.iloc[0]))
        self.assertEqual(len(trajectory),12)
        plt.close(fig)

    def test_events_cli_and_frame_threshold(self):
        from scripts.run_era5_stgan import main, parser
        args=parser().parse_args(["prepare","--output-dir","cache"])
        self.assertEqual((args.start_year,args.train_end_year,args.end_year),(1980,2004,"latest"))
        layout=grid(3,3)
        run=self.root/"run"
        run.mkdir()
        np.save(run/"scores.npy",np.arange(27,dtype=np.float32).reshape(3,9))
        np.save(run/"test_timestamps.npy",pd.date_range("2005",periods=3,freq="3h").as_unit("ns").asi8)
        (run/"metadata.json").write_text(json.dumps({"status":"complete","grid":layout.to_dict(),"scores_file":"scores.npy"}))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["events","--run-dir",str(run),"--output-dir",str(self.root/"cli"),
                "--top-percent","2.5","--threshold-scope","frame","--opening-iterations","0",
                "--closing-iterations","0","--chunk-size","1"]),0)
        rows=self.query(self.root/"cli","SELECT threshold FROM frames ORDER BY time_index")
        self.assertEqual(len(rows),3)
        self.assertLess(rows[0][0],rows[1][0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
