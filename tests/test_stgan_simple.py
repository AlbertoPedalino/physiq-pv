"""Separate loader/inference/logging controls must not alter training updates."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tests'))
from test_stgan_performance import fixture
from physiq_pv.anomaly_detection.stgan import STGAN, STGANCNNConfig, fit_and_score_stgan
from physiq_pv.anomaly_detection.stgan.scoring import score_components, normalize_scores
from physiq_pv.anomaly_detection.stgan.training import DeviceLossTotals
from physiq_pv.anomaly_detection.stgan.loading import make_loader


class SimpleTests(unittest.TestCase):
    def test_configuration_and_cli(self):
        from scripts.run_pvgis_stgan import parse_args
        args=parse_args(['--manifest','input.csv','--out-dir','out','--train-batch-size','256',
            '--score-batch-size','2048','--train-num-workers','4','--score-num-workers','2',
            '--prefetch-factor','3','--log-interval','100'])
        config=STGANCNNConfig(batch_size=args.batch_size,score_batch_size=args.score_batch_size,
            train_num_workers=args.train_num_workers,score_num_workers=args.score_num_workers,
            log_interval=args.log_interval,prefetch_factor=args.prefetch_factor)
        self.assertEqual(config.train_batch_size,256)
        self.assertEqual(config.score_batch_size,2048)
        self.assertEqual((config.train_num_workers,config.score_num_workers),(4,2))
        self.assertIsNone(STGANCNNConfig().score_batch_size)
        for invalid in (dict(train_num_workers=-1),dict(score_num_workers=True),
                        dict(score_batch_size=0),dict(log_interval=0),dict(log_interval=1.5)):
            with self.assertRaises(ValueError):
                STGANCNNConfig(**invalid)

    def test_window_metrics_and_epoch_metrics(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            totals=DeviceLossTotals(device)
            expected=np.zeros(2,np.float64);all_count=0
            for window in range(3):
                subtotal=np.zeros(2,np.float64);count=0
                for index in range(4):
                    g=torch.tensor((window+index)/7,device=device)
                    d=torch.tensor((window-index)/11,device=device)
                    n=7 if index<3 else 2
                    totals.update(g,d,n)
                    subtotal+=np.array([float(g),float(d)])*n;count+=n
                expected+=subtotal;all_count+=count
                # Float64 division/reduction may differ by one ULP on CUDA.
                np.testing.assert_allclose(totals.means_since_last_log(),subtotal/count,atol=1e-15,rtol=1e-15)
                np.testing.assert_allclose(totals.means(),expected/all_count,atol=1e-15,rtol=1e-15)
            with self.assertRaises(ValueError):
                totals.means_since_last_log()

    def test_scoring_batch_sizes_and_partial_batch(self):
        from physiq_pv.anomaly_detection.common import seed_everything
        seed_everything(20)
        model=STGAN(n_features=3,hidden_size=8,n_layers=1,cnn_channels=4,cnn_layers=2).eval()
        baseline=None
        for batch_size in (7,17,64,512):
            g,d,f,store=score_components(model,fixture(),batch_size=batch_size,device='cpu',n_features=3)
            try:
                scores,_,_=normalize_scores(g,d,store,chunk_size=13)
                arrays=[scores,g,d,f]
                if baseline is None:
                    baseline=[np.array(a) for a in arrays]
                else:
                    for a,b in zip(baseline,arrays):
                        np.testing.assert_allclose(a,b,atol=2e-5,rtol=2e-4)
            finally:
                store.close()

    def test_oom_has_actionable_message_and_releases_memmaps(self):
        model=STGAN(n_features=3,hidden_size=8,n_layers=1,cnn_channels=4,cnn_layers=2)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(model,'components',side_effect=torch.cuda.OutOfMemoryError('simulated')):
                with self.assertRaisesRegex(torch.cuda.OutOfMemoryError,'score_batch_size=4096.*Reduce --score-batch-size'):
                    score_components(model,fixture(),batch_size=4096,device='cpu',n_features=3,
                        storage='memmap',output_dir=tmp)
            # Windows fails to unlink an open mapping; the files must be released.
            for path in Path(tmp).glob('*.npy'):
                path.unlink()

    def test_pipeline_loader_overrides_logging_and_training_weights(self):
        from pyproj import Transformer
        times=pd.date_range('2018-12-31 12:00',periods=16,freq='h')
        x,y=np.meshgrid(400000.+np.arange(3)*5000,5000000.-np.arange(3)*5000)
        lon,lat=Transformer.from_crs(32632,4326,always_xy=True).transform(x.ravel(),y.ravel())
        data=fixture().data
        kwargs=dict(train_timestamps=times[:12],test_timestamps=times[12:],
            location_names=tuple(map(str,range(9))),feature_names=('solar','temp','wind'),
            latitudes=lat,longitudes=lon,epochs=2,batch_size=17,hidden_size=8,n_layers=1,
            cnn_channels=4,cnn_layers=2,recent_steps=3,trend_steps=4,device='cpu')
        baseline=None;weights=None
        with tempfile.TemporaryDirectory() as tmp:
            for name,controls in [('default',{}),('independent',dict(train_num_workers=2,
                    score_num_workers=0,score_batch_size=23,log_interval=2))]:
                path=Path(tmp)/name/'model.pt';stream=io.StringIO()
                with patch('physiq_pv.anomaly_detection.stgan.pipeline.make_loader',wraps=make_loader) as train_loader, \
                     patch('physiq_pv.anomaly_detection.stgan.scoring.make_loader',wraps=make_loader) as score_loader, \
                     contextlib.redirect_stdout(stream):
                    result=fit_and_score_stgan(data[:12],data[12:16],**kwargs,**controls,checkpoint_path=path)
                try:
                    current=torch.load(path,weights_only=False)['model_state_dict']
                    arrays=[result.test_scores,result.test_generator_scores,result.test_discriminator_scores,result.test_feature_scores]
                    if baseline is None:
                        baseline=[np.array(a) for a in arrays];weights=current
                        self.assertEqual(stream.getvalue().count('D_mean='),2)
                    else:
                        self.assertTrue(all(torch.equal(weights[k],v) for k,v in current.items()))
                        self.assertEqual(train_loader.call_args.kwargs['batch_size'],17)
                        self.assertEqual(train_loader.call_args.kwargs['num_workers'],2)
                        self.assertEqual(score_loader.call_args.kwargs['batch_size'],23)
                        self.assertEqual(score_loader.call_args.kwargs['num_workers'],0)
                        self.assertEqual(stream.getvalue().count('D_mean='),6)
                        for a,b in zip(baseline,arrays):
                            np.testing.assert_allclose(a,b,atol=2e-5,rtol=2e-4)
                finally:
                    result.close()
                    # Mock call histories retain input datasets/memmaps on Windows.
                    train_loader.reset_mock();score_loader.reset_mock()


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
