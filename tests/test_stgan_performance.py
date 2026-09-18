"""Equivalence and bounded-memory contracts for STGAN performance paths."""
import copy
import pickle
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from physiq_pv.anomaly_detection.stgan import STGANWindowDataset, build_spatial_grid, STGAN
from physiq_pv.anomaly_detection.stgan.loading import make_loader
from physiq_pv.anomaly_detection.stgan.training import gan_train_step
from physiq_pv.anomaly_detection.stgan.model import masked_cell_mean


def legacy_step(model, batch, go, do, observe):
    """Independent reference for the original alternating loop, including two G calls."""
    recent,trend,mask,calendar,observed=batch
    normal=torch.zeros((len(recent),1),device=recent.device)
    generated_target=torch.ones_like(normal)
    do.zero_grad()
    with torch.no_grad():
        generated=model.generator(recent,trend,mask,calendar)
    real=model.discriminator(torch.cat((recent,observed[:,None]),1),mask)
    fake=model.discriminator(torch.cat((recent,generated[:,None]),1),mask)
    observe('D_forward',dict(generated=generated,real=real,fake=fake))
    dl=.5*(torch.nn.functional.binary_cross_entropy(real,normal)+
            torch.nn.functional.binary_cross_entropy(fake,generated_target))
    dl.backward()
    observe('D_backward',dict(model=model,loss=dl))
    do.step()
    observe('D_step',dict(model=model))
    go.zero_grad()
    for p in model.discriminator.parameters():
        p.requires_grad_(False)
    generated=model.generator(recent,trend,mask,calendar)
    fake=model.discriminator(torch.cat((recent,generated[:,None]),1),mask)
    observe('G_forward',dict(generated=generated,fake=fake))
    errors=torch.where(mask.bool(),generated-observed,0.0).square()
    gl=500*masked_cell_mean(errors,mask).mean()+torch.nn.functional.binary_cross_entropy(fake,normal)
    gl.backward()
    observe('G_backward',dict(model=model,loss=gl))
    go.step()
    observe('G_step',dict(model=model))
    for p in model.discriminator.parameters():
        p.requires_grad_(True)
    return gl.detach(),dl.detach()


def trace_observer(trace, *, assert_detach=False):
    def observe(stage, values):
        for key,value in values.items():
            if key=='model':
                if stage=='D_backward' and assert_detach:
                    assert all(p.grad is None for p in value.generator.parameters())
                prefix='discriminator' if stage.startswith('D') else 'generator'
                for name,p in value.named_parameters():
                    if name.startswith(prefix):
                        if stage.endswith('backward'):
                            trace[stage+'/'+name] = p.grad.detach().clone() if p.grad is not None else None
                        else:
                            trace[stage+'/'+name] = p.detach().clone()
            else:
                trace[stage+'/'+key]=value.detach().clone()
    return observe


def fixture(data=None, *, recent=3, trend=6, patch=3, stride=1):
    from pyproj import Transformer
    x, y = np.meshgrid(400000. + np.arange(3)*5000, 5000000. - np.arange(3)*5000)
    lon, lat = Transformer.from_crs(32632, 4326, always_xy=True).transform(x.ravel(), y.ravel())
    grid = build_spatial_grid(lat, lon, patch_size=patch)
    if data is None:
        data = np.random.default_rng(9).normal(size=(30, 9, 3)).astype(np.float32)
    times = pd.date_range('2018-01-01', periods=len(data), freq='h')
    ds = STGANWindowDataset(data, times, grid, feature_minimum=np.array([-3, -2, -1],np.float32),
        feature_scale=np.array([6, 5, 4],np.float32), recent_steps=recent, trend_steps=trend, stride=stride)
    return ds


class LoaderTests(unittest.TestCase):
    def test_runtime_config_cli_and_cache_overwrite_guard(self):
        from physiq_pv.anomaly_detection.stgan.config import STGANCNNConfig
        from physiq_pv.anomaly_detection.stgan.data import normalized_memmap
        from scripts.run_pvgis_stgan import parse_args
        args=parse_args(['--manifest','input.csv','--out-dir','output','--num-workers','2',
            '--prefetch-factor','3','--no-pin-memory','--no-persistent-workers',
            '--shuffle-mode','block','--score-storage','memmap','--score-chunk-size','17'])
        self.assertEqual((args.num_workers,args.prefetch_factor,args.score_chunk_size),(2,3,17))
        self.assertFalse(args.pin_memory);self.assertFalse(args.persistent_workers)
        for invalid in (dict(num_workers=-1),dict(prefetch_factor=0),dict(score_chunk_size=0),
                        dict(shuffle_mode='invalid'),dict(execution_mode='invalid')):
            with self.assertRaises(ValueError):
                STGANCNNConfig(**invalid)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'source.npy'
            values=fixture().data
            np.save(path,values)
            source=np.load(path,mmap_mode='r')
            try:
                with self.assertRaises(ValueError):
                    normalized_memmap(source,np.zeros(3),np.ones(3),path)
                np.testing.assert_array_equal(source,values)
            finally:
                source._mmap.close()

    def test_vectorized_and_cached_inputs_bitwise(self):
        from physiq_pv.anomaly_detection.stgan.data import normalized_memmap
        for patch in (1,3,5):
            for recent in (1,3):
                ds=fixture(patch=patch,recent=recent,stride=2)
                indices=[len(ds)-1,0,9,1,8,9,27,4]
                expected=next(iter(DataLoader(ds,batch_size=len(indices),sampler=indices)))
                actual=ds.fetch_batch(indices)
                for a,b in zip(expected,actual,strict=True):
                    self.assertTrue(torch.equal(a,b))
                with tempfile.TemporaryDirectory() as tmp:
                    cached=normalized_memmap(ds.data,ds.minimum,ds.scale,Path(tmp)/'normalized.npy',buffer_bytes=500)
                    other=fixture(cached,patch=patch,recent=recent,stride=2)
                    other.normalized=True
                    for a,b in zip(expected,other.fetch_batch(indices),strict=True):
                        self.assertTrue(torch.equal(a,b))
                    del other
                    cached._mmap.close()

    def test_spawn_memmap_and_workers_preserve_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = fixture().data
            path = Path(tmp)/'cube.npy'
            np.save(path, values)
            ds = fixture(np.load(path,mmap_mode='r')[2:])
            payload = pickle.dumps(ds)
            restored = pickle.loads(payload)
            self.assertIsInstance(restored.data, np.memmap)
            np.testing.assert_array_equal(restored.data, values[2:])
            order = np.random.default_rng(3).permutation(len(ds)).tolist()
            expected = list(DataLoader(ds, batch_size=17, sampler=order))
            loader = make_loader(ds,batch_size=17,sampler=order,num_workers=2,
                persistent_workers=True,prefetch_factor=2)
            for _ in range(2):
                for a,b in zip(expected, loader, strict=True):
                    for x,y in zip(a,b,strict=True):
                        self.assertTrue(torch.equal(x,y))
            del loader, restored, ds

    def test_generator_training_forward_is_bitwise_deterministic(self):
        torch.manual_seed(20)
        model=STGAN(n_features=3,hidden_size=8,n_layers=1,cnn_channels=4,cnn_layers=2).train()
        batch=next(iter(DataLoader(fixture(),batch_size=7)))
        state={k:v.clone() for k,v in model.state_dict().items()}
        rng=torch.get_rng_state().clone()
        first=model.generator(*batch[:4])
        second=model.generator(*batch[:4])
        self.assertTrue(torch.equal(first,second))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in state.items()))
        self.assertFalse(any(isinstance(m,torch.nn.modules.dropout._DropoutNd) for m in model.modules()))


class TrainingTests(unittest.TestCase):
    def test_device_metrics_preserve_float64_accumulation(self):
        from physiq_pv.anomaly_detection.stgan.training import DeviceLossTotals
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            totals=DeviceLossTotals(device)
            expected=np.zeros(2,np.float64)
            count=0
            for index in range(17):
                g,d=torch.tensor(index/7,device=device),torch.tensor(index/11,device=device)
                n=7 if index<16 else 3
                totals.update(g,d,n)
                expected+=np.array([float(g),float(d)])*n
                count+=n
            np.testing.assert_array_equal(totals.means(),expected/count)

    def test_nonfinite_losses_stop_before_corresponding_step(self):
        from unittest.mock import patch
        model=STGAN(n_features=3,hidden_size=8,n_layers=1,cnn_channels=4,cnn_layers=2)
        batch=next(iter(DataLoader(fixture(),batch_size=7)))[:5]
        go,do=torch.optim.Adam(model.generator.parameters()),torch.optim.Adam(model.discriminator.parameters())
        before={k:v.clone() for k,v in model.state_dict().items()}
        with patch('physiq_pv.anomaly_detection.stgan.training.binary_cross_entropy',return_value=torch.tensor(float('nan'))):
            with self.assertRaisesRegex(FloatingPointError,'discriminator'):
                gan_train_step(model,batch,go,do)
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
        with self.assertRaisesRegex(FloatingPointError,'generator'):
            gan_train_step(model,batch,go,do,reconstruction_weight=float('inf'))
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items() if k.startswith('generator')))
        self.assertTrue(all(p.requires_grad for p in model.discriminator.parameters()))

    def compare_steps(self, device='cpu', **options):
        from physiq_pv.anomaly_detection.common import seed_everything
        seed_everything(20)
        a=STGAN(n_features=3,hidden_size=8,n_layers=1,cnn_channels=4,cnn_layers=2).to(device).train()
        b=copy.deepcopy(a)
        ag,ad=torch.optim.Adam(a.generator.parameters(),lr=.001),torch.optim.Adam(a.discriminator.parameters(),lr=.001)
        bg,bd=torch.optim.Adam(b.generator.parameters(),lr=.001),torch.optim.Adam(b.discriminator.parameters(),lr=.001)
        batches=list(DataLoader(fixture(),batch_size=7))[:4]
        exact,total,max_error=0,0,0.
        for batch in batches:
            batch=tuple(x.to(device) for x in batch[:5])
            before_a,before_b={},{}
            legacy_step(a,batch,ag,ad,trace_observer(before_a))
            gan_train_step(b,batch,bg,bd,observe=trace_observer(before_b,assert_detach=True),**options)
            self.assertEqual(before_a.keys(),before_b.keys())
            for key,x in before_a.items():
                y=before_b[key]
                if x is None:
                    self.assertIsNone(y)
                    continue
                max_error=max(max_error,float((x-y).abs().max()))
                exact+=int(torch.equal(x,y)); total+=1
                torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-5,msg=key)
        print(f'EQUIVALENCE {device} {options}: bitwise {exact}/{total}, max_abs={max_error:.9g}',flush=True)

    def test_single_generator_cpu(self):
        self.compare_steps(share_history=False)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_single_generator_cuda(self):
        self.compare_steps('cuda',share_history=False)

    def test_shared_history_cpu(self):
        self.compare_steps(share_history=True)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_shared_history_cuda(self):
        self.compare_steps('cuda',share_history=True)

    def test_call_counts_and_frozen_discriminator(self):
        model=STGAN(n_features=3,hidden_size=8,n_layers=1,cnn_channels=4,cnn_layers=2)
        batch=next(iter(DataLoader(fixture(),batch_size=7)))[:5]
        counts={'G':0,'history':0}
        history_grad_enabled=[]
        handles=[]
        for name,module in [('G',model.generator),('history',model.discriminator.sequence_encoder)]:
            def count(m,a,o,name=name):
                counts[name]+=1
                if name=='history':
                    history_grad_enabled.append(o.requires_grad)
            handles.append(module.register_forward_hook(count))
        go,do=torch.optim.Adam(model.generator.parameters()),torch.optim.Adam(model.discriminator.parameters())
        gradients={}
        def observe(stage,values):
            if stage=='D_step':
                gradients.update({n:p.grad.clone() for n,p in model.discriminator.named_parameters()})
            if stage=='G_backward':
                self.assertTrue(all(not p.requires_grad for p in model.discriminator.parameters()))
                self.assertTrue(all(torch.equal(p.grad,gradients[n]) for n,p in model.discriminator.named_parameters()))
        gan_train_step(model,batch,go,do,observe=observe)
        self.assertEqual(counts,{'G':1,'history':2})
        self.assertEqual(history_grad_enabled,[True,False])
        self.assertTrue(all(p.requires_grad for p in model.discriminator.parameters()))
        for h in handles:
            h.remove()

    def test_scoring_shared_history(self):
        model=STGAN(n_features=3,hidden_size=8,n_layers=1,cnn_channels=4,cnn_layers=2).eval()
        batch=next(iter(DataLoader(fixture(),batch_size=7)))[:5]
        with torch.no_grad():
            for a,b in zip(model.components(*batch,share_history=False),model.components(*batch),strict=True):
                self.assertTrue(torch.equal(a,b))


class SamplerTests(unittest.TestCase):
    def test_block_bijection_reproducibility_and_large_length(self):
        from itertools import islice
        from physiq_pv.anomaly_detection.stgan.sampling import EpochShuffleSampler, permuted_block
        for size in (1,2,3,7,17,63,64,65,1234):
            self.assertEqual(sorted(permuted_block(i,size,83) for i in range(size)),list(range(size)))
        a=EpochShuffleSampler(range(503),mode='block',seed=3,block_size=17)
        b=EpochShuffleSampler(range(503),mode='block',seed=3,block_size=17)
        first=list(a)
        self.assertEqual(first,list(b))
        self.assertEqual(sorted(first),list(range(503)))
        second=list(a)
        self.assertNotEqual(first,second)
        self.assertEqual(second,list(b))
        huge=EpochShuffleSampler(range(6_600_000_000),mode='block',block_size=1024)
        indices=list(islice(huge,5000))
        self.assertEqual(len(set(indices)),5000)
        self.assertTrue(all(0<=i<len(huge) for i in indices))
        self.assertLess(len(pickle.dumps(huge)),1000)

    def test_global_matches_original_loader_rng_for_multiple_epochs(self):
        from physiq_pv.anomaly_detection.stgan.sampling import EpochShuffleSampler
        ds=fixture()
        torch.manual_seed(71)
        original=DataLoader(ds,batch_size=17,shuffle=True)
        expected=[[(x[-2].tolist(),x[-1].tolist()) for x in original] for _ in range(3)]
        final_rng=torch.get_rng_state()
        from physiq_pv.anomaly_detection.stgan.loading import close_loader
        for mode,workers in (('global',0),('legacy',0),('global',2)):
            torch.manual_seed(71)
            sampler=EpochShuffleSampler(ds,mode=mode,legacy_rng=True)
            actual=make_loader(ds,batch_size=17,sampler=sampler,num_workers=workers,
                generator=torch.Generator().manual_seed(99))
            self.assertEqual(expected,[[(x[-2].tolist(),x[-1].tolist()) for x in actual] for _ in range(3)])
            self.assertTrue(torch.equal(final_rng,torch.get_rng_state()))
            close_loader(actual)

    def test_block_order_independent_of_workers(self):
        from physiq_pv.anomaly_detection.stgan.sampling import EpochShuffleSampler
        ds=fixture()
        loaders=[make_loader(ds,batch_size=17,num_workers=w,
            sampler=EpochShuffleSampler(ds,mode='block',seed=8,block_size=23)) for w in (0,2)]
        for left,right in zip(*loaders,strict=True):
            for a,b in zip(left,right,strict=True):
                self.assertTrue(torch.equal(a,b))
        del loaders


class StreamingTests(unittest.TestCase):
    def test_context_virtual_matches_materialized_and_spawn_pickle(self):
        from physiq_pv.anomaly_detection.stgan.data import ContextArray
        with tempfile.TemporaryDirectory() as tmp:
            values=fixture().data
            path=Path(tmp)/'data.npy';np.save(path,values)
            source=np.load(path,mmap_mode='r')
            virtual=ContextArray(source[:7],source[7:])
            ds=fixture(virtual)
            restored=pickle.loads(pickle.dumps(ds))
            self.assertIsInstance(restored.data.prefix,np.memmap)
            self.assertIsInstance(restored.data.test,np.memmap)
            indices=[0,8,9,len(ds)-1,17]
            for a,b in zip(fixture(values).fetch_batch(indices),restored.fetch_batch(indices),strict=True):
                self.assertTrue(torch.equal(a,b))
            for i in indices:
                for a,b in zip(fixture(values)[i][:5],ds[i][:5],strict=True):
                    self.assertTrue(torch.equal(a,b))
            del ds,restored,virtual
            source._mmap.close()

    def test_scoring_storage_and_global_normalization_equivalence(self):
        from physiq_pv.anomaly_detection.stgan.scoring import score_components,normalize_scores
        from physiq_pv.anomaly_detection.stgan.pipeline import _component_range,_normalise_component
        model=STGAN(n_features=3,hidden_size=8,n_layers=1,cnn_channels=4,cnn_layers=2).eval()
        ds=fixture()
        baseline=None
        with tempfile.TemporaryDirectory() as tmp:
            for name,storage,share in [('legacy','memory',False),('optimized','memory',True),('streaming','memmap',True)]:
                g,d,f,store=score_components(model,ds,batch_size=17,device='cpu',n_features=3,
                    storage=storage,output_dir=Path(tmp)/name,share_history=share,
                    loader_options=dict(vectorized=share))
                try:
                    scores,gr,dr=normalize_scores(g,d,store,chunk_size=7)
                    expected=_normalise_component(g,_component_range(g))+_normalise_component(d,_component_range(d))
                    np.testing.assert_array_equal(scores,expected)
                    self.assertEqual(gr,_component_range(g));self.assertEqual(dr,_component_range(d))
                    actual=[scores,g,d,f]
                    if baseline is None:
                        baseline=[np.array(x) for x in actual]
                    else:
                        for a,b in zip(baseline,actual,strict=True):
                            np.testing.assert_array_equal(a,b)
                    self.assertEqual(isinstance(scores,np.memmap),storage=='memmap')
                finally:
                    store.close()

    def test_external_ranking_ties_nonfinite_and_boundaries(self):
        from physiq_pv.anomaly_detection.stgan.scoring import ScoreStore
        from physiq_pv.anomaly_detection.stgan.ranking import rank_scores,boundary_summaries
        from scripts.run_pvgis_stgan import paper_top_k_ranking
        scores=np.array([[2,1,2],[np.nan,np.inf,-np.inf],[1,-0.,0.]],np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            for storage in ('memory','memmap'):
                store=ScoreStore(root=Path(tmp)/storage,backend=storage)
                try:
                    actual=rank_scores(scores,33,store,chunk_size=2)
                    for a,b in zip(paper_top_k_ranking(scores,33),(actual.flags,actual.ranks,actual.percentiles),strict=True):
                        np.testing.assert_array_equal(a,b)
                finally:
                    store.close()
            values=np.random.default_rng(5).random((13,9),dtype=np.float32)
            store=ScoreStore(root=Path(tmp)/'groups',backend='memmap')
            try:
                ranking=rank_scores(values,20,store,chunk_size=7)
                select=np.arange(9)%2==0
                rows=boundary_summaries(values,ranking,[('a',select),('b',~select)],chunk_size=5)
                for row,mask in zip(rows,[select,~select],strict=True):
                    original=values[:,mask]
                    self.assertAlmostEqual(row['score_mean'],float(original.mean()),places=6)
                    self.assertEqual(row['score_median'],float(np.median(original)))
                    self.assertAlmostEqual(row['score_q95'],float(np.quantile(original,.95)),places=7)
                    self.assertEqual(row['n_anomaly'],int(ranking.flags[:,mask].sum()))
            finally:
                store.close()

    def test_incremental_exports_match_all_rows_and_headers(self):
        from types import SimpleNamespace
        from physiq_pv.anomaly_detection.stgan.result import STGANResult
        from physiq_pv.anomaly_detection.stgan.scoring import ScoreStore
        from scripts.run_pvgis_stgan import _export_scores
        ds=fixture();rng=np.random.default_rng(7)
        scores=rng.random((11,9),dtype=np.float32)
        features=rng.random((11,9,3),dtype=np.float32)
        manifest=pd.DataFrame({'site_key':[f's{i}' for i in range(9)]})
        cubes=SimpleNamespace(latitudes=np.arange(9),longitudes=np.arange(9))
        with tempfile.TemporaryDirectory() as tmp:
            for name,storage,chunk in [('memory','memory',65536),('disk','memmap',3),('single','memmap',1),
                                       ('flag_memory','memory',65536),('flag_disk','memmap',1)]:
                root=Path(tmp)/name;root.mkdir()
                store=ScoreStore(root=root/'scores',backend=storage)
                result=STGANResult(pd.date_range('2019-01-01',periods=11,freq='h'),tuple(map(str,range(9))),
                    ('solar','temp','wind'),scores,features,scores,scores,{},store)
                try:
                    _export_scores(result,manifest,cubes,ds.grid,root,seed=20,percentage=10,
                        chunk_size=chunk,export_all_features=not name.startswith('flag_'))
                finally:
                    result.close()
            for left in (Path(tmp)/'memory').rglob('*.csv'):
                for mode in ('disk','single'):
                    right=Path(tmp)/mode/left.relative_to(Path(tmp)/'memory')
                    pd.testing.assert_frame_equal(pd.read_csv(left),pd.read_csv(right))
            for left in (Path(tmp)/'flag_memory').rglob('*.csv'):
                right=Path(tmp)/'flag_disk'/left.relative_to(Path(tmp)/'flag_memory')
                pd.testing.assert_frame_equal(pd.read_csv(left),pd.read_csv(right))

    def test_pipeline_legacy_optimized_and_streaming(self):
        from physiq_pv.anomaly_detection.stgan import fit_and_score_stgan
        from pyproj import Transformer
        values=fixture().data
        times=pd.date_range('2018-12-31 12:00',periods=16,freq='h')
        x,y=np.meshgrid(400000.+np.arange(3)*5000,5000000.-np.arange(3)*5000)
        lon,lat=Transformer.from_crs(32632,4326,always_xy=True).transform(x.ravel(),y.ravel())
        kwargs=dict(train_timestamps=times[:12],test_timestamps=times[12:],
            location_names=tuple(map(str,range(9))),feature_names=('solar','temp','wind'),
            latitudes=lat,longitudes=lon,epochs=2,batch_size=17,hidden_size=8,n_layers=1,
            cnn_channels=4,cnn_layers=2,recent_steps=3,trend_steps=4,device='cpu')
        baseline=None
        with tempfile.TemporaryDirectory() as tmp:
            for mode,storage,workers in [('legacy','memory',0),('optimized','memmap',2)]:
                result=fit_and_score_stgan(values[:12],values[12:16],**kwargs,
                    execution_mode=mode,score_storage=storage,num_workers=workers,
                    normalized_cache_dir=Path(tmp)/mode/'normalized',score_dir=Path(tmp)/mode/'scores',
                    score_chunk_size=7)
                try:
                    arrays=[result.test_scores,result.test_generator_scores,result.test_discriminator_scores,result.test_feature_scores]
                    if baseline is None:
                        baseline=[np.array(a) for a in arrays]
                    else:
                        for a,b in zip(baseline,arrays,strict=True):
                            np.testing.assert_allclose(a,b,atol=2e-6,rtol=2e-5)
                finally:
                    result.close()


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
