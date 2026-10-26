"""Check the delivered arrays against source rows and verify both loaders."""
from pathlib import Path
import hashlib, io, json, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from data_loader import SolarData, SolarPilot

root=Path(__file__).resolve().parents[1]
full=SolarData(root)
pilot=SolarPilot(root)
rng=np.random.default_rng(4531)
selected=rng.choice(len(full.samples),size=48,replace=False)
source_files={}
for sid in selected:
    for fid in [full.arrays['input_frame_ids'][sid,0],full.arrays['input_frame_ids'][sid,-1],full.arrays['target_frame_ids'][sid]]:
        row=full.frames.iloc[fid]
        path=root/row.source_file
        if path not in source_files:source_files[path]=pq.ParquetFile(path)
        item=source_files[path].read_row_group(int(row.row_group),columns=['time','pv'],use_threads=False).slice(int(row.row_in_group),1).to_pydict()
        assert pd.Timestamp(item['time'][0]).tz_convert('UTC')==row.time_utc
        assert np.float32(item['pv'][0])==np.float32(row.pv_kw)
        assert hashlib.sha256(full.image(int(fid)).tobytes()).hexdigest()==row.image_pixel_sha256

with np.load(root/'data/pilot/pilot_data.npz',allow_pickle=False) as z:
    original_samples=z['original_sample_ids']
    original_frames=z['original_frame_ids']
assert not np.any(pilot.arrays['split']=='test')
assert np.array_equal(original_frames[pilot.arrays['input_frame_ids']],full.arrays['input_frame_ids'][original_samples])
assert np.array_equal(original_frames[pilot.arrays['target_frame_ids']],full.arrays['target_frame_ids'][original_samples])
assert np.array_equal(pilot.arrays['target_pv_kw'],full.arrays['target_pv_kw'][original_samples])
assert np.array_equal(pilot.arrays['pv_history_kw'],full.arrays['pv_history_kw'][original_samples])

pilot_ids=rng.choice(pilot.ids('train'),size=16,replace=False)
start=time.perf_counter()
batch=pilot.batch(pilot_ids)
pilot_time=time.perf_counter()-start
start=time.perf_counter()
full_batch=full.batch(original_samples[pilot_ids])
full_time=time.perf_counter()-start
for key in batch:
    assert batch[key].shape==full_batch[key].shape
    assert np.array_equal(batch[key],full_batch[key]),key
    assert np.isfinite(batch[key]).all()
assert batch['images'].shape==(16,16,64,64,3)
assert 0<=batch['images'].min()<=batch['images'].max()<=1

x,y,names=pilot.tabular(pilot_ids)
x_full,y_full,names_full=full.tabular(original_samples[pilot_ids])
assert x.shape==(16,404) and len(names)==404
assert np.array_equal(x,x_full) and np.array_equal(y,y_full) and names==names_full
assert np.isfinite(x).all()
history=pilot.grid[pilot.arrays['input_frame_ids'][pilot_ids]]
assert np.allclose(x[:,116:212],history.mean(axis=1),atol=1e-6)
assert np.allclose(x[:,212:308],history.std(axis=1),atol=1e-6)

optional=[]
for folder in ['ERA5','GFS']:
    for path in sorted((root/'data/raw/solarbench'/folder).glob('*.parquet')):
        f=pq.ParquetFile(path)
        # Read all optional files to verify they are intact. Their semantic
        # suitability as forecast inputs is explicitly not certified here.
        t=f.read(use_threads=False)
        optional.append(dict(file=str(path.relative_to(root)),rows=t.num_rows,columns=t.num_columns,
                             null_cells=sum(t[c].null_count for c in t.column_names),readable=True))

nb=json.loads((root/'Solar_Forecasting_Data_Starter.ipynb').read_text())
compiled=0
for i,cell in enumerate(nb['cells']):
    if cell['cell_type']=='code':
        source=''.join(cell['source'])
        if not source.startswith('%'):
            compile(source,f'notebook_cell_{i}','exec');compiled+=1

report=dict(status='passed',source_rows_spot_checked=144,all_pilot_index_and_label_mappings_match=True,
             full_and_pilot_batches_identical=True,batch_shape=list(batch['images'].shape),
             batch_image_tensor_bytes=batch['images'].nbytes,baseline_feature_count=404,
             temporal_image_summaries_independently_checked=True,
             pilot_batch_seconds=round(pilot_time,4),full_batch_seconds=round(full_time,4),
             timing_scope='One local CPU data-loading batch only; not GPU training or general speed benchmark',
             optional_weather_file_checks=optional,notebook_code_cells_syntax_checked=compiled,
             notebook_executed_in_colab=False,neural_training_tested=False)
(root/'validation/loader_checks.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
