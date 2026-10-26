"""Load the prepared data without expanding every image sequence on disk."""
from pathlib import Path
from collections import OrderedDict
import io
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

class SolarData:
    def __init__(self, root=None, cached_row_groups=32):
        self.root=Path(root) if root else Path(__file__).resolve().parents[1]
        prep=self.root/'data/prepared'
        self.frames=pq.ParquetFile(prep/'frames.parquet').read(use_threads=False).to_pandas(use_threads=False)
        self.samples=pq.ParquetFile(prep/'forecast_samples.parquet').read(use_threads=False).to_pandas(use_threads=False)
        with np.load(prep/'forecast_arrays.npz',allow_pickle=False) as z:
            self.arrays={k:z[k] for k in z.files}
        with np.load(prep/'image_grid_features.npz',allow_pickle=False) as z:
            self.grid=z['features']
        self.files={};self.cache=OrderedDict();self.cache_limit=cached_row_groups
        self.source_files=self.frames.source_file.to_numpy()
        self.row_groups=self.frames.row_group.to_numpy()
        self.row_offsets=self.frames.row_in_group.to_numpy()
        origins=pd.DatetimeIndex(self.frames.time_local.iloc[self.arrays['input_frame_ids'][:,-1]])
        minutes=origins.hour+origins.minute/60+origins.second/3600
        angle_day=2*np.pi*minutes/24
        angle_year=2*np.pi*(origins.dayofyear-1)/np.where(origins.is_leap_year,366,365)
        self.calendar=np.column_stack([np.sin(angle_day),np.cos(angle_day),np.sin(angle_year),np.cos(angle_year)]).astype(np.float32)

    def ids(self, split='train', pilot=False):
        mask=self.arrays['split']==split
        if pilot:mask &= self.arrays['pilot']
        return np.flatnonzero(mask)

    def image(self, frame_id):
        file=self.source_files[frame_id];group=int(self.row_groups[frame_id])
        key=(file,group)
        if key not in self.cache:
            if file not in self.files:self.files[file]=pq.ParquetFile(self.root/file)
            self.cache[key]=self.files[file].read_row_group(group,columns=['image'],use_threads=False)['image'].to_pylist()
            if len(self.cache)>self.cache_limit:self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        item=self.cache[key][int(self.row_offsets[frame_id])]
        with Image.open(io.BytesIO(item['bytes'])) as im:
            return np.array(im.convert('RGB'),dtype=np.uint8)

    def batch(self, sample_ids):
        sample_ids=np.asarray(sample_ids,dtype=np.int64)
        frame_ids=self.arrays['input_frame_ids'][sample_ids]
        unique=np.unique(frame_ids)
        decoded={int(i):self.image(int(i)) for i in unique}
        images=np.stack([[decoded[int(i)] for i in seq] for seq in frame_ids]).astype(np.float32)/255
        return dict(images=images,power_history=self.arrays['pv_history_kw'][sample_ids]/30.1,
                    calendar=self.calendar[sample_ids],target_kw=self.arrays['target_pv_kw'][sample_ids],
                    target_scaled=self.arrays['target_pv_kw'][sample_ids]/30.1)

    def tabular(self, sample_ids, include_images=True):
        sample_ids=np.asarray(sample_ids,dtype=np.int64)
        parts=[self.arrays['pv_history_kw'][sample_ids]/30.1,self.calendar[sample_ids]]
        names=[f'pv_lag_{15-i}_min_scaled' for i in range(16)]+['hour_sin','hour_cos','day_of_year_sin','day_of_year_cos']
        if include_images:
            seq=self.arrays['input_frame_ids'][sample_ids]
            current=self.grid[seq[:,-1]]
            change=current-self.grid[seq[:,0]]
            # All sixteen images contribute, without allocating N x 16 x 96.
            mean=np.zeros_like(current);m2=np.zeros_like(current)
            for step in range(16):
                values=self.grid[seq[:,step]]
                delta=values-mean
                mean+=delta/(step+1)
                m2+=delta*(values-mean)
            spread=np.sqrt(np.maximum(m2/16,0))
            parts.extend([current,mean,spread,change])
            names.extend([f'image_grid_{i}' for i in range(96)]
                         +[f'image_grid_history_mean_{i}' for i in range(96)]
                         +[f'image_grid_history_std_{i}' for i in range(96)]
                         +[f'image_grid_change_15min_{i}' for i in range(96)])
        return np.concatenate(parts,axis=1).astype(np.float32),self.arrays['target_pv_kw'][sample_ids],names

    def batches(self, split='train', batch_size=16, pilot=False, shuffle=True, seed=4531):
        ids=self.ids(split,pilot)
        if shuffle:np.random.default_rng(seed).shuffle(ids)
        for start in range(0,len(ids),batch_size):
            yield self.batch(ids[start:start+batch_size])

class SolarPilot(SolarData):
    """A self-contained small subset; does not need the full Parquet data."""
    def __init__(self, root=None):
        self.root=Path(root) if root else Path(__file__).resolve().parents[1]
        with np.load(self.root/'data/pilot/pilot_data.npz',allow_pickle=False) as z:
            self.arrays={k:z[k] for k in ['input_frame_ids','target_frame_ids','pv_history_kw','target_pv_kw','split','pilot']}
            self.images=z['images']
            self.grid=z['image_grid_features']
            self.calendar=z['calendar']
    def image(self, frame_id):
        return self.images[frame_id]
