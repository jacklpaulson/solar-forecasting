"""Rebuild audited forecasting inputs from the pinned SKIPP'D Parquet files.

Run from any directory: python prepare_data.py
Required packages: numpy, pandas, pyarrow, pillow.
"""
from pathlib import Path
import hashlib, io, json, math, sys, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/solarbench'
OUT = ROOT / 'data/prepared'
QA = ROOT / 'validation'
OUT.mkdir(parents=True, exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)

def main():
    started=time.time()
    records=[]
    feature_parts=[]
    image_checks={'decoded':0,'invalid':0,'wrong_shape':0}
    frame_id=0
    source_checks=[]
    for source_split in ['train','test']:
        split_start=frame_id
        for path in sorted((RAW/'ground').glob(source_split+'-*.parquet')):
            source=pq.ParquetFile(path)
            for rg in range(source.num_row_groups):
                tab=source.read_row_group(rg,use_threads=False)
                observations=tab.select(['time','pv']).to_pandas(use_threads=False)
                batch=[]
                for j,item in enumerate(tab['image'].to_pylist()):
                    with Image.open(io.BytesIO(item['bytes'])) as image:
                        rgb=np.asarray(image.convert('RGB'))
                    if rgb.shape != (64,64,3):
                        image_checks['wrong_shape']+=1
                        raise ValueError(f'Unexpected image shape at {path.name}:{rg}:{j}')
                    x=rgb.astype(np.float32)/255.0
                    grid=x.reshape(4,16,4,16,3)
                    features=np.concatenate([grid.mean(axis=(1,3)).ravel(),grid.std(axis=(1,3)).ravel()])
                    batch.append(features)
                    row=observations.iloc[j]
                    records.append(dict(frame_id=frame_id,source_split=source_split,
                         source_file=str(path.relative_to(ROOT)),row_group=rg,row_in_group=j,
                         time_utc=row.time.tz_convert('UTC'),time_local=row.time,
                         local_date=str(row.time.date()),pv_kw=float(row.pv),
                         image_pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                         image_std_0_255=float(rgb.std())))
                    if frame_id in [0,500,1000]:
                        Image.fromarray(rgb).save(QA/f'example_frame_{frame_id}.png')
                    frame_id+=1
                    image_checks['decoded']+=1
                feature_parts.append(np.asarray(batch,dtype=np.float32))
            print(f'Decoded and checked {frame_id:,} images',flush=True)
        labels=pq.ParquetFile(RAW/'ground_no_sky_img'/f'{source_split}-00000-of-00001.parquet').read(use_threads=False).to_pandas(use_threads=False)
        paired=pd.DataFrame(records[split_start:])
        timestamps_equal=np.array_equal(pd.DatetimeIndex(paired.time_utc).as_unit('ns').asi8,pd.DatetimeIndex(labels.time).as_unit('ns').asi8)
        values_equal=np.array_equal(paired.pv_kw.to_numpy(dtype=np.float32),labels.pv.to_numpy(dtype=np.float32))
        assert timestamps_equal and values_equal, 'Standalone labels disagree with image-paired observations'
        source_checks.append(dict(split=source_split,rows=len(labels),labels_match_images=True))

    frames=pd.DataFrame(records)
    assert len(frames)==363375
    assert frames.time_utc.notna().all() and np.isfinite(frames.pv_kw).all()
    assert not frames.time_utc.duplicated().any()
    assert (frames.pv_kw>=0).all()
    assert len(set(frames.query("source_split=='train'").local_date) & set(frames.query("source_split=='test'").local_date))==0

    # A fixed validation split by whole days, sampled within each year-month.
    # This is a within-site held-out-day experiment, not a future-year backtest.
    train_days=sorted(frames.loc[frames.source_split=='train','local_date'].unique())
    by_month={}
    for day in train_days:
        by_month.setdefault(day[:7],[]).append(day)
    val_days=set()
    for month,days in by_month.items():
        if len(days)>1:
            ranked=sorted(days,key=lambda d:hashlib.sha256(('4531:'+d).encode()).hexdigest())
            val_days.update(ranked[:min(len(days)-1,max(1,round(len(days)*0.2)))])
    frames['split']=np.where(frames.source_split=='test','test',np.where(frames.local_date.isin(val_days),'validation','train'))
    cross_hash=frames.groupby('image_pixel_sha256').split.nunique()
    cross_hash=cross_hash[cross_hash>1].index
    # Keep the official test observations, exclude exact copied pixels from
    # earlier splits. Also remove train pixels duplicated in validation.
    frames['usable']=True
    for digest in cross_hash:
        sub=frames.loc[frames.image_pixel_sha256==digest]
        keep='test' if 'test' in set(sub.split) else 'validation'
        frames.loc[sub.index[sub.split!=keep],'usable']=False

    ids=[]
    target_ids=[]
    split_names=[]
    times_sec=pd.DatetimeIndex(frames.time_utc).as_unit('s').asi8
    # Require all 31 source records t-15,...,t+15 to be exactly one minute apart.
    # No rounding, interpolation, nearest-neighbor future match, or gap filling.
    for (split,day),sub in frames.groupby(['split','local_date'],sort=True):
        sub=sub.sort_values('time_utc')
        seq=sub.frame_id.to_numpy()
        if len(seq)<31:
            continue
        secs=times_sec[seq]
        good=frames.usable.to_numpy()[seq]
        for start in range(len(seq)-30):
            block=seq[start:start+31]
            if good[start:start+31].all() and np.all(np.diff(secs[start:start+31])==60):
                ids.append(block[:16])
                target_ids.append(block[30])
                split_names.append(split)
    input_ids=np.asarray(ids,dtype=np.int32)
    target_ids=np.asarray(target_ids,dtype=np.int32)
    split_names=np.asarray(split_names,dtype='<U10')
    issue_ids=input_ids[:,-1]
    assert np.all(times_sec[target_ids]-times_sec[issue_ids]==900)
    assert np.all(np.diff(times_sec[input_ids],axis=1)==60)
    day_array=frames.local_date.to_numpy()
    split_array=frames.split.to_numpy()
    assert np.all(day_array[input_ids]==day_array[issue_ids,None])
    assert np.all(day_array[target_ids]==day_array[issue_ids])
    assert np.all(split_array[input_ids]==split_names[:,None])
    assert np.all(split_array[target_ids]==split_names)
    pv=frames.pv_kw.to_numpy(dtype=np.float32)
    y=pv[target_ids]
    windows=pd.DataFrame(dict(sample_id=np.arange(len(y)),split=split_names,
                   issue_frame_id=issue_ids,target_frame_id=target_ids,
                   issue_time_utc=frames.time_utc.iloc[issue_ids].to_numpy(),
                   target_time_utc=frames.time_utc.iloc[target_ids].to_numpy(),
                   local_date=day_array[issue_ids],target_pv_kw=y,persistence_kw=pv[issue_ids]))
    pilot_days={}
    for split,n in [('train',30),('validation',8)]:
        days=np.array(sorted(windows.loc[windows.split==split,'local_date'].unique()))
        pilot_days[split]=days[np.linspace(0,len(days)-1,min(n,len(days)),dtype=int)].tolist()
    windows['pilot']=False
    for split,days in pilot_days.items():
        windows.loc[(windows.split==split)&windows.local_date.isin(days),'pilot']=True
    features=np.concatenate(feature_parts)
    frames.to_parquet(OUT/'frames.parquet',index=False,compression='zstd')
    windows.to_parquet(OUT/'forecast_samples.parquet',index=False,compression='zstd')
    windows.to_csv(OUT/'forecast_samples.csv.gz',index=False,compression='gzip')
    np.savez_compressed(OUT/'forecast_arrays.npz',input_frame_ids=input_ids,target_frame_ids=target_ids,
                        pv_history_kw=pv[input_ids],target_pv_kw=y,split=split_names,pilot=windows.pilot.to_numpy())
    np.savez_compressed(OUT/'image_grid_features.npz',features=features)
    days=frames.groupby(['split','local_date'],as_index=False).agg(frames=('frame_id','size'))
    counts=windows.groupby(['split','local_date'],as_index=False).size().rename(columns={'size':'forecast_samples'})
    days.merge(counts,how='left',on=['split','local_date']).fillna({'forecast_samples':0}).to_csv(OUT/'day_splits.csv',index=False)
    (OUT/'pilot_days.json').write_text(json.dumps(pilot_days,indent=2))
    summary=[]
    for split in ['train','validation','test']:
        s=frames[frames.split==split]
        w=windows[windows.split==split]
        summary.append(dict(split=split,frames=len(s),days=s.local_date.nunique(),forecast_samples=len(w),pilot_samples=int(w.pilot.sum())))
    val=windows[windows.split=='validation']
    err=val.persistence_kw.to_numpy()-val.target_pv_kw.to_numpy()
    audit=dict(status='data preparation verified',source_rows=len(frames),image_checks=image_checks,
       image_and_standalone_label_checks=source_checks,missing_timestamps=0,missing_or_nonfinite_pv=0,
       duplicate_timestamps=0,source_pv_range_kw=[float(pv.min()),float(pv.max())],
       date_range=[frames.local_date.min(),frames.local_date.max()],unique_days=frames.local_date.nunique(),
       cross_split_exact_image_hashes=len(cross_hash),excluded_frames_for_cross_split_image_duplicates=int((~frames.usable).sum()),
       split_summary=summary,forecast_samples=len(windows),input_images_per_example=16,history_minutes=15,
       horizon_minutes=15,exact_60_second_sequence_checks=True,exact_900_second_target_checks=True,
       whole_day_split_disjointness=True,validation_persistence_mae_kw=float(np.mean(np.abs(err))),
       validation_persistence_rmse_kw=float(np.sqrt(np.mean(err**2))),
       learned_models_trained=False,seconds=round(time.time()-started,2))
    (QA/'data_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':
    main()
