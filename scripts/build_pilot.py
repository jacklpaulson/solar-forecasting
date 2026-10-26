"""Build the smaller package from the verified full dataset."""
from pathlib import Path
import json
import numpy as np
from data_loader import SolarData

def main():
    root=Path(__file__).resolve().parents[1]
    data=SolarData(root)
    selected=np.flatnonzero(data.arrays['pilot'])
    inputs=data.arrays['input_frame_ids'][selected]
    targets=data.arrays['target_frame_ids'][selected]
    unique=np.unique(np.concatenate([inputs.ravel(),targets]))
    local_inputs=np.searchsorted(unique,inputs).astype(np.int32)
    local_targets=np.searchsorted(unique,targets).astype(np.int32)
    images=np.stack([data.image(int(i)) for i in unique])
    out=root/'data/pilot'
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'pilot_data.npz',images=images,original_frame_ids=unique,
           input_frame_ids=local_inputs,target_frame_ids=local_targets,
           pv_history_kw=data.arrays['pv_history_kw'][selected],target_pv_kw=data.arrays['target_pv_kw'][selected],
           split=data.arrays['split'][selected],pilot=np.ones(len(selected),dtype=bool),
           calendar=data.calendar[selected],image_grid_features=data.grid[unique],
           original_sample_ids=selected)
    data.frames.iloc[unique].drop(columns=['image_pixel_sha256']).to_csv(out/'pilot_frames.csv.gz',index=False,compression='gzip')
    data.samples.iloc[selected].to_csv(out/'pilot_examples.csv.gz',index=False,compression='gzip')
    report=dict(unique_images=len(unique),forecast_samples=len(selected),
                split_counts={s:int(np.sum(data.arrays['split'][selected]==s)) for s in ['train','validation']},
                includes_test=False,uncompressed_image_array_bytes=images.nbytes,
                compressed_file_bytes=(out/'pilot_data.npz').stat().st_size)
    (out/'pilot_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
