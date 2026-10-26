# Data dictionary

## Raw source data

`data/raw/solarbench/ground/*.parquet` contains the complete downloaded image benchmark.

| Field | Meaning | Units or representation |
|---|---|---|
| image.bytes | Encoded image bytes, readable with Pillow | 64 x 64 RGB after decoding |
| image.path | Source image field supplied by the repository | Often null; use embedded bytes |
| time | Source observation timestamp | Timezone-aware US/Pacific |
| pv | Solar power paired with the image | kW, float32 |

`ground_no_sky_img` supplies the same timestamps and power values without images. Every pair was checked against the image files and matched exactly. These are redundant labels, not an additional site or independent observations.

## Prepared frames

`data/prepared/frames.parquet` has one row for each of the 363,375 original observations.

- `frame_id`: stable zero-based row ID; indexes `image_grid_features.npz`.
- `source_split`: original repository split, either `train` or `test`. The source `train` includes this project's validation days.
- `source_file`, `row_group`, `row_in_group`: location of the encoded image in its original Parquet file. Read through `scripts/data_loader.py`.
- `time_utc`, `time_local`: the same instant in UTC and US/Pacific. No clock correction, rounding, or interpolation has been applied.
- `local_date`: calendar date used to keep whole days in one split.
- `pv_kw`: original power reading in kW.
- `image_pixel_sha256`: checksum of the decoded RGB pixels, used to detect exact cross-split duplicates.
- `image_std_0_255`: standard deviation across decoded pixel values; an audit statistic, not a weather label.
- `split`: `train`, `validation`, or `test` for this project.
- `usable`: retained by the exact-image duplicate policy. All downloaded frames passed that policy.

## Forecast examples

`forecast_arrays.npz` contains arrays that can be read with NumPy and `allow_pickle=False`.

| Array | Shape | Meaning |
|---|---|---|
| input_frame_ids | N x 16 | Ordered image IDs at t-15 through t, in minutes |
| target_frame_ids | N | Frame ID at exactly t+15 minutes |
| pv_history_kw | N x 16 | Power paired with each history frame |
| target_pv_kw | N | Future power reading, the regression label |
| split | N | Project split for each example |
| pilot | N | Whether the example belongs to the small development pilot |

`forecast_samples.parquet` and `forecast_samples.csv.gz` contain the same example index in table form: `sample_id`, `split`, `issue_frame_id`, `target_frame_id`, UTC issue and target timestamps, `local_date`, `target_pv_kw`, `persistence_kw`, and `pilot`. The `persistence_kw` value is the latest available history reading. It is a baseline prediction, not a measured future value.

Every retained example has 31 consecutive original observations, spaced exactly 60 seconds apart from t-15 through t+15. Only the first 16 are model inputs. The intermediate future observations are used solely to screen for complete records, and are not input features. The target is a future minutely averaged power reading, not a 15-minute energy total. This intentionally conservative protocol is not an exact reproduction of the authors' forecasting split or sampling code.

`day_splits.csv` documents all 517 day assignments and the number of retained examples. `pilot_days.json` documents the 30 training and 8 validation days selected by evenly spaced positions in sorted date lists, without using outcomes.

## Image summaries for traditional models

`image_grid_features.npz` contains a float32 array named `features`, shape 363375 x 96. Values are calculated from RGB pixels scaled to 0-1:

1. Columns 0-47: RGB means in a 4 x 4 spatial grid, ordered grid row, grid column, color channel.
2. Columns 48-95: RGB standard deviations in the same grid and order.

The loader supplies 404 baseline features: 16 scaled power-history readings, 4 cyclical time features, 96 current image summaries, 96 means and 96 standard deviations across all 16 history images, and 96 differences between the current summaries and the summaries 15 minutes earlier. Every input image therefore contributes to the traditional-model features. All per-frame summaries are also available if a richer classical baseline is warranted. Feature selection and tuning must use development data only.

Power scaling by 30.1 uses the published nominal DC capacity as a fixed reference. It is not fitted to the test set and does not imply that measured output is DC power. Report final errors in kW. Any learned standardization must be fitted only on training data.

## Compact pilot

`data/pilot/pilot_data.npz` stores unique decoded images once, local frame indices, power histories, labels, grid features, and cyclical time features. `original_frame_ids` and `original_sample_ids` link back to the full package. The pilot contains no test examples. Its metadata CSVs keep original IDs; its tensor arrays use local IDs. Use `SolarPilot` to avoid mixing these indices.

## Optional weather files

ERA5 and GFS files are preserved as supplied by the same public repository. They are not joined into the core forecasting examples. ERA5 is reanalysis and cannot be assumed available in real time. GFS includes forecast-time metadata, but issuance times, lead times, units, and interpolation need a separate audit before causal use. Their full source schemas are preserved in the dataset card. These optional files are not required to complete the proposal's core experiment.
