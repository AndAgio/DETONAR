# DETONAR

## Feature extraction from CSVs
python feature_extractor.py --scenario="Blackhole" --out_feat_files="log/features_extracted/" --single_sim="102"

## IDS run
python new_arima_ids.py --scenario="Blackhole" --chosen_simulation="simulation-102" --simulation_time=1500 --time_window=10 --lag_val=30
