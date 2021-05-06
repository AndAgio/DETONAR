# DETONAR

This repo contains the official implementation of our paper entitled *DETONAR: Detection of Routing Attacks in RPL-based IoT*. The full paper can be found [here.]: https://ieeexplore.ieee.org/document/9415869
Consider citing our work:
```
@article{agiollo2021detonar,
  title={DETONAR: Detection of Routing Attacks in RPL-based IoT},
  author={Agiollo, Andrea and Conti, Mauro and Kaliyar, Pallavi and Lin, TsungNan and Pajola, Luca},
  journal={IEEE Transactions on Network and Service Management},
  year={2021},
  publisher={IEEE}
}
```

## Feature extraction from CSVs
We produce a python file in order to extract features of each device for each time window given the CSV file of a simulation.
In particular, the file runs on all simulations of a specific attack like Blackhole.
User can select the output folder where extracted features will be placed. This folder will then be used by the IDS runner in order to check if an attack is identified.

    python feature_extractor.py --scenario="Blackhole" --out_feat_files="log/features_extracted/"

#### Possible parameters for scenario:
* Legitimate
* Blackhole
* Clone_ID
* Continuous_Sinkhole
* DIS
* Hello_Flood
* Local_Repair
* Rank
* Replay
* Selective_Forward
* Sinkhole
* Sybil
* Version
* Wormhole
* Worst_Parent

## IDS run
We produce a python file that runs DETONAR over the files containing extracted features usingg previous python script.
The script accepts a path scenario and a single simulation name. It builds the corresponding path to the files containing features extracted using the out_feat_files coordinates.
This script also accepts simulation_time - i.e. the amount of seconds on which DETONAR is run -, time_window - i.e. the length, in seconds, of the considered time window -, and lag_val - i.e. the length of the history used by ARIMA during anomaly detection.

    python new_arima_ids.py --scenario="Blackhole" --chosen_simulation="simulation-102" --simulation_time=1500 --time_window=10 --lag_val=30
