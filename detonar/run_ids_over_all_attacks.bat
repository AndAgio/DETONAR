@echo off
:: set variables
set SCENARIOS=Blackhole,Clone_ID,Continuous_Sinkhole,DIS,Hello_Flood,Legitimate,Local_Repair,Rank,Replay,Selective_Forward,Sinkhole,Sybil,Version,Wormhole,Worst_Parent
set SIMULATIONS=simulation-101,simulation-102,simulation-103,simulation-104,simulation-105
::set SIMULATIONS=simulation-111,simulation-112,simulation-113,simulation-114,simulation-115
::set SIMULATIONS=simulation-201,simulation-202,simulation-203,simulation-204,simulation-205
:: Start with shell commands
:: Activate python virtual environment and run python script
CALL ..\ids-env\Scripts\activate.bat 
:: Call features extraction and IDS on every scenario
SLEEP 1
for %%a in ("%SCENARIOS:,=" "%") do (
    python feature_extractor.py --scenario=%%a
    for %%b in ("%SIMULATIONS:,=" "%") do (
        python new_arima_ids.py --scenario=%%a --chosen_simulation=%%b
    )
)