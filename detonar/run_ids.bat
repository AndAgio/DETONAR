@echo off
:: set variables
set SCENARIO=%1
set /a SIM_TIME=%2
set /a WINDOW=%3
set /a LAG=%4
:: Start with shell commands
:: Activate python virtual environment and run python script
CALL ..\ids-env\Scripts\activate.bat 
SLEEP 1
set "condition=false"
if "%condition%" == "true" (
	python feature_extractor.py --scenario=%SCENARIO% --simulation_time=%SIM_TIME% --time_window=%WINDOW% --lag_val=%LAG%
)
python features_plots.py --scenario=%SCENARIO% --simulation_time=%SIM_TIME% --time_window=%WINDOW% --lag_val=%LAG%
echo Extracting Features...
set sims=simulation-101,simulation-102,simulation-103,simulation-104,simulation-105
for %%a in ("%sims:,=" "%") do (
    echo %%a
	@echo on
	python new_arima_ids.py --scenario=%SCENARIO% --chosen_simulation=%%a --simulation_time=%SIM_TIME% --time_window=%WINDOW% --lag_val=%LAG%
	@echo off
)