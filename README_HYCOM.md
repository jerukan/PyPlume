# RUNNING HYCOM SIMULATIONS

This README documents how to specifically run simulations for HYCOM forecast data.

Original instructions by Alli Ho (from https://www.evernote.com/l/AklscPbkj1xIV6PATufdQl2D4whvDgxJ0mg)

## 1. PARCELS SETUP/INSTALLATION
---
(If necessary) Update python, get conda.
Set up conda environment
1. Follow tutorial https://oceanparcels.org/#installing (steps summarized below)
	- Create a conda environment containing Parcels and all necessary dependencies
        ```shell
        conda activate root  # Linux / macOS
        activate root        # Windows
        conda env create -f environment.yml
        ```
        If the environment creation didn't work, specify the required packages instead 
        ```shell
        conda create -n py3-parcels -c conda-forge python=3.6 parcels jupyter cartopy ffmpeg
        ```
	- Activate new Parcels environment
        ```shell
        conda activate py3-parcels
        ```
	- Download examples to test install
        ```shell
        parcels_get_examples parcels_examples
        ```
	- Run the example
        ```shell
        cd parcels_examples
        python example_peninsula.py --fieldset 100 100
        ```

## 2. SETTING UP MWB TRAJECTORY PROJECT / GETTING JERRY’S PARCELS-WESTCOAST
---
Option 1: Download zip file: https://drive.google.com/file/d/14k51esx5iqsUd7w9F3TS6D0dyU7rTaL-/view?usp=sharing

Option 2: Clone from Jerry’s repo on Gitlab: https://seachest.ucsd.edu/jeyan/parcels-westcoast/-/tree/hycom

## 3. RUNNING TRAJECTORY FORECAST (start here if already installed)
---
1. Activate environment & jupyter notebook in correct directory
    ```
    conda activate py3_parcels_westcoast
    jupyter notebook
    ```
2. `access_thredds_hycom.ipynb`: Downloads HYCOM forecast from THREDDS server
	- Can adjust the daterange to download HYCOM but the default (starts at todays date and downloads the forecast +7days from now) should be fine and doesn’t need to be changed.  
3. `run_mwbGS.ipynb`: Runs the OceanParcels simulation
	- This should generate the trajectory nc file to `particledata/particle_mwb_trajectories.nc`
	- Only thing need to change is the setup code that writes the config file: 
		- Change start time tstart
		- Can change `repeat_N` to release partcles every x hours, but the default is set up for only one initial drop time at the time tstart
	- Can change the spawn points spawnpointfile from `seeds_keywest.mat` if want to:
		- Use different spawn file
		- Making a new file (lat lon matrices saved as .mat file, with labels `yf` and `xf`). This is done in `spawn_points_mats/save_spawn_points.m`

## PLOTTING
---
Default plot (below) made with `asesstrajectories.m` in folder `matlabanalysis`, make sure `fpath`/`fname` for trajectories and HYCOM are updated.
![Plot Example](/images/trajectories_24-Aug-2021.jpg)

## OPTIONAL
---
Installation:
- Use dependencies in Jerry environment file (see readme):
```
conda env create -f environment.yml
```
- This worked for me:
```
conda create -n py3-parcels -c conda-forge parcels jupyter cartopy ffmpeg --file spec-file-osx-64.txt
```

---
Updated 08-24-2021 by Alli Ho

Updated 03-31-2022 by Jerry Yan
