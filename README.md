# JMECoffea

This code contains a columnar-based calculations of MC jet energy corrections (JEC) and flavor uncertainties, following the perscriptions of arxiv:1607.03663. At the moment, the code (L5_flavour_dep_corr branch) is adapted for the calculation of L5 flavor JEC and L2/L3 correction part is not tested.

The main structure of the code:
- `run_processor.py` run the suplied coffea processor and saves the results in .coffea files.
- `fit_response_distributions.py` fits the response distributions and stores the results in .txt files.
- `correction_fitter.ipynb` reads the response fit results, fits them vs the reco pt and stores JEC as `.txt` files
- `flavor_fractions_and_uncertainties.ipynb` reads the response histograms, response fits and the flavor corrections and produces flavor uncertainties and relevent plots. Also, it produces the plots of the fractions of each jet flavor in bins of jet pt.
- `plotters/Plotting_comparison.ipynb` allows to plot different responses and corrections vs pt or jet_eta.


## Set-up
Several set-ups option exist depending on if you are running on [lxplus](#set-up-coffea-on-the-lxplus-with-the-lcg-nightlies-environment-and-dask-on-htcondor), [lpc](#instructions-for-the-lpc-and-singularity) or [coffea casa](#instructions-for-coffea-casa). Running on any lxpus or lpc using singularity is possible but not stated here.
### Set up coffea on the **lxplus** with the lcg nightlies environment and dask on HTCondor:

Load the `lcg` environment with the relevant packages including `coffea`
```
source /cvmfs/sft-nightlies.cern.ch/lcg/views/dev4/latest/x86_64-centos7-gcc11-opt/setup.sh
```
Clone HTConodor wrapper for sending condor jobs through dask on lxplus 
https://github.com/cernops/dask-lxplus
```
git clone git@github.com:cernops/dask-lxplus.git
```

To get imports of the path work well, install the package in the editable state in a virtual environment. Explanation of this is here: https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder/50194143#50194143:
```
python -m venv venv
pip install -e .
```

Every time logging in, activate the virtual environment and after that load the lcg
```
. venv/bin/activate
source /cvmfs/sft-nightlies.cern.ch/lcg/views/dev4/latest/x86_64-centos7-gcc11-opt/setup.sh
```


**Note** Unfortunatelly, lcg104 and nightlies that have the newest coffea (0.7.21) have a bug with dask. It does not allow running the jobs on the condor cluster. To make it work one needs to use the lch103 environment and install coffea 2.2.21 locally manually and put the path to the coffea by hand on top of the lcg environment path.
Load the lcg environment.
```
. /cvmfs/sft.cern.ch/lcg/views/LCG_103/x86_64-centos7-gcc11-opt/setup.sh
```

Install coffea locally, choose path under `target`.
```
pip install --no-deps --target=. coffea
```
One might need to delete some dependancies that are installed but not needed.




### Instructions for the **LPC** (and singularity)

1. Clone the repository
2. Inside the JMECoffea directory:
```
wget https://raw.githubusercontent.com/CoffeaTeam/lpcjobqueue/main/bootstrap.sh
bash bootstrap.sh
./shell
```

### Instructions for **Coffea Casa**

1. Log into https://coffea.casa/hub/login?next=%2Fhub%2F with your CERN account
2. Go to the "Git" menu and select "Clone a repository" 

## ssh port forwarding in to use `jupyter notebooks`:
To use jupyter notebooks you have to log into lxplus using port forwarding
```
ssh -L <local_port>:localhost:<remote_port> <username>@lxplus.cern.ch
jupyter notebook --no-browser --port=<remote_port>
```
e.g
```
ssh -L 8099:localhost:8095 <username>@lxplus.cern.ch
source /cvmfs/sft-nightlies.cern.ch/lcg/views/dev4/latest/x86_64-centos7-gcc11-opt/setup.sh
jupyter notebook --no-browser --port=8095
```
Then copy the link in the output and connect in to your broswer, replacing the remote_port valur with local_port. Here you are mapping the port local_port (e.g 8099) of the localhost (your machine) to the port remote_port (e.g. 8095) of the remote server (lxplus.cern.ch). We assume that the two ports are free and therefore available.

## Running the code:
Clone the repo
```
git clone git@github.com:AndrissP/JMECoffea.git
```

### Running the L5 (or L2L3Rel) processors
Run the histogram creation (for testing, change `test_run=True` under 'Parameters of the run and switches' in `run_processor.py`)
```
python run_processor.py
```
By default it runs the CoffeaJERCProcessor_L5.py.
The results are stored in .coffea files in `out/`.

### Fitting and plotting the response distributions
Change the appropriate data_tags over which to run the data in `fit_response_distributions.py`. Run the fits:
```
python fit_response_distributions.py
```
The fit results are stored in .txt files in `out_txt/` and plots in `fig/`.

### Fitting respponses and creating output txt files
Follow `correction_fitter.ipynb`.

### Fitting the flavor uncertainties
Follow `flavor_fractions_and_uncertainties.ipynb`.
