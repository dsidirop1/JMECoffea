# JMECoffea

## Set up coffea on the lxplus with the lcg nightlies environment and dask on HTCondor:

Load the `lcg` environment with the relevant packages including `coffea`
```
source /cvmfs/sft-nightlies.cern.ch/lcg/views/dev4/latest/x86_64-centos7-gcc11-opt/setup.sh
```
Clone HTConodor wrapper for sending condor jobs through dask on lxplus 
https://github.com/cernops/dask-lxplus
```
git clone git@github.com:cernops/dask-lxplus.git
```

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
jupyter notebook --no-browser --port=8099
```
Then copy the link in the output and connect in to your broswer, replacing the remote_port valur with local_port. Here you are mapping the port local_port (e.g 8099) of the localhost (your machine) to the port remote_port (e.g. 8095) of the remote server (lxplus.cern.ch). We assume that the two ports are free and therefore available.

## Running the code:
Clone the repo
```
git clone git@github.com:AndrissP/JMECoffea.git
```
Run the histogram creation and responce fitting code by the following `CoffeaJERC-Andris.ipynb` or running
```
python CoffeaJERC-Andris.py
```

### Fitting respponses and creating output txt files
Follow `Response_fitting.ipynb`


## Instructions for the LPC (and singularity)

1. Clone the repository
2. Inside the JMECoffea directory:
```
wget https://raw.githubusercontent.com/CoffeaTeam/lpcjobqueue/main/bootstrap.sh
bash bootstrap.sh
./shell
```

## Instructions for Coffea Casa

1. Log into https://coffea.casa/hub/login?next=%2Fhub%2F with your CERN account
2. Go to the "Git" menu and select "Clone a repository" 
