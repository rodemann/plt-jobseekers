# Replication Code

## Data Availability

The data used in this study are not publicly available due to their sensitive nature. Access can be requested through the Research Data Centre (FDZ) of the Institute for Employment Research (IAB). We provide all preprocessing and analysis code to enable replication for authorized users.


### Step 1: Cohort Construction

[`0-jobseeker-cohort.ipynb`](0-jobseeker-cohort.ipynb)

This notebook constructs the tabular prediction task data for jobseeker cohorts at different time points. It:

- Loads and preprocesses the SIAB administrative data
- Defines train/test cohorts of individuals entering unemployment in 2012
- Generates covariates and outcome labels at multiple time offsets (t=0, t=1)
- Outputs train/test CSV files to `data/train-test-data/`

### Step 2: Model Training and Evaluation

[`1-training.ipynb`](1-training.ipynb)

This notebook handles training and evaluation across different temporal scenarios. It:

- Trains models under various train/test time configurations (e.g., train at t=0, test at t=1)
- Analyzes feature and outcome stability across time points
- Runs performative effect simulations
- Saves model predictions and observed outcomes to `results/`

The saved predictions and outcomes from this notebook are used in subsequent bound calculations.
