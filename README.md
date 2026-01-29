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

### Step 3: Compute Generalization Gap on Historical Data

[`performative_bounds_notebook.ipynb`](performative_bounds_notebook.ipynb)

This notebook computes the generalization gap bound from Corollary as described in Appendix C.2

### Step 4: Compute and Plot Bounds on Semi-Simulated Data

[`jobseeker_tradeoff_scienceplots_markers_linestyles.ipynb`](jobseeker_tradeoff_scienceplots_markers_linestyles.ipynb)

This notebook reproduces the *Change–Learn Trade-Off* figures from the paper using the saved model predictions produced in Step 2. It:

- Loads `performative-sweep.zip`, which contains the prediction outputs for each sweep run
- Parses the treatment share from the run folder names
- Computes empirical quantities and the components of the theoretical bounds
- Implements the bounds from Theorem 3.13 and Theorem 3.15 with  
- Produces two figures visualizing those bounds using `SciencePlots`:
  - `change_learn_tradeoff_thm313_science_DZ5p29_LellSqrt28.png`
  - `change_learn_tradeoff_thm315_science_DZ5p29_LellSqrt28.png`

Each plot shows, as a function of the treatment share:

- The total generalization-gap bound
- The complexity term
- The performative term
- The sampling term

All curves are rendered as lines with point markers .


