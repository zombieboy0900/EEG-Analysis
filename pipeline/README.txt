pipeline/
=========
Modular pipeline scripts. Import shared functions from TDADHD_copy.py.

Files:
  config.py               -- shared constants (FREQ_BANDS, TARGET_CHANNELS,
                             N_SURROGATES, etc.) extracted from TDADHD_copy.py
                             Both subject and group scripts import from here.

  pipeline_subject.py     -- process one subject end-to-end
                             Usage: python pipeline_subject.py data/raw/v10p.mat
                             Writes all outputs to results/subjects/{subject_id}/

  pipeline_group.py       -- aggregate across all subjects and run group analysis
                             Usage: python pipeline_group.py
                             Reads:  results/subjects/*/betti_features.pkl
                             Reads:  data/metadata.csv
                             Writes: results/group/figures/
