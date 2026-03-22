results/group/
==============
Group-level outputs from pipeline_group.py.
Aggregated across all subjects in data/metadata.csv.

Files written here:
  betti_features_all.pkl      -- all subjects' features in one dict
                                 keys: subject_id
                                 values: betti_features dict

  figures/
    betti_b0_adhd_vs_tdc.png  -- group average B0 curves ± 95% CI, per measure/band
    betti_b1_adhd_vs_tdc.png  -- same for B1
    auc_boxplots.png           -- AUC distributions ADHD vs TDC, per measure/band
    auc_regression.txt         -- logistic regression results (AUC, slope, kurtosis)
                                  matching Gracia-Tabuenca 2020 Table 2 format
