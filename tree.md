```py

├── criscas
│   ├── attnetion_analysis.py
│   ├── data_preprocess.py
│   ├── dataset.py
│   ├── model.py
│   ├── predict_model.py
│   └── utilities.py
├── demo
│   ├── bystander_csv.py
│   ├── bystander_csv_flask.py
│   ├── bystander_demo.py
│   ├── perbase_csv.py
│   └── perbase_demo.py
├── haplotype
│   ├── __pycache__
│   │   ├── data_preprocess.cpython-310.pyc
│   │   ├── dataset.cpython-310.pyc
│   │   ├── hyperparam.cpython-310.pyc
│   │   ├── model.cpython-310.pyc
│   │   ├── predict_model.cpython-310.pyc
│   │   └── utilities.cpython-310.pyc
│   ├── data_preprocess.py
│   ├── dataset.py
│   ├── hyperparam.py
│   ├── model.py
│   ├── predict_model.py
│   ├── utilities.py
│   └── viz_resources
│       ├── begin.txt
│       ├── end.txt
│       ├── header.txt
│       └── jupcellstyle.css
├── sample_data
│   ├── abemax_sampledata.csv
│   ├── bystander_sampledata.csv
│   ├── bystander_webapp_abedata.csv
│   ├── fig_dir
│   │   └── mean
│   │       ├── ABEmax_seqattn_CTRL_DOCK3NO2_basepos_5_predoption_mean.pdf
│   │       ├── ABEmax_seqattn_CTRL_TARDBPNO2_basepos_5_predoption_mean.pdf
│   │       ├── ABEmax_seqattn_seq_0_basepos_5_predoption_mean.pdf
│   │       ├── Target-AID_seqattn_seq_0_basepos_2_predoption_mean.pdf
│   │       └── Target-AID_seqattn_seq_0_basepos_4_predoption_mean.pdf
│   ├── predictions
│   │   ├── predictions_allruns.csv
│   │   └── predictions_predoption_mean.csv
│   ├── predictions_haplo
│   │   ├── ABEmax_seq_0_haplotype.html
│   │   ├── ABEmax_seq_1_haplotype.html
│   │   ├── ABEmax_seq_demo_haplotype.html
│   │   └── index.html
│   ├── source_data
│   │   ├── 41467_2021_25375_MOESM1_ESM.pdf
│   │   ├── 41467_2021_25375_MOESM2_ESM.xlsx
│   │   ├── 41467_2021_25375_MOESM3_ESM.xlsx
│   │   ├── 41467_2021_25375_MOESM4_ESM.xlsx
│   │   ├── 41467_2021_25375_MOESM5_ESM.pdf
│   │   ├── 41467_2021_25375_MOESM6_ESM.pdf
│   │   └── 41467_2021_25375_MOESM8_ESM.xlsx
│   └── source_data_sample
│       ├── M2_ESM.xlsx
│       ├── M3_ESM.xlsx
│       ├── M4_ESM.xlsx
│       ├── M6_ESM.pdf
│       └── M8_ESM.xlsx
├── trained_models
│   ├── bystander
│   │   ├── ABE8e
│   │   │   └── train_val
│   │   │       ├── run_0
│   │   │       │   ├── config
│   │   │       │   │   ├── exp_options.pkl
│   │   │       │   │   └── mconfig.pkl
│   │   │       │   └── model_statedict
│   │   │       │       └── HaplotypeTransformer.pkl
│   │   │       ├── run_1
│   │   │       │   ├── config
│   │   │       │   │   ├── exp_options.pkl
│   │   │       │   │   └── mconfig.pkl
│   │   │       │   └── model_statedict
│   │   │       │       └── HaplotypeTransformer.pkl
│   │   │       ├── run_2
│   │   │       │   ├── config
│   │   │       │   │   ├── exp_options.pkl
│   │   │       │   │   └── mconfig.pkl
│   │   │       │   └── model_statedict
│   │   │       │       └── HaplotypeTransformer.pkl
│   │   │       ├── run_3
│   │   │       │   ├── config
│   │   │       │   │   ├── exp_options.pkl
│   │   │       │   │   └── mconfig.pkl
│   │   │       │   └── model_statedict
│   │   │       │       └── HaplotypeTransformer.pkl
│   │   │       └── run_4
│   │   │           ├── config
│   │   │           │   ├── exp_options.pkl
│   │   │           │   └── mconfig.pkl
│   │   │           └── model_statedict
│   │   │               └── HaplotypeTransformer.pkl
│   │   ├── ABEmax
│   │   │   └── train_val
│   │   │       ├── run_0
│   │   │       │   ├── config
│   │   │       │   │   ├── exp_options.pkl
│   │   │       │   │   └── mconfig.pkl
│   │   │       │   └── model_statedict
│   │   │       │       └── HaplotypeTransformer.pkl
│   │   │       ├── run_1
│   │   │       │   ├── config
│   │   │       │   │   ├── exp_options.pkl
│   │   │       │   │   └── mconfig.pkl
│   │   │       │   └── model_statedict
│   │   │       │       └── HaplotypeTransformer.pkl
│   │   │       ├── run_2
│   │   │       │   ├── config
│   │   │       │   │   ├── exp_options.pkl
│   │   │       │   │   └── mconfig.pkl
│   │   │       │   └── model_statedict
│   │   │       │       └── HaplotypeTransformer.pkl
│   │   │       ├── run_3
│   │   │       │   ├── config
│   │   │       │   │   ├── exp_options.pkl
│   │   │       │   │   └── mconfig.pkl
│   │   │       │   └── model_statedict
│   │   │       │       └── HaplotypeTransformer.pkl
│   │   │       └── run_4
│   │   │           ├── config
│   │   │           │   ├── exp_options.pkl
│   │   │           │   └── mconfig.pkl
│   │   │           └── model_statedict
│   │   │               └── HaplotypeTransformer.pkl
│   │   ├── BE4max
│   │   │   └── train_val
│   │   │       ├── run_0
│   │   │       │   ├── config
│   │   │       │   │   ├── exp_options.pkl
│   │   │       │   │   └── mconfig.pkl
│   │   │       │   └── model_statedict
│   │   │       │       └── HaplotypeTransformer.pkl
│   │   │       ├── run_1
│   │   │       │   ├── config
│   │   │       │   │   ├── exp_options.pkl
│   │   │       │   │   └── mconfig.pkl
│   │   │       │   └── model_statedict
│   │   │       │       └── HaplotypeTransformer.pkl
│   │   │       ├── run_2
│   │   │       │   ├── config
│   │   │       │   │   ├── exp_options.pkl
│   │   │       │   │   └── mconfig.pkl
│   │   │       │   └── model_statedict
│   │   │       │       └── HaplotypeTransformer.pkl
│   │   │       ├── run_3
│   │   │       │   ├── config
│   │   │       │   │   ├── exp_options.pkl
│   │   │       │   │   └── mconfig.pkl
│   │   │       │   └── model_statedict
│   │   │       │       └── HaplotypeTransformer.pkl
│   │   │       └── run_4
│   │   │           ├── config
│   │   │           │   ├── exp_options.pkl
│   │   │           │   └── mconfig.pkl
│   │   │           └── model_statedict
│   │   │               └── HaplotypeTransformer.pkl
│   │   └── Target-AID
│   │       └── train_val
│   │           ├── run_0
│   │           │   ├── config
│   │           │   │   ├── exp_options.pkl
│   │           │   │   └── mconfig.pkl
│   │           │   └── model_statedict
│   │           │       └── HaplotypeTransformer.pkl
│   │           ├── run_1
│   │           │   ├── config
│   │           │   │   ├── exp_options.pkl
│   │           │   │   └── mconfig.pkl
│   │           │   └── model_statedict
│   │           │       └── HaplotypeTransformer.pkl
│   │           ├── run_2
│   │           │   ├── config
│   │           │   │   ├── exp_options.pkl
│   │           │   │   └── mconfig.pkl
│   │           │   └── model_statedict
│   │           │       └── HaplotypeTransformer.pkl
│   │           ├── run_3
│   │           │   ├── config
│   │           │   │   ├── exp_options.pkl
│   │           │   │   └── mconfig.pkl
│   │           │   └── model_statedict
│   │           │       └── HaplotypeTransformer.pkl
│   │           └── run_4
│   │               ├── config
│   │               │   ├── exp_options.pkl
│   │               │   └── mconfig.pkl
│   │               └── model_statedict
│   │                   └── HaplotypeTransformer.pkl
│   └── perbase
│       ├── ABE8e
│       │   └── train_val
│       │       ├── run_0
│       │       │   ├── config
│       │       │   │   ├── exp_options.pkl
│       │       │   │   └── mconfig.pkl
│       │       │   └── model_statedict
│       │       │       └── Transformer.pkl
│       │       ├── run_1
│       │       │   ├── config
│       │       │   │   ├── exp_options.pkl
│       │       │   │   └── mconfig.pkl
│       │       │   └── model_statedict
│       │       │       └── Transformer.pkl
│       │       ├── run_2
│       │       │   ├── config
│       │       │   │   ├── exp_options.pkl
│       │       │   │   └── mconfig.pkl
│       │       │   └── model_statedict
│       │       │       └── Transformer.pkl
│       │       ├── run_3
│       │       │   ├── config
│       │       │   │   ├── exp_options.pkl
│       │       │   │   └── mconfig.pkl
│       │       │   └── model_statedict
│       │       │       └── Transformer.pkl
│       │       └── run_4
│       │           ├── config
│       │           │   ├── exp_options.pkl
│       │           │   └── mconfig.pkl
│       │           └── model_statedict
│       │               └── Transformer.pkl
│       ├── ABEmax
│       │   └── train_val
│       │       ├── run_0
│       │       │   ├── config
│       │       │   │   ├── exp_options.pkl
│       │       │   │   └── mconfig.pkl
│       │       │   └── model_statedict
│       │       │       └── Transformer.pkl
│       │       ├── run_1
│       │       │   ├── config
│       │       │   │   ├── exp_options.pkl
│       │       │   │   └── mconfig.pkl
│       │       │   └── model_statedict
│       │       │       └── Transformer.pkl
│       │       ├── run_2
│       │       │   ├── config
│       │       │   │   ├── exp_options.pkl
│       │       │   │   └── mconfig.pkl
│       │       │   └── model_statedict
│       │       │       └── Transformer.pkl
│       │       ├── run_3
│       │       │   ├── config
│       │       │   │   ├── exp_options.pkl
│       │       │   │   └── mconfig.pkl
│       │       │   └── model_statedict
│       │       │       └── Transformer.pkl
│       │       └── run_4
│       │           ├── config
│       │           │   ├── exp_options.pkl
│       │           │   └── mconfig.pkl
│       │           └── model_statedict
│       │               └── Transformer.pkl
│       ├── BE4max
│       │   └── train_val
│       │       ├── run_0
│       │       │   ├── config
│       │       │   │   ├── exp_options.pkl
│       │       │   │   └── mconfig.pkl
│       │       │   └── model_statedict
│       │       │       └── Transformer.pkl
│       │       ├── run_1
│       │       │   ├── config
│       │       │   │   ├── exp_options.pkl
│       │       │   │   └── mconfig.pkl
│       │       │   └── model_statedict
│       │       │       └── Transformer.pkl
│       │       ├── run_2
│       │       │   ├── config
│       │       │   │   ├── exp_options.pkl
│       │       │   │   └── mconfig.pkl
│       │       │   └── model_statedict
│       │       │       └── Transformer.pkl
│       │       ├── run_3
│       │       │   ├── config
│       │       │   │   ├── exp_options.pkl
│       │       │   │   └── mconfig.pkl
│       │       │   └── model_statedict
│       │       │       └── Transformer.pkl
│       │       └── run_4
│       │           ├── config
│       │           │   ├── exp_options.pkl
│       │           │   └── mconfig.pkl
│       │           └── model_statedict
│       │               └── Transformer.pkl
│       └── Target-AID
│           └── train_val
│               ├── run_0
│               │   ├── config
│               │   │   ├── exp_options.pkl
│               │   │   └── mconfig.pkl
│               │   └── model_statedict
│               │       └── Transformer.pkl
│               ├── run_1
│               │   ├── config
│               │   │   ├── exp_options.pkl
│               │   │   └── mconfig.pkl
│               │   └── model_statedict
│               │       └── Transformer.pkl
│               ├── run_2
│               │   ├── config
│               │   │   ├── exp_options.pkl
│               │   │   └── mconfig.pkl
│               │   └── model_statedict
│               │       └── Transformer.pkl
│               ├── run_3
│               │   ├── config
│               │   │   ├── exp_options.pkl
│               │   │   └── mconfig.pkl
│               │   └── model_statedict
│               │       └── Transformer.pkl
│               └── run_4
│                   ├── config
│                   │   ├── exp_options.pkl
│                   │   └── mconfig.pkl
│                   └── model_statedict
│                       └── Transformer.pkl

```





158 directories, 205 files
