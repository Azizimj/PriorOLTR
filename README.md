## Installation
Run `pip install -r reqs.txt`

## Usage
This module contains the synthetic experiments and algorithms source codes. For all the synthetic experiments, we use 

```
$cd <project dir>
$python synthetic.py [ex_type] [expr_num]
```

For each experiment, set the variables in the main() function of `synthetic.py` as follows

- Non-contextual experiment

  `$python synthetic --ex_type=stand_ex_type --expr_num=1` 
- Prior initialization experiment

  `$python synthetic --ex_type=stand_ex_type --expr_num=7`
- Prior misspecification experiment (Fig 3)

  `$python synthetic --ex_type=stand_ex_type --expr_num=5`
- Linear contextual experiments (Fig 4)

  `$python synthetic --ex_type=linear_ex_type`
- Logistic contextual experiments (Fig 5)

  `$python synthetic --ex_type=log_ex_type`

### Notes:
- You can use multiprocessing by setting `parr=1`. Note that this might run into a deadlock due to memory issues. See examples here.
- After an experiment is run, the result is saved in a pickle file and the plot is generated in PDF format. 