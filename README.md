# IMDR
This is the code implementation of the "Incomplete Multi-View Drug Recommendation via Multi-Level Representation Learning and Curriculum Learning"

![IMDR Framework](./figs/overall%20.jpg)

## Installation

To set up the environment using conda, follow these steps:

1. Create a new conda environment:

    ```conda create --name imdr python=3.9```

2. Activate the environment:

    ```conda activate imdr```

3. Install the required dependencies:

    ```pip install -r requirements.txt```

## Configuration
1. Configure the MIMIC dataset path in the ```config.yaml``` file (which has multiple csv files under it) and adjust the hyperparameters as appropriate.
2. Configure your ```base_url``` and ```api_key``` in the ```gen_token_complexity.py``` file, and then run the ```python gen_token_complexity.py``` command for additional icd encoding knowledge.

## Run
Use the following command to run the IMDR model:

```python run.py```


The results of the experiment will be printed out in the terminal.


## Citation
TODO
