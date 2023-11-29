# Rashomon Sets for Factorial Design Experiments

## Requirements

- Python >= 3.11

## Setup

1. Clone the repo
```
$ git clone https://github.com/AparaV/rashomon-tva
$ cd rashomon-tva/Code
```

2. Setup the virtual environment
```
$ python -m venv venv
$ source venv/bin/activate
(venv) $ pip install requirements.txt
```

3. Install the CTL package. The [original package](https://github.com/edgeslab/CTL) is outdated and pip install will fail with most recent Python/numpy versions. So download the [fork](https://github.com/AparaV/CTL/tree/cython-typing) inside `rashomon-tva/Code` and build it locally.
```
(venv) $ cd CTL
(venv) $ python setup.py build_ext --inplace
(venv) $ python setup.py install
(venv) $ cd ..
```
