# FEM
<img src="assets/example.gif"/>

Python CLI application to calculate and animate waves in 1D and 2D, using the weak formulation of the wave equation and the finite element method to numerically calculate the solution for any user-defined continuous function where $f(R) = 0$.

reference paper:
- https://studenttheses.uu.nl/bitstream/handle/20.500.12932/29861/thesis.pdf?sequence=2
- https://github.com/jeverink/BachelorsThesis

## Installation
- *python 3.10 or higher is required.*
- *if you want to save animations with `--save` flag, the software **`ffmpeg`** is required and must be in system path.*
    - *if on mac, just download with homebrew: `brew install ffmpeg`*
1. clone repo:
```shell
git clone git@github.com:LocoForHotcocoa/FEM-on-2D-membrane.git
```
2. install package with pip or poetry
```shell

cd FEM-on-2D-membrane

# 1. with pip (recommended to use a venv to not mess up your local python env)
python3 -m venv .venv
source .venv/bin/activate
pip install .

# 2. or with poetry (venv is handled automatically)
poetry install
```

## How to Run
This supports animation on a 1D line: `fem line`, and a 2D circular membrane: `fem circle`

```shell
# get help menus:
fem --help
fem circle --help

# example function call:
fem circle 'cos(3*arctan2(y,x))*sin(pi*sqrt(x**2+y**2))' --num-elements 500 --speed 4 --iterations 10000
```

*TODO*: 
- figure out a way to separate wave calc logic and plotting logic


