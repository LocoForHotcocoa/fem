# FEM (Finite Element Method)
<img src="assets/example.gif"/>

Python CLI application to calculate and animate waves in 1D and 2D, using the weak formulation of the wave equation and the finite element method to numerically calculate the solution for any user-defined continuous function where $f(R) = 0$.

reference paper:
- https://studenttheses.uu.nl/bitstream/handle/20.500.12932/29861/thesis.pdf?sequence=2
- https://github.com/jeverink/BachelorsThesis

## Installation
- *python 3.10 or higher is required.*
- *if you want to save animations with `--save` flag, the software **`ffmpeg`** is required and must be in system path.*
    - *if on mac, just download with homebrew: `brew install ffmpeg`*
- this project is managed with the [uv](https://docs.astral.sh/uv/guides/install-python/) Python package manager, which is the intended way to install this package.
1. clone repo:
```shell
git clone git@github.com:LocoForHotcocoa/fem.git
cd fem
```
2. install package:
```shell

# 1. install with uv
uv venv && uv sync
source .venv/bin/activate

# 2. or with plain pip (still recommended to use a venv to not mess up your local python env)
python3 -m venv .venv
source .venv/bin/activate
pip install .

# 3. to install dev packages with uv
uv sync --extra dev
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

## File Structure
```
circlyboi
├── __init__.py
├── FEM_circle.py
├── FEM_linear.py
├── main.py
└── parse_func.py
```

*TODO*: 
- figure out a way to separate wave calc logic and plotting logic into different files