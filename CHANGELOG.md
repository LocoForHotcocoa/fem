## v0.2.1 (2025-11-24)

### Refactor

- use pathlib.Path instead of str from the start
- use pathlib instead of string concat

### Perf

- update FEM_linear with similar logic from circle solution
- calculate static inverse matrix before iteration, instead of during
- calculate arbritrary function for the entire mesh at the same time using np.ndarray

## v0.2.0 (2025-08-18)

### Feat

- **FEM_circle.py**: add fps calculation to this script too

### Fix

- **circlyboi**: change math.floor to math.ceil for step_size calc to prevent divide by 0 error!!
- **FEM_linear.py**: fix framerate and add explanation of framerate calcalation. will do the same for FEM_circle

## v0.1.0 (2025-04-01)

### Feat

- **FEM_linear**: add arbitrary func logic to FEM_linear and add it as command in main.py
- **main.py**: add typer to simplify CLI, and finally figure out a way to take in user defined functions!

### Fix

- **main.py**: fix default constants
- **FEM_methods.py**: tried to estimate the spacing of boundary points based on numTriangles, but idk if chatgpt is telling the truth

### Refactor

- **oldsrc**: got rid of old stuff. not the focus of the project anymore!
- **circlyboi**: create a pip package so you can just install with pip install !!
- **tools**: remove unnecessary files/dirs
- **animations**: remove animations directory from repo
