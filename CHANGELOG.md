## Unreleased

### Feat

- **FEM_circle.py**: add fps calculation to this script too
- **FEM_linear**: add arbitrary func logic to FEM_linear and add it as command in main.py
- **main.py**: add typer to simplify CLI, and finally figure out a way to take in user defined functions!

### Fix

- **circlyboi**: change math.floor to math.ceil for step_size calc to prevent divide by 0 error!!
- **FEM_linear.py**: fix framerate and add explanation of framerate calcalation. will do the same for FEM_circle
- **main.py**: fix default constants
- **FEM_methods.py**: tried to estimate the spacing of boundary points based on numTriangles, but idk if chatgpt is telling the truth

### Refactor

- **oldsrc**: got rid of old stuff. not the focus of the project anymore!
- **circlyboi**: create a pip package so you can just install with pip install !!
- **tools**: remove unnecessary files/dirs
- **animations**: remove animations directory from repo
