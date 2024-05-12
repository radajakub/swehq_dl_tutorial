# swehq_dl_tutorial

Neural Networks and PyTorch tutorial

## Setup

I recommend using VSCode as IDE, it is perfect for interactive Python and Jupyter notebooks and also potentially for remote development (I will show how at the beginning).
Also we will need the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) to have interactive notebooks, which are widely used in AI development.

### Python environment

I use [Pyenv](https://github.com/pyenv/pyenv) as a version and virtual environment manager (their virtualenv plugin) for Python.
One of the advantages of Pyenv is that it can automatically activate environments when entering a directory with a `.python-version` file.
Also, the environment can be activated anywhere in the filesystem without knowing the path, so it can reuse the same environment for multiple projects.

Here are the steps to make this repository work if pyenv is already set up:

- `pyenv install <version>` (if it is not already installed)
  - I used Python 3.11.5 but other versions might work as well
- `pyenv virtualenv <version> swehq_dl_tutorial`
- `pyenv activate swehq_dl_tutorial` (if not activate already)
- `pip install -r requirements.txt`

To test the environment run `python -c "import torch; print(torch.__version__)"` and it should execute without errors.

It is important to tell VSCode which environment is being used.
This can be done by a button in the bottom right corner when a `.py` file is open or by `Cmd+Shift+P` and typing `Python: Select Interpreter`.
From a dropdown menu select the created virtual environment (it might be necessary to refresh the list if the environment is there).

If other option to create the virtual environment is used, it is still necessary to select the interpreter in VSCode, so that the interactive notebooks and files work with the correct environment.
