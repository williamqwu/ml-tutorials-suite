If you're running the demos from this repo on [Google Colab](https://colab.research.google.com/), you may try this workflow below. It uses the first module `1_linear_classifier.ipynb` as an example.

1. On [colab main page](https://colab.research.google.com/), select File -> New Notebook in Drive
2. Clone the repo and enter the workspace
```bash
!git clone https://github.com/williamqwu/ml-tutorials-suite ml_tutorials
%cd ml_tutorials
!pwd
!ls
```
3. Import the local libraries via this block below, instead of `import lib.linear_classifier as LC` and `import infra.plot as plot` as written in the original notebook:
```bash
import importlib.util
import os

module_path = os.path.join(os.getcwd(), "lib", "linear_classifier.py")
spec = importlib.util.spec_from_file_location("linear_classifier", module_path)
LC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LC)

module_path = os.path.join(os.getcwd(), "infra", "plot.py")
spec = importlib.util.spec_from_file_location("plot", module_path)
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)
```
