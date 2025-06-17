## Codebase Structure Overview

```bash
.
├── 1_linear_classifier.ipynb        # Entry point for Module 1
├── 2_gradient_descent.ipynb         # Entry point for Module 2
├── 3_transformer_bc.ipynb           # Entry point for Module 3
├── 4_transformer_vision.ipynb       # Entry point for Module 4
│
├── infra/                           # Utility modules
│   ├── exp_tracker.py               # A minimal abstract interface
│   └── plot.py                      # Visualization utilities
├── lib/                             # Core algorithm implementations
│   ├── linear_classifier.py
│   ├── gradient_descent.py
│   ├── transformer_bc.py
│   └── transformer_vision.py
├── model/                           # Model definition
│   ├── perceptron.py
│   ├── tbc.py
│   └── vitfc.py
│
└── Readme.md
```
