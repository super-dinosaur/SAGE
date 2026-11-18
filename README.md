# commtree — Three-Agent Communication-Decision Dataset

DataTree builder for learning when agent A should **NO_COMM / ASK_B / ASK_C**.
It generates scenarios, branches A's decisions, estimates each action's value with K rollouts, and
exports SFT / DPO / RL data at any λ without new model calls.

* Design (English): [docs/DESIGN.md](docs/DESIGN.md)

```bash
python3 -m commtree build --config configs/mock.yaml --out runs/mock     # offline, seconds
python3 -m commtree export --run runs/mock --lam 0.05 0.5
python3 -m commtree check --run runs/mock
python3 -m commtree split --run runs/mock --ratios 0.8 0.1 0.1
python3 -m unittest discover -s tests -v
```

Standard library only (PyYAML for `.yaml` configs). Python ≥ 3.9.
