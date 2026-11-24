# Open isogeny lifts

SageMath (version 10.10) implementation accompanying *Computing Elliptic-Curve Endomorphism Traces via Open Isogeny Lifts*. The code computes the trace of a decoded isogeny chain by open lifting and verifies a prescribed trace at reduced precision. It uses the lifting routines of Panny, Robert and Sferlazza and includes their closed-lifting method for comparison. [PRS implementation](https://gitlab.inria.fr/isogenies/lifting_traces)

## How to run

Compute the trace on Houben's published base cycle using open lifting with batched Vélu evaluation. At the default precision `s = 9`, the expected trace is **29171033**. [Source of the base cycle](https://github.com/houbenmr/LargeDiscriminantOrientations/tree/02389279e5a265cffec2c9013646c00fa5b5e654)

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
sage -python src/run.py trace --method open-batched --out runs/trace.json
```

Compare the endpoint recurrences at precisions 9, 8 and 7, then verify the prescribed trace using the fixed-width runs from that comparison.

```sh
sage -python src/run.py paired --out runs/paired.json
sage -python src/run.py verify --paired runs/paired.json --out runs/verification.json
```

Run the other three implementations on the same base cycle.

```sh
sage -python src/run.py trace --method prs --out runs/prs.json
sage -python src/run.py trace --method prs-batched --out runs/prs-batched.json
sage -python src/run.py trace --method open --repeat 3 --out runs/open.json
```
