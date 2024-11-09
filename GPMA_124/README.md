# GPMA

The code in this directory is "forked" from [github gpma_demo](https://github.com/desert0616/gpma_demo).

To execute:

```bash
python preparation.py
make
./gpma_demo bfs pokec.txt 0
```

## Update Log:

Add thrust libs ub gpma_pr.cuh.

Comment out `cErr(cudaDeviceSynchronize());` in device functions in gpma.cuh.