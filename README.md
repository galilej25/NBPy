# NBPy
NBPy: Network Based Python-statistics for Mixed Effects Models



#### Test example (time)

- File `./data/sample_input_1.csv` has 28 nodes and 48 subjects
- File `./data/sample_input_2.csv` has 116 nodes and 117 subjects

```bash
time  python run_nbpy_linear_model.py --file-name="./data/sample_input_1.csv" --n-nodes=28 --alternative="two.sided" --thr-p=0.01 --n-cores=1 --n-perm=10
```
