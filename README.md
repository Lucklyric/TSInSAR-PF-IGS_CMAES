# IGS-CMAES: A Two-Stage Optimization for Ground Deformation and DEM Error Estimation in Time Series InSAR Data

# pl_igscmaes
Recent refactored code with snamekae, hydra, pytorhc-lightening


### Sim experiments
```bash
snakemake -s workflow/run_sim_fix_exp.snake.py -j 1 --config processing_root=/quobyte/dev_machine_learning/pf/igscmaes/sim -p
```

### Real experiments
generate ps

```bash
snakemake -s workflow/gen_ps.snake.py -j 8 --config processing_root=/quobyte/dev_machine_learning/pf/igscmaes -p
```

run real
```bash
snakemake -s workflow/run_real_ps_exp.snake.py -j 1 --config processing_root=/quobyte/dev_machine_learning/pf/igscmaes/real -p
```
