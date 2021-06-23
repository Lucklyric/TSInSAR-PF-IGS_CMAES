import os

processing_root = config['processing_root']
solvers = ['igscmaes']
# Put your DB config list
dbs = [
]

rule all:
    input:
        # expand("{}/{solver}_{datset}_sim_out.json".format(processing_root), solver=solvers, dataset=dbs)
        expand("{processing_root}/{solver}_{dataset}_out.json", solver=solvers, dataset=dbs, processing_root=processing_root)

rule gen_fix_sim_signals:
    output: '{}/sim_signals/sim.npy'.format(processing_root)
    params:
        num_mr=30,
        num_he=60,
        min_mr=-26,
        max_mr=26,
        max_he=200,
        min_he=-200,
        seed=1234
    run:
        from pl_igscmaes.data.utils import gen_sim_mr_he
        import os
        import numpy as np
        os.makedirs(os.path.dirname(str(output)),exist_ok=True)
        np.random.seed(params.seed)
        sim_signals = gen_sim_mr_he(params.num_mr, params.num_he, params.min_mr, params.max_mr,params.min_he, params.max_he)
        np.save(str(output), sim_signals)

rule run_sim_exp:
    input:
        '{}/sim_signals/sim.npy'.format(processing_root)
    output:
        '{processing_root}/{solver}_{dataset}_out.json'
    shell:
        'python -m pl_igscmaes.run processing_dir={processing_root} solver={wildcards.solver} db={wildcards.dataset} db.sim_signals_path={input} output_path={output}' 


