import os

processing_root = config['processing_root']
solvers = ['igscmaes']
# Put your DB config list
dbs = [
]

rule all:
    input:
        expand("{processing_root}/real/{solver}_{dataset}/out", solver=solvers, dataset=dbs, processing_root=processing_root),

rule allps:
    input:
        expand("{processing_root}/psv2/{dataset}/ps.json", dataset=dbs, processing_root=processing_root)

rule run_real_exp:
    input:
        '{processing_root}/psv2/{dataset}/ps.json'
    output:
        '{processing_root}/real/{solver}_{dataset}/out'
    shell:
        'python -m pl_igscmaes.run processing_dir={processing_root}/real/{wildcards.solver}_{wildcards.dataset} solver={wildcards.solver} db={wildcards.dataset} db.ps_coords_path={input} output_path={output}' 

rule gen_real_ps:
    output:
        '{processing_root}/psv2/{dataset}/ps.json'
    shell:
        'python -m pl_igscmaes.run_genps processing_dir={processing_root}/psv2/{wildcards.dataset} db={wildcards.dataset} output_path={output}' 
