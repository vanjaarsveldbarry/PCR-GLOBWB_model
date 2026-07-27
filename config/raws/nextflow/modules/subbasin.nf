process MAKE_INIS {

    publishDir "${params.outdir}/_inis", mode: 'copy'

    output:
    path "*.ini"

    script:
    """
    python3 ${moduleDir}/make_inis.py \\
        --template-ini     ${params.template_ini} \\
        --domains          ${params.domains_csv} \\
        --outdir           ${params.outdir} \\
        --upstream-timeout ${params.upstream_discharge_timeout}
    """
}


process RUN_MODEL {

    tag "${name}"

    input:
    tuple val(name), path(ini)

    script:
    """
    export PCRASTER_NR_WORKER_THREADS=${task.cpus}

    pixi run --manifest-path ${params.model_directory}/pixi.toml \\
        python ${params.model_directory}/model/deterministic_runner.py ${ini}
    """
}
