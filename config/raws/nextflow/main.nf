#!/usr/bin/env nextflow

include { MAKE_INIS; RUN_MODEL } from './modules/subbasin.nf'


workflow {

    def n = file(params.domains_csv).splitCsv(header: true).size()

    // coerce before comparing: params set on the command line arrive as strings
    def threads = params.threads as int
    def budget = params.executor_cpus as int
    if (n * threads > budget) {
        error "${n} sub-basins x ${threads} threads = ${n * threads} cores, but " +
              "executor_cpus is ${budget}. They all have to run at once -- lower threads " +
              "to ${(budget / n) as int} or raise executor_cpus."
    }

    log.info "${n} sub-basins -> ${params.outdir}"

    RUN_MODEL(MAKE_INIS().flatten().map { ini -> tuple(ini.simpleName, ini) })
}
