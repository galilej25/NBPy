from optparse import OptionParser
from nbpy.nbpy_linear_model import NBR


def parse_options():

    parser = OptionParser()
    parser.add_option("--file-name", dest="file_name", help="Name of the input fMRI file")
    parser.add_option("--n-nodes", dest="n_nodes", help="Number of fMRI nodes")
    parser.add_option("--predictor-cols", dest="predictor_cols", help="Columns selected as exog")
    parser.add_option("--alternative", dest="alternative", help="P/T value selection")
    parser.add_option("--thr-p", dest="thr_p", help="p value threshold")
    parser.add_option("--thr-t", dest="thr_t", help="t value threshold")
    parser.add_option("--n-cores", dest="n_cores", help="Number of cores for parallelization")
    parser.add_option("--n-perm", dest="n_perm", help="Number of permutations")

    return parser.parse_args()


(options, args) = parse_options()


def main():

    nbr_model = NBR(
        fname=options.file_name,
        n_nodes=int(options.n_nodes),
        mod="",
        alternative=options.alternative,
        thr_p=float(options.thr_p),
        # thr_t=float(options.thr_t),
    )

    nbr_model.run_linear_mixture_models(
        n_cores=int(options.n_cores),
        n_perm=int(options.n_perm),
    )

    nbr_model.measured_observations.to_csv('tmp.csv', index=False)


if __name__ == "__main__":
    main()