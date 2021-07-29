import signal
import numpy as np
import pandas as pd
import scipy.stats
import multiprocessing as mp

from sklearn import linear_model
from collections import OrderedDict


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def fit_linear_model(X, y):
    """

    :param X:  input variables
    :param y:  dependent variables
    :return:
    """

    # Step 1: train model
    lr_model = linear_model.LinearRegression().fit(
        X,
        y
    )
    # Step 2: prediction
    predictions = lr_model.predict(X)

    # Step 3: pval and tval calculation

    # 3a: get coefficients and stack them with intercept
    lr_parameters = np.vstack((lr_model.intercept_.T, lr_model.coef_.T)).T

    # 3b: append 1s to input data (for intercept)
    X_intercept = np.append(np.ones((len(X), 1)), X, axis=1)
    # 3b: get mean squared error between true values and predictions and scale it
    mse = np.sum((y - predictions) ** 2, axis=0) / (X_intercept.shape[0] - X_intercept.shape[1])

    # 3c: calculate invariant of input dataset (this might be tricky to do for larger matrix)
    X_inv = np.linalg.inv(np.dot(X_intercept.T, X_intercept)).diagonal()

    # 3d: estimate variance and standard deviation to calculate tvals
    var = np.dot(mse.values.reshape(mse.shape[0], 1), X_inv.reshape(1, X_inv.shape[0]))
    std = np.sqrt(var)
    tvals = lr_parameters / std

    # 3e: calculate pvalues
    pvals = [2 * (1 - scipy.stats.t.cdf(np.abs(i), (X_intercept.shape[0] - X_intercept.shape[1]))) for i in tvals]

    # 3f: save it in dataframe
    index = ['Intercept'] + X.columns.tolist()
    columns = ['tvals_%s' % c for c in y.columns] + ['pvals_%s' % c for c in y.columns]
    df_result = pd.DataFrame(
        np.vstack((tvals, pvals)).T,
        columns=columns,
        index=index,
    ).iloc[1:]

    return df_result


def get_observations(X, y, n_nodes, thr_t, thr_p, alternative, ci, id_perm):
    """

    :return:
    """

    # Get linear models
    lms = fit_linear_model(X, y)

    df_tvals = lms[[col for col in lms.columns if 'tvals_' in col]]
    df_pvals = lms[[col for col in lms.columns if 'pvals_' in col]]
    qt_2 = scipy.stats.t.ppf(1 - thr_p / 2, X.shape[0] - lms.shape[0] - 1)
    qt = scipy.stats.t.ppf(1 - thr_p, X.shape[0] - lms.shape[0] - 1)

    if thr_t is not None:
        # calculate strength
        strength = (df_tvals.abs() - abs(thr_t)) * (1 - 2 * (df_tvals < 0))

        if alternative == "two.sided":
            edges = df_tvals.abs() > thr_t
        elif alternative == "less":
            edges = df_tvals < thr_t
        else:
            edges = df_tvals > thr_t
    else:
        if alternative == "two.sided":
            edges = df_pvals < thr_p
            strength = (df_tvals.abs() - qt_2) * (1 - 2 * (df_tvals < 0))
        elif alternative == "less":
            edges = pd.DataFrame((df_pvals < 2 * thr_p).values & (df_tvals < 0).values, columns=df_pvals.columns,
                                 index=df_pvals.index)
            strength = (df_tvals.abs() - qt) * (1 - 2 * (df_tvals < 0))
        else:
            edges = pd.DataFrame((df_pvals < 2 * thr_p).values & (df_tvals > 0).values, columns=df_pvals.columns,
                                 index=df_pvals.index)
            strength = (df_tvals.abs() - qt) * (1 - 2 * (df_tvals < 0))

    # FIXME: this should be iteration over DF index
    observations = []
    for idx, var in enumerate(edges.index):
        e = np.where(edges.loc[var] == 1)[0]
        ci_row = ci[e, 0]
        ci_col = ci[e, 1]
        s = strength.loc[var].values[e]

        # get components
        # FIXME: Not sure if this is correct, even from R code!
        # Can this part of code be any faster? Do we need to iterate over edges or no?
        c = -1 * np.ones(len(e))
        # define a matrix of components
        components = np.tile(np.arange(n_nodes), (lms.shape[0], 1)).T
        if id_perm == -1 and thr_p is not None:
            for ide, ve in enumerate(e):
                max_lab = np.max(components[ci[ve, :], idx])
                min_lab = np.min(components[ci[ve, :], idx])
                if np.sum(c == max_lab) > 0:
                    c[c == max_lab] = min_lab
                pre_idx = np.where(components[:, idx] == max_lab)
                components[pre_idx, idx] = min_lab
                c[ide] = min_lab
        else:
            for ide, ve in enumerate(e):
                min_lab = np.min(components[ci[ve, :], idx])
                components[ci[ve, :], idx] = min_lab
                c[ide] = min_lab

        d_stack = np.vstack((e, ci_row, ci_col, s, c)).T
        df = pd.DataFrame(d_stack, columns=["2Dcol", "3Drow", "3Dcol",  "strength", "components"])
        df['variable'] = var
        df['id_perm'] = id_perm
        observations.append(df)

    return pd.concat(observations, ignore_index=True)


class NBR(object):
    """
    NBR Linear Model (2D input)

    Parameters
    ----------

    """

    def __init__(
            self,
            fname,
            n_nodes,
            predictor_cols,
            mod,
            alternative,
            diag=False,
            thr_p=0.05,
            thr_t=None,
            nudist=False,
            exp_list=None,
            verbose=True,
        ):

        # FIXME: What about missing values, etc...
        try:
            # load data
            self.data = pd.read_csv(
                fname,
                sep=",",
                header=0
            ).apply(pd.to_numeric, errors='ignore')
            # convert dots from header into slash
            self.data.columns = [c.replace('.', '_') for c in self.data.columns]
        except FileNotFoundError:
            print('[error] File Not Found!')

        self.n_nodes = n_nodes
        self.predictor_cols = predictor_cols
        self.mod = mod
        self.alternative = alternative
        self.diag = diag
        self.thr_p = thr_p
        self.thr_t = thr_t
        self.nudist = nudist
        self.expList = exp_list
        self.verbose = verbose

        self._args_check()
        self._create_connection_indices()
        self._data_preprocessing()

    def _args_check(self):
        """
        Check correct type of arguments
        :return:
        """

        # check if diag is boolean
        if not isinstance(self.diag, bool):
            raise ValueError("[error] diag parameter must be boolean; got (diag=%r)" % self.diag)

        if not isinstance(self.n_nodes, int) and self.n_nodes > 0:
            raise ValueError("[error] n_nodes parameter must be positive non-zero integer; got (n_nodes=%r)" % self.n_nodes)

        if self.alternative not in ["two.sided", "less", "greater"]:
            raise ValueError("[error] alternative parameter is not correctly specified (options: two.sided, less, greater); got (alternative=%r) " % self.alternative)

        if (self.thr_p is None and self.thr_t is None) or (self.thr_t is not None and self.thr_p is not None):
            raise ValueError('Thresholds for P and T are incorrectly set; got(pval=%r, tval=%r)' % (self.thr_p, self.thr_t))

    def _create_connection_indices(self):
        """

        :return:
        """
        cn = np.vstack((np.where(np.triu(np.ones(self.n_nodes), k=0) == 0))).T
        cn[:, [0, 1]] = cn[:, [1, 0]]
        self.connection_indices = cn

    def _data_preprocessing(self):
        """

        :return:
        """
        # Step 2a: Separate data into exog and edog
        y = self.data.drop(columns=self.predictor_cols)
        X = self.data[self.predictor_cols]

        # Step 2b: create dummy values for categorical data in
        column_types = X.dtypes
        object_columns = column_types[column_types == 'object'].index
        if object_columns.shape[0] > 0:
            dummy_columns = pd.get_dummies(X[object_columns], drop_first=True)
            X = pd.concat([dummy_columns, X], axis=1).drop(columns=object_columns)
            # FIXME: we should read this from input, but atm we will keep it here as hardcoded
            X['Sex_m_Age'] = X['Sex_M'] * X['Age']
        else:
            X['Sex_m_Age'] = X['Sex'] * X['Age']

        self.X = X
        self.y = y

    def _linear_mixture_models_initial(self):
        """

        :return:
        """
        self.init_observations = get_observations(
            X=self.X,
            y=self.y,
            n_nodes=self.n_nodes,
            thr_t=self.thr_t,
            thr_p=self.thr_p,
            alternative=self.alternative,
            ci=self.connection_indices,
            id_perm=-1,
        )

    def _linear_mixture_models_with_permutations(self, n_cores=None, n_perm=None):

        if n_cores is None:
            n_cores = 1
        elif n_cores == -1:
            n_cores = mp.cpu_count()
        if not isinstance(n_cores, int) and n_cores > 0:
            raise ValueError("[error] n_core parameter must be positive non-zero integer; got (n_core=%r)" % n_cores)

        if n_perm is None:
            n_perm = 1
        if not isinstance(n_perm, int) and n_perm > 0:
            raise ValueError("[error] n_perm parameter must be positive non-zero integer; got (n_perm=%r)" % n_perm)

        pool = mp.Pool(n_cores, init_worker)
        results = []
        try:
            for id_perm in range(n_perm):
                # apply permutation
                X_ = self.X.sample(self.X.shape[0])
                results.append(pool.apply_async(
                    func=get_observations,
                    args=(
                        X_,
                        self.y,
                        self.n_nodes,
                        self.thr_t,
                        self.thr_p,
                        self.alternative,
                        self.connection_indices,
                        id_perm,
                    )
                ))
        except KeyboardInterrupt:
            print('[Exception: Keyboard] Terminate MultiProcessing')
            pool.terminate()
            pool.join()
        except Exception as e:
            print('[Exception: Unknown] Terminate MultiProcessing')
            pool.terminate()
            pool.join()
        finally:
            # Close and wait for the pool to finish
            pool.close()
            pool.join()
            df_obs_perm = pd.concat([r.get() for r in results], ignore_index=True)

        # FIXME: Avoid groupby + groupby again
        # prepare FWE values
        df_obs_perm_grp = df_obs_perm.groupby(by=['variable', 'id_perm'])[['strength', 'components']].apply(
            lambda x: pd.DataFrame(np.vstack((np.max(x.groupby(by=['components'])['strength'].sum()), np.max(x['components']))).T, columns=['max_sum_str', 'max_comp'])
        ).reset_index('id_perm')

        # iterate over variable
        df_final = []
        for variable, group in self.init_observations.groupby(by=['variable']):
            ncompFWE = group['components'].apply(
                lambda x: np.sum(df_obs_perm_grp.loc[variable]['max_comp'] > x) / n_perm
            ).to_dict(OrderedDict)
            # FIXME: end of conversion to dict
            strnFWE = group.groupby(by=['components'])['strength'].apply(
                lambda x: pd.DataFrame(
                    np.vstack(
                        (
                            np.sum(df_obs_perm_grp.loc[variable]['max_sum_str'] > np.sum(np.abs(x))) / n_perm,
                            np.sum(np.abs(x)),
                        )
                    ).T,
                    columns=['strnFWE', 'strn']
                )
            ).reset_index().drop(columns='level_1').set_index('components').to_dict(orient='dict')

            df = pd.DataFrame(
                np.vstack((list(strnFWE['strnFWE'].keys()), list(strnFWE['strnFWE'].values()), list(strnFWE['strn'].values()))).T,
                columns=['Component', 'strnFWE', 'strn']
            )
            df['variable'] = variable
            df_final.append(df)

        return pd.concat(df_final, ignore_index=True)

    def run_linear_mixture_models(self, n_cores=None, n_perm=None):
        """

        :param n_cores:
        :param n_perm:
        :return:
        """

        self._linear_mixture_models_initial()
        self._linear_mixture_models_with_permutations(n_cores=n_cores, n_perm=n_perm)