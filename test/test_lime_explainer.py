import unittest
import h2o
import numpy as np
import pandas as pd
import xgboost as xgb
from lime_xgboost.lime_explainer import LIMEExplainer


class TestLIMEExplainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # frame of numerics to be used for multiple tests
        frame = pd.read_csv('../data/train.csv')
        frame = frame.select_dtypes(include=['int16', 'int32', 'int64',
                                             'float16', 'float32', 'float64'])

        # local constants
        seed = 12345
        row_id = 'Id'
        y = 'SalePrice'
        X = [name for name in frame.columns if name not in [y, row_id]]

        frame[y] = np.log(frame[y])

        # split data
        np.random.seed(seed)  # set random seed for reproducibility
        split_ratio = 0.7     # 70%/30% train/test split
        split = np.random.rand(len(frame)) < split_ratio
        tr_frame = frame[split]
        v_frame = frame[~split]

        # train xgboost
        ave_y = tr_frame['SalePrice'].mean()

        # xgboost uses svmlight data structure
        dtrain = xgb.DMatrix(tr_frame[X],
                             tr_frame[y])
        dvalid = xgb.DMatrix(v_frame[X],
                             v_frame[y])

        # xgboost tuning parameters
        params = {
            'objective': 'reg:linear',
            'booster': 'gbtree',
            'eval_metric': 'rmse',
            'eta': 0.001,
            'subsample': 0.1,
            'colsample_bytree': 0.8,
            'max_depth': 5,
            'reg_alpha': 0.01,
            'reg_lambda': 0.0,
            'base_score': ave_y,
            'silent': 0,
            'seed': 12345,
        }

        # watchlist is used for early stopping
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

        # train model here so it is only trained once
        # instead of at beginning of every test function
        bst = xgb.train(params,
                        dtrain,
                        10000,
                        evals=watchlist,
                        early_stopping_rounds=50,
                        verbose_eval=True)

        # save model to be reused in each test function
        bst.save_model('test.model')

        h2o.init()

    def setUp(self):

        # setup instance constants for each function

        self.seed = 12345
        self.row_id = 'Id'

        bst = xgb.Booster({'nthread': 4})
        bst.load_model('test.model')
        self.model = bst

        self.frame = pd.read_csv('../data/train.csv')
        self.frame = self.frame.select_dtypes(include=['int16', 'int32',
                                                       'int64', 'float16',
                                                       'float32', 'float64'])

        self.y = 'SalePrice'
        self.frame[self.y] = np.log(self.frame[self.y])

        self.X = [name for name in self.frame.columns
                  if name not in [self.y, self.row_id]]

    def test_generate_local_sample(self):

        row = self.frame.iloc[0, :]
        explainer = LIMEExplainer(training_frame=self.frame,
                                  X=self.X, model=self.model)
        local_sample = explainer._generate_local_sample(row)
        N = explainer.N
        self.assertAlmostEqual(local_sample.loc[:, 'MSSubClass'].std(),
                               self.frame.loc[:, 'MSSubClass'].std(), places=0)
        self.assertEqual(local_sample.shape[0], N)
        del explainer

    def test_score_local_sample(self):

        row = self.frame.iloc[0, :]
        explainer = LIMEExplainer(training_frame=self.frame,
                                  X=self.X, model=self.model)
        local_sample = explainer._generate_local_sample(row)
        scored_local_sample = \
            explainer._score_local_sample(local_sample,
                                          row[local_sample.columns])
        self.assertEqual(scored_local_sample.shape[1],
                         local_sample.shape[1] + 1)
        self.assertEqual(scored_local_sample.columns[-1], 'predict')
        del explainer

    def test_calculate_distance_weights(self):

        row = self.frame.iloc[0, :]
        explainer = LIMEExplainer(training_frame=self.frame,
                                  X=self.X, model=self.model)
        local_sample = explainer._generate_local_sample(row)
        scored_local_sample = \
            explainer._score_local_sample(local_sample,
                                          row[local_sample.columns])
        weighted_scored_local_sample = \
            explainer._calculate_distance_weights(0,
                                                  scored_local_sample)
        self.assertEqual(weighted_scored_local_sample.shape[1],
                         local_sample.shape[1] + 2)
        self.assertEqual(weighted_scored_local_sample.columns[-1], 'distance')
        del explainer

    def test_calculate_distance_weights(self):

        row = self.frame.iloc[0, :]
        explainer = LIMEExplainer(training_frame=self.frame,
                                  X=self.X, model=self.model)
        local_sample = explainer._generate_local_sample(row)
        scored_local_sample = \
            explainer._score_local_sample(local_sample,
                                          row[local_sample.columns])

        weighted_scored_local_sample = \
            explainer._calculate_distance_weights(0,
                                                  scored_local_sample)
        self.assertEqual(weighted_scored_local_sample.shape[1],
                         local_sample.shape[1] + 2)
        self.assertEqual(weighted_scored_local_sample.columns[-1], 'distance')
        del explainer

    def test_discretize_numeric(self):

        row = self.frame.iloc[0, :]
        explainer = LIMEExplainer(training_frame=self.frame,
                                  X=self.X, model=self.model)
        N = explainer.N
        local_sample = explainer._generate_local_sample(row)
        scored_local_sample = \
            explainer._score_local_sample(local_sample,
                                          row[local_sample.columns])
        weighted_scored_local_sample = \
            explainer._calculate_distance_weights(0,
                                                  scored_local_sample)
        discretized_weighted_scored_local_sample = \
            explainer._discretize_numeric(weighted_scored_local_sample)
        self.assertEqual(discretized_weighted_scored_local_sample.shape,
                         (N, local_sample.shape[1] + 2))
        self.assertEqual(discretized_weighted_scored_local_sample[
                             self.X].dtypes.unique()[0], 'category')
        del explainer

    def test_regress_w_discretize(self):

        row = self.frame.iloc[0, :]
        explainer = LIMEExplainer(training_frame=self.frame,
                                  X=self.X, model=self.model)
        local_sample = explainer._generate_local_sample(row)
        scored_local_sample = \
            explainer._score_local_sample(local_sample,
                                          row[local_sample.columns])
        weighted_scored_local_sample = \
            explainer._calculate_distance_weights(0,
                                                 scored_local_sample)
        discretized_weighted_scored_local_sample = \
            explainer._discretize_numeric(weighted_scored_local_sample)

        disc_row = pd.DataFrame(columns=self.X)
        for name in self.X:
            disc_row[name] = pd.cut(pd.Series(row[name]),
                                    bins=explainer.bins_dict[name])

        lime = explainer._regress(discretized_weighted_scored_local_sample,
                                  h2o.H2OFrame(disc_row))
        
        self.assertTrue(explainer.discretize)
        self.assertIsNotNone(lime)
        self.assertAlmostEqual(0.9121119123027541, explainer.lime_r2)

        del explainer

    def test_regress_w_o_discretize(self):

        row = self.frame.iloc[0, :]
        explainer = LIMEExplainer(training_frame=self.frame,
                                  X=self.X, model=self.model,
                                  discretize=False)
        local_sample = explainer._generate_local_sample(row)
        scored_local_sample = \
            explainer._score_local_sample(local_sample,
                                          row[local_sample.columns])
        weighted_scored_local_sample = \
            explainer._calculate_distance_weights(0,
                                                  scored_local_sample)
        lime = explainer._regress(weighted_scored_local_sample,
                                  h2o.H2OFrame(pd.DataFrame(row).T))
        self.assertFalse(explainer.discretize)
        self.assertIsNotNone(lime)
        self.assertAlmostEqual(0.9649889709835218, explainer.lime_r2)
        del explainer

    def test_plot_local_contrib_w_discretize(self):

        row = self.frame.iloc[0, :]

        explainer = LIMEExplainer(training_frame=self.frame,
                                  X=self.X, model=self.model)
        local_sample = explainer._generate_local_sample(row)
        scored_local_sample = \
            explainer._score_local_sample(local_sample,
                                          row[local_sample.columns])
        weighted_scored_local_sample = \
            explainer._calculate_distance_weights(0,
                                                  scored_local_sample)
        discretized_weighted_scored_local_sample = \
            explainer._discretize_numeric(weighted_scored_local_sample)

        disc_row = pd.DataFrame(columns=self.X)
        for name in self.X:
            disc_row[name] = pd.cut(pd.Series(row[name]),
                                    bins=explainer.bins_dict[name])

        explainer.lime = \
            explainer._regress(discretized_weighted_scored_local_sample,
                               h2o.H2OFrame(disc_row))

        self.assertEqual(explainer.reason_code_values.shape, (36, 2))

    def test_plot_local_contrib_w_o_discretize(self):

        row = self.frame.iloc[0, :]

        explainer = LIMEExplainer(training_frame=self.frame,
                                  X=self.X, model=self.model)
        local_sample = explainer._generate_local_sample(row)
        scored_local_sample = \
            explainer._score_local_sample(local_sample,
                                          row[local_sample.columns])
        weighted_scored_local_sample = \
            explainer._calculate_distance_weights(0,
                                                  scored_local_sample)
        explainer.lime = \
            explainer._regress(weighted_scored_local_sample,
                               h2o.H2OFrame(pd.DataFrame(row).T))

        self.assertEqual(explainer.reason_code_values.shape, (26, 2))

    def test_explain_w_discretize(self):

        row_id = 0
        explainer = LIMEExplainer(training_frame=self.frame,
                                  X=self.X, model=self.model)
        explainer.explain(row_id)
        self.assertAlmostEqual(explainer.lime_pred,
                               explainer.lime.coef()['Intercept'] +
                               explainer.reason_code_values['Local Contribution'].sum())

    def test_explain_w_o_discretize(self):

        row_id = 0
        explainer = LIMEExplainer(training_frame=self.frame,
                                  X=self.X, model=self.model,
                                  discretize=False)
        explainer.explain(row_id)
        self.assertAlmostEqual(explainer.lime_pred,
                               explainer.lime.coef()['Intercept'] +
                               explainer.reason_code_values['Local Contribution'].sum())

    def test_explain_w_discretize_w_o_intercept(self):

        row_id = 0
        explainer = LIMEExplainer(training_frame=self.frame,
                                  X=self.X, model=self.model,
                                  discretize=True, intercept=False)
        explainer.explain(row_id)
        self.assertAlmostEqual(explainer.lime.coef()['Intercept'], 0)
        self.assertAlmostEqual(explainer.lime_pred,
                               explainer.lime.coef()['Intercept'] +
                               explainer.reason_code_values['Local Contribution'].sum())

    def tearDown(self):
        self.frame = None
        self.X = None


if __name__ == '__main__':

    unittest.main()
    h2o.cluster().shutdown(prompt=False)
