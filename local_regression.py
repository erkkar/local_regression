"""Module for modelling related functions and classes"""
from multiprocessing import Pool

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

MIN_DATA_POINTS = 20
SAMPLE_INCREASE = 10
TIME_WINDOW_INCREASE = pd.Timedelta("24h")
TEST_SIZE = 0.2
N_STRATIFY_QUANTILES = 3


def _timeslice_length(window: slice) -> pd.Timedelta:
    """Calculate length of a timeslice"""
    return window.stop - window.start


class _LocalModel(object):
    """Class for local regression model instances"""

    def __init__(
        self,
        regression: LinearRegression | Pipeline,
        score: float,
        window: slice,
        traindata: pd.DataFrame,
        testdata: pd.DataFrame,
    ):
        self.regression = regression
        self.test_score = score
        self.window = window
        self.traindata = traindata
        self.testdata = testdata
        self.predict = self.regression.predict

    def get_window_width(self):
        """Get the window width of this model"""
        return _timeslice_length(self.window)


class LocalRegression(object):
    """Class for making localized linear regression models

    Attributes:
        exog (pd.DataFrame): Exogenous data
        endog (pd.Series): Endogenous data
        modeldata (pd.DataFrame): All model data
        stratifier (pd.Series): Stratifier variable
        score (pd.Series): Coefficients of determination of the models
        min_samples (int): Minimum number of samples to include.
        min_score (float): Minimum model score.
        max_window_width (pd.Timedelta): Maximum width of data window.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        exog: list[str],
        endog: str,
        sample_weight: str = None,
        test_size: float = TEST_SIZE,
        dates: list[pd.Timestamp] = None,
        min_samples: int = MIN_DATA_POINTS,
        min_score: float = None,
        max_window_width: str | pd.Timedelta = None,
        stratifier: str = None,
        n_proc=1,
    ):
        """Initilize local regression

        Args:
            data (pd.DataFrame): Modelling data
            exog (list): List of names to use as exgogenous variables
            endog (str): Name of endogenous variable
            sample_weight (str, optional): Name of sample weight vector
            dates (list[pd.Timestamp], optional): List of dates to fit the models.
            min_samples (int, optional): Minimum samples to include,
                defaults to MIN_DATA_POINTS.
            min_score (float, optional): Minimum score, defaults to None.
            max_window_width (str | pd.Timedelta, optional): Maximum width of data
                window, defaults to None.
            stratifier (str, optional): Use this variable to for splitting
                to train and test sets in a stratified fashien, defaults to None.
            n_proc (int, optional): Number of (sub)processes, defaults to 1.
        """

        # Set of observations
        self.modeldata = data.copy()
        self.modeldata["weight"] = (
            self.modeldata[sample_weight] if sample_weight is not None else 1
        )

        # Exogenous and endogenous variables
        self.exog = self.modeldata[exog]
        self.endog = self.modeldata[endog]

        if stratifier is not None:
            self.stratifier = self.modeldata[stratifier]
        else:
            self.stratifier = None

        # Other attributes
        self._test_size = test_size
        self._dates = dates if dates is not None else self._get_dates()
        self.model_windows = None
        self._models = None
        self.score = None
        self._traindata = None
        self._testdata = None
        self.min_samples = min_samples
        self.min_score = min_score
        self.max_window_width = (
            pd.Timedelta(max_window_width) if max_window_width is not None else None
        )
        self.n_proc = n_proc

    def _get_dates(self):
        """Dates for which models are fitted

        Only includes dates where there are exogenous values available
        """
        dates = pd.to_datetime(self.exog.dropna().index.date).unique().sort_values()
        dates.name = "modeldate"
        return dates

    def _get_window(
        self, start: pd.Timestamp, stop: pd.Timestamp, min_samples: int
    ) -> slice:
        """Set modelling window start and stop to smallest/largest
        *existing* time indices
        """
        # Check number of data points and widen the window if necessary
        while len(self.modeldata.dropna().loc[start:stop]) < min_samples:
            start -= TIME_WINDOW_INCREASE
            stop += TIME_WINDOW_INCREASE
        return slice(*self.modeldata.loc[start:stop].index[[0, -1]])

    def _window_width_ok(self, window) -> bool:
        return self.max_window_width is None or (
            _timeslice_length(window) <= self.max_window_width
        )

    def get_window_widths(self) -> pd.Series:
        """Get width of each model window as a pandas Series"""
        return pd.Series(
            {date: _timeslice_length(mdl.window) for date, mdl in self._models.items()}
        )

    def _stratify(self, data, n_quantiles=N_STRATIFY_QUANTILES):
        try:
            classes = pd.qcut(
                data[self.stratifier.name], n_quantiles, duplicates="raise"
            )
        except ValueError as err:
            if "Bin edges must be unique" in str(err):
                classes = None
            else:
                raise
        else:
            # Reduce number of quantiles if less than 2 samples in some group
            if data[self.stratifier.name].groupby(classes).count().min() < 2:
                classes = self._stratify(data, n_quantiles=n_quantiles - 1)

        return classes

    def fit(self):
        """Fit the models"""
        # Fit local models and get R2 scores
        if self.n_proc > 1:
            with Pool(self.n_proc) as pool:
                models = pool.map(self._fit_and_test, self._dates)
        else:
            models = map(self._fit_and_test, self._dates)

        self._models = {
            date: mdl for date, mdl in zip(self._dates, models) if mdl is not None
        }

        # Collect modelling data
        self.score = pd.Series(
            {date: mdl.test_score for date, mdl in self._models.items()}
        )
        self._traindata = pd.concat(
            {date: mdl.traindata for date, mdl in self._models.items()},
            names=[self._dates.name],
        )
        self._testdata = pd.concat(
            {date: mdl.testdata for date, mdl in self._models.items()},
            names=[self._dates.name],
        )
        return self

    def _fit_and_test(self, date) -> _LocalModel | None:
        window = self._get_window(date, date + pd.Timedelta("24h"), self.min_samples)
        if not self._window_width_ok(window):
            return None

        while True:
            # Get data for this model
            df = self.modeldata.dropna().loc[window]  # pylint: disable=invalid-name

            # Get training and testing arrays
            (
                X_train,  # pylint: disable=invalid-name
                X_test,  # pylint: disable=invalid-name
                y_train,
                y_test,
                weight_train,
                weight_test,
            ) = train_test_split(
                df[self.exog.columns],
                df[self.endog.name],
                df["weight"],
                test_size=self._test_size,
                random_state=0,
                stratify=(self._stratify(df) if self.stratifier is not None else None),
            )

            # Fit the model
            pipe = Pipeline(
                [("scaler", StandardScaler()), ("linregress", LinearRegression())]
            ).fit(
                X_train,
                y_train,
                linregress__sample_weight=weight_train,
            )
            score = pipe.score(X_test, y_test, sample_weight=weight_test)

            # Stop if minimum score not set or was reached
            if self.min_score is None or score >= self.min_score:
                break

            # Widen the the window
            new_window = self._get_window(
                window.start - TIME_WINDOW_INCREASE,
                window.stop + TIME_WINDOW_INCREASE,
                len(df) + SAMPLE_INCREASE,
            )
            # Stop if maximum window was reached
            if not self._window_width_ok(new_window):
                break
            else:
                window = new_window

        return _LocalModel(
            pipe,
            score,
            window,
            pd.concat([X_train, y_train, weight_train], axis=1),
            pd.concat([X_test, y_test, weight_test], axis=1),
        )

    def get_models(self, condition=None):
        """Get models with an optional condition"""
        if condition is None or self._models is None:
            return pd.Series(self._models)
        else:
            return pd.Series(self._models).loc[condition]

    def evaluate(self):
        if self._models is None:
            return None

        return pd.concat(
            [
                pd.DataFrame(
                    {
                        "observed": df[self.endog.name],
                        "predicted": self._models[d].predict(df[self.exog.columns]),
                    }
                ).join(df[self.exog.columns])
                for d, df in self._testdata.groupby(level=self._dates.name)
            ],
            axis=0,
        )

    def get_traindata(self):
        # TODO
        ...

    def predict(self, exog: pd.DataFrame, model_condition=None):
        if self._models is None:
            return None

        models = self.get_models(model_condition)

        return (
            pd.concat(
                [
                    pd.Series(models[pd.to_datetime(d)].predict(df), index=df.index)
                    for d, df in exog.dropna().groupby(lambda idx: idx.date)
                    if pd.to_datetime(d) in models.keys()
                ]
            )
            .sort_index()
            .reindex_like(exog)
        )

    def apply_metric(self, metric: callable, model_condition=None) -> float:
        """Apply a metric to this LocalRegression

        Args:
            metric: Callable with signature (y_true, y_pred)

        Returns:

        """

        models = self.get_models(model_condition)

        return models.apply(
            lambda mdl: len(mdl.testdata)
            * metric(
                mdl.testdata[self.endog.name],
                mdl.predict(mdl.testdata[self.exog.columns]),
            )
            / len(self._testdata)
        ).sum()
