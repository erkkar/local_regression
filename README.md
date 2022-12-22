Local Regression
================


Installation
------------

```
pip install git+https://github.com/erkkar/local_regression
```

Alternatively, download the repository as ZIP archive and run

``` 
pip install <ZIPFILE>
``` 

Usage
-----

Gather model data in a DataFrame

| TIMESTAMP_END       |     H |   NetRad |   SW_IN |
|:--------------------|------:|---------:|--------:|
| 2009-01-01 00:00:00 | nan   |   -56.81 |       0 |
| 2009-01-01 00:30:00 | -40.4 |   -57.35 |       0 |
| 2009-01-01 01:00:00 | -36.3 |   -54.52 |       0 |
| 2009-01-01 01:30:00 | -19.3 |   -49.78 |       0 |
| 2009-01-01 02:00:00 |  61.7 |   -38.16 |       0 |

Fit the models

```python
models = LocalRegression(
    data=data,
    exog=['NetRad'],
    endog='H',
    stratifier='SW_IN',
    min_score=0.8,
    max_window_width="30d",
).fit()
```

Get mean of $R^2$ scores

```python
>>> models.score.mean()
```

Calculate predictions with a condition on score

```python
>>> predicted = models.predict(
    data[["NetRad"]], 
    model_condition=models.score >= 0.5
)
```