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

```python
models = LocalRegression(
    exog=data[['NetRad']],
    endog=data['H'],
    stratifier=data['SW_IN'],
    min_score=0.8,
    max_window_width="30d",
).fit()
```

Get mean of R2 scores

```python
models.score.mean()
```

Calculate predictions with a condition on score

```python
predicted = models.predict(
    data[["NetRad"]], 
    model_condition=models.score >= 0.5
)
```