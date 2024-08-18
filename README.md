# CSR-KB - continuous segmented regression with known breakpoints

This repo implements an optionally weighted CSR-KB model with various methods and optional $L_2$ regularisation. It additionally contains a function to plot the resultant model. 

**Mathematical details of implementation:**

[![Build Latex document](https://github.com/SebFoulger/CSR-KB/actions/workflows/build_latex.yaml/badge.svg)](https://github.com/SebFoulger/CSR-KB/actions/workflows/build_latex.yaml)

[Download latest PDF](https://nightly.link/SebFoulger/CSR-KB/workflows/build_latex.yaml/main/PDF.zip)

## What is the CSR-KB problem?

Say there are $m$ segments s.t. segment $j$ is intended to have linear regression $y_j = X_j \beta_j$ and is of size $s_j$ where

$$y_j = \begin{pmatrix}
y_{j,1} \\
\vdots \\
y_{j,s_j}
\end{pmatrix}, X_j = 
\begin{bmatrix}
-x_{j,1}-  \\
\vdots \\
-x_{j,s_j}-
\end{bmatrix}, \beta_j = 
\begin{pmatrix}
\beta_{j,1}  \\
\vdots \\
\beta_{j,d}
\end{pmatrix}.$$

Then the continuous segmented regression problem with known breakpoints (CSR-KB) is

$$\min_{\beta_1,...,\beta_m} \sum_{j=1}^m  (y_j-X_j\beta_j)^T(y_j-X_j\beta_j)$$

such that

$$x_{j, 1}^T \beta_j = x_{j, 1}^T \beta_{j-1}, \text{ for }j=2, ..., m.$$

This is the standard (segmented) linear regression setup, but with constraints to ensure the outputted piecewise linear model is continuous.

## Usage

The main usage is as follows:
```python
from main import CSR_KB

model = CSR_KB(endog=endog, breakpoints=breakpoints, exog=exog, weights=weights, seg_weights=seg_weights, hasconst=hasconst)

betas = model.fit(method=method)
# OR
betas = model.fit_regularized(mu=mu, method=method)

model.plot(...)
```
The data is whitened upon instantiation of the CSR_KB object, but whitening of any ndarray can also be performed using the model weights as long as the ndarray has the same length as the endogenous (and exogenous) variable:
```python
x_white = model.whiten(x)
```

An example usage can be found in [example.ipynb](example.ipynb).
