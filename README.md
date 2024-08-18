# CSR-KB

Continuous segmented regression with known breakpoints. 

Say there are $m$ segments s.t. segment $j$ is intended to have linear regression $\by_j = \bX_j \bbeta_j$ and is of size $s_j$ where
$$\by_j = \begin{pmatrix}$$
y_{j,1} \\
\vdots \\
y_{j,s_j}
\end{pmatrix}, \bX_j = 
\begin{bmatrix}
-\bx_{j,1}-  \\
\vdots \\
-\bx_{j,s_j}-
\end{bmatrix}, \boldsymbol{\beta}_j = 
\begin{bmatrix}
\beta_{j,1}  \\
\vdots \\
\beta_{j,d}
\end{bmatrix}.$$
Then the continuous segmented regression problem with known breakpoints (CSR-KB) and weights is
$$\min_{\bbeta_1,...,\bbeta_m} \sum_{j=1}^m w'_j\sum_{i=1}^{s_j} w_{ji}(y_{j, i}-\bx_{j, i}^T\bbeta_j)^2 = \min_{\bbeta_1,...,\bbeta_m} \sum_{j=1}^m w'_j (\by_j-\bX_j\bbeta_j)^T\bW_j(\by_j-\bX_j\bbeta_j)$$
such that
$$\bx_{j, 1}^T \bbeta_j = \bx_{j, 1}^T \bbeta_{j-1}, \text{ for }j=2, ..., m.$$
The regression weights are given by $w_{ji} > 0$ for $j\in \{1, ..., m\}, i \in \{1, ..., s_j\}$, $\bW_j = \text{diag}(w_{j1}, ..., w_{js_j})$. The segment weights are given by $w'_j > 0$ for $j \in \{1, ..., m\}$. Note that the segment weights could've been implemented through the regression weights, but for ease of use, these have been separated. 

[![Build Latex document](https://github.com/SebFoulger/CSR-KB/actions/workflows/build_latex.yaml/badge.svg)](https://github.com/SebFoulger/CSR-KB/actions/workflows/build_latex.yaml)

[Download latest PDF](https://nightly.link/SebFoulger/CSR-KB/workflows/build_latex.yaml/main/PDF.zip)
