# Regression derivation notes

Owner: Member 2, Regression engineer, Ehab Fakhralden Mohamed Hamid (25/27950).

These notes match `src/regression.py`. The model estimates `actual_yield_kg` with batch gradient descent in NumPy. It does not call a library regression estimator.

## 1. Question and inputs

The dispatch desk needs an expected harvest weight in kilograms. For each complete row the model uses the six inputs from the data lead:

- plot area (ha)
- rainfall (mm)
- soil pH
- seed (kg)
- distance to the collection point (km)
- planned arrival hour

`record_id` is only a label. `dispatch_attention` is the classification target and is not an input here. The past measured weight, `actual_yield_kg`, is the target `y`. The prediction is a new estimate. It is not a scale reading.

## 2. Split, before any learning

Let there be `n` complete rows. A fixed seed (`random_seed = 42`) shuffles their positions. About 20% become the test rows and the rest stay for training. The counts are whatever the file produces. They are not typed in by hand.

The learning rate and the iteration count are chosen before looking at the test rows. The test rows are used once, at the end, to report MAE, RMSE, and R-squared. They are not used to pick a better rate or a better iteration count.

## 3. Standardize the training features only

Plot area, rainfall, and seed are not on the same scale. Gradient descent behaves more steadily when each input is recentered.

For feature `j`, using only the `m` training rows:

```
mu_j  = (1 / m) * sum of x_ij over training rows
var_j = (1 / m) * sum of (x_ij - mu_j)^2
sigma_j = square root of var_j
```

The division is by `m`, not by `m - 1`. That is population standard deviation (`ddof = 0`), the same convention as a standard scaler. If a training column does not vary, `sigma_j` is replaced by 1 so the later division is safe. That column is then all zeros after centering.

Each feature, including a test row or a later prediction, is transformed with those training values:

```
z_ij = (x_ij - mu_j) / sigma_j
```

The test rows do not change `mu` or `sigma`. The target stays in kilograms, so the prediction stays in kilograms.

## 4. Prediction

A column of ones is placed in front of the scaled features. That column is the intercept and is not standardized. For training row `i`:

```
x_i = [1, z_i1, z_i2, z_i3, z_i4, z_i5, z_i6]
y_hat_i = w0 + w1*z_i1 + w2*z_i2 + w3*z_i3 + w4*z_i4 + w5*z_i5 + w6*z_i6
        = x_i · w
```

`w` has 7 numbers: one intercept and one weight per input. In matrix form, with `X` the `m` by 7 design matrix and `y` the training yields:

```
y_hat = X w
```

## 5. Loss

The training loss is half the mean squared error on the training rows:

```
L(w) = (1 / (2m)) * sum over i of (x_i · w - y_i)^2
```

The one-half does not move the best `w`. It cancels the 2 that appears when the square is differentiated, so the gradient below has a simple `1/m`. Minimizing `L` is the same task as minimizing the mean squared error.

At the start, `w` is the zero vector. The first point on the loss curve is `L` at that start. Each later point is `L` after one update. The history therefore has `n_iterations + 1` values. The run uses `learning_rate = 0.05` and `n_iterations = 800`.

## 6. Gradient of one weight

Write the residual of row `i` as:

```
r_i = x_i · w - y_i
```

Then:

```
L(w) = (1 / (2m)) * sum over i of r_i^2
```

Differentiate with respect to weight `w_j`. The chain rule on `r_i^2` brings down `2 r_i`, and `r_i` changes with `w_j` at the rate `x_ij`:

```
dL / dw_j = (1 / (2m)) * sum over i of 2 * r_i * x_ij
          = (1 / m) * sum over i of r_i * x_ij
```

The 2 and the 1/2 cancel. In words: the slope for weight `j` is the average, over the training rows, of (residual times the `j`-th column of `X`).

Stacking every weight gives the batch gradient. `X^T (Xw - y)` multiplies every residual by every column and sums down the rows. Dividing by `m` matches the line above:

```
gradient = (1 / m) * X^T (X w - y)
```

That is the return value of `batch_gradient` in `src/regression.py`.

## 7. Update

One batch step moves every weight against its slope:

```
w := w - learning_rate * gradient
```

The same `m` training rows are used on every step. That is why it is batch gradient descent rather than a single-row update. After 800 steps the weights, the training mean, the training standard deviation, and the seed are saved. A new row is scaled with that same mean and standard deviation, then multiplied by `w`.

## 8. One step with small numbers

One training row, already scaled, with one input. Then `m = 1`, `x = [1, 2]`, `y = 5`, and `w = [0, 0]`.

```
prediction = 0
residual   = 0 - 5 = -5
gradient   = [1, 2] * -5 / 1 = [-5, -10]
```

With learning rate 0.05:

```
w_new = [0, 0] - 0.05 * [-5, -10] = [0.25, 0.5]
prediction_new = 0.25 + 0.5 * 2 = 1.25
```

Loss before the step: `(1/2) * (0 - 5)^2 = 12.5`.
Loss after the step: `(1/2) * (1.25 - 5)^2 = 7.03125`.

The loss fell, and the prediction moved toward 5. The unit test `test_analytic_gradient_matches_a_numerical_check` compares this same gradient with a small finite difference, so the algebra and the code stay tied together.

## 9. Scores reported after training

These scores are not the training loss. They are computed for the training rows and, separately, for the test rows.

```
MAE  = average of |prediction - actual|
RMSE = square root of the average of (prediction - actual)^2
R^2  = 1 - (sum of squared residuals) / (sum of (actual - mean actual)^2)
```

The mean in `R^2` is the mean of the actual values in the set being scored. If that set has no variation, `R^2` is left empty rather than divided by zero. MAE and RMSE stay in kilograms. `R^2` has no unit: 1 would mean the predictions match the actual yields exactly, and 0 would mean they do no better than predicting the average yield.

## 10. What a live change should do

If the learning rate is raised and the run is repeated, each step is larger. The loss curve can fall faster, or it can rise and the weights can become non-finite. The code stops if the weights or the test predictions are not finite.

If the learning rate is lowered, the same 800 steps move less far. The final training loss is usually higher than with 0.05 unless the iteration count is also increased.

Changing the seed changes which rows are held out. The test MAE, RMSE, and `R^2` can change. The formula does not.

The test set is still not a knob. A rate is not kept because it made the test score look better.
