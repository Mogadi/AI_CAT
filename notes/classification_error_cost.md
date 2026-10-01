# Which classification error costs more

Owner: Member 3, Classification engineer, Mojtaba Abdalitieef Ahmed (25/27660).

These notes match `src/classification.py`. The classifier estimates `dispatch_attention`. Label 1 means the consignment needs a look before it leaves. Label 0 means it can be treated as ready. The positive class in the scores is 1.

## 1. The four cells

`confusion_matrix.png` and `classification_metrics.json` use this order. Rows are the actual label. Columns are the predicted label. Both axes are `[0, 1]`.

```
                 predicted 0          predicted 1
actual 0         true ready           false Hold
actual 1         missed attention     true Hold
```

- True ready: the load did not need attention, and the model said 0.
- False Hold: the load did not need attention, and the model said 1. The desk keeps a ready load.
- Missed attention: the load needed a look, and the model said 0. The desk can send it out.
- True Hold: the load needed a look, and the model said 1.

## 2. The costly mistake

The costly mistake is the missed attention: actual 1 predicted as 0.

A load that leaves and later proves it needed a check is already off the desk. Calling it back is harder than holding a ready load for a few more minutes. A false Hold is a delay. The officer can still choose Dispatch anyway, and the sacks are still there. A missed attention can leave with the truck.

So a false Hold is the cheaper error. A missed attention is the expensive one. The model does not send the truck. The officer does. The score exists so that expensive miss is visible before the decision.

## 3. How the four scores see that mistake

The test metrics use label 1 as the positive class.

```
precision = true Holds / (true Holds + false Holds)
recall    = true Holds / (true Holds + missed attention)
f1        = the balance of precision and recall
accuracy  = share of test rows whose label matches, either class
```

Recall falls when missed-attention cells grow. That is the score tied to the costly error. Precision falls when false Holds grow. That is the cheaper delay. Accuracy can stay high when label 1 is uncommon, because many easy ready rows hide a few missed loads. Accuracy alone is not enough for this desk. Recall of class 1 has to be read next to it.

## 4. Where the label comes from

The reported label is the logistic regression decision at a probability of 0.5 for class 1. The probability of attention is stored on each test row as `probability_attention`. The penalty `C = 1.0` is fixed before the test rows are scored. It is not chosen because it made the test F1 look better.

Lowering the cut, so that a probability under 0.5 can still be called Hold, would catch more true Holds and would also hold more ready loads. Recall of class 1 would tend to rise, and precision would tend to fall. That trade accepts more of the cheap error to avoid the costly one. The saved metrics use the 0.5 cut. The probability is kept so a person can still Hold when the two outcomes are close.

## 5. What a live change should show

If the seed changes, the held-out rows change, and the four counts can change. The meaning of the cells does not.

If a test row that was actual 1 is predicted 0, it belongs in the bottom-left cell. That cell is the one to point at when explaining the cost. The top-right cell is a delay, not a lost load.
