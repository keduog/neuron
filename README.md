## How the Neuron Learns: A Simple Explanation

Let us understand the neuron like a student learning from mistakes.

A single artificial neuron uses this formula:

$$
\hat{y} = x_1w_1 + x_2w_2 + x_3w_3 + b
$$

Where:

```text
x1, x2, x3 = input values
w1, w2, w3 = weights
b          = bias
ŷ          = predicted output
```

At the beginning, the neuron does not know the correct weights. Therefore, it starts with small random weights.

For example:

```text
w1 = 0.01
w2 = 0.02
w3 = 0.03
b  = 0
```

Now take one training example:

```text
4 ⊙ 6 ⊙ 8 = 100
```

In the program, the input and output are scaled to make training easier.

So:

```text
x1 = 0.4
x2 = 0.6
x3 = 0.8
correct answer = 1.0
```

---

## Step 1: Forward Pass

The neuron makes a prediction using the current weights.

$$
\hat{y} = 0.4(0.01) + 0.6(0.02) + 0.8(0.03) + 0
$$

$$
\hat{y} = 0.004 + 0.012 + 0.024
$$

$$
\hat{y} = 0.040
$$

So the neuron predicts:

```text
Predicted = 0.040
Correct   = 1.000
```

The neuron is very wrong at the beginning.

---

## Step 2: Calculate Error

Error means the difference between the predicted answer and the correct answer.

$$
error = predicted - correct
$$

So:

$$
error = 0.040 - 1.000
$$

$$
error = -0.960
$$

The error is negative. This means:

```text
The neuron predicted too small.
```

---

## Step 3: Find Which Weight Caused More Error

Look at the input values:

```text
x1 = 0.4
x2 = 0.6
x3 = 0.8
```

The biggest input is:

```text
x3 = 0.8
```

This means `w3` affected the output more than `w1` and `w2`.

So, during learning, `w3` should change more.

---

## Step 4: Calculate Weight Changes

The neuron calculates how much each weight contributed to the error.

$$
dw_1 = error \times x_1
$$

$$
dw_2 = error \times x_2
$$

$$
dw_3 = error \times x_3
$$

Now calculate each one.

### Weight 1

$$
dw_1 = -0.960 \times 0.4
$$

$$
dw_1 = -0.384
$$

### Weight 2

$$
dw_2 = -0.960 \times 0.6
$$

$$
dw_2 = -0.576
$$

### Weight 3

$$
dw_3 = -0.960 \times 0.8
$$

$$
dw_3 = -0.768
$$

Notice that `w3` has the biggest change because `x3` was the biggest input.

```text
Weight 3 changes the most because input 3 was biggest.
```

---

## Step 5: Update the Weights

The weight update formula is:

$$
new\ weight = old\ weight - learning\ rate \times gradient
$$

In simple form:

$$
w = w - learning\ rate \times dw
$$

Assume:

```text
learning rate = 0.5
```

### Update w1

Old value:

```text
w1 = 0.01
```

Update:

$$
w_1 = 0.01 - 0.5(-0.384)
$$

$$
w_1 = 0.01 + 0.192
$$

$$
w_1 = 0.202
$$

### Update w2

$$
w_2 = 0.02 - 0.5(-0.576)
$$

$$
w_2 = 0.308
$$

### Update w3

$$
w_3 = 0.03 - 0.5(-0.768)
$$

$$
w_3 = 0.414
$$

Now the new weights are:

```text
w1 = 0.202
w2 = 0.308
w3 = 0.414
```

---

## Step 6: What Happened?

Initially, the weights were:

```text
w1 = 0.01
w2 = 0.02
w3 = 0.03
```

After one learning step, the weights became:

```text
w1 = 0.202
w2 = 0.308
w3 = 0.414
```

The neuron increased the weights because the prediction was too small.

In simple words, the neuron says:

```text
"I need bigger weights next time."
```

---

## Step 7: Repeat for Many Epochs

The neuron repeats the same process many times:

```text
1. Predict
2. Calculate error
3. Calculate weight changes
4. Update weights
5. Repeat
```

In this project, the neuron is trained for:

```text
1500 epochs
```

After many epochs, the neuron slowly learns better weights and bias.

---

## Real Intuition

Think of the neuron like a student solving math problems.

At first, the student guesses randomly.

The teacher gives the correct answer.

If the student is wrong, the student adjusts their thinking.

After many corrections, the student learns the pattern.

That is how neural network training works.

---

## The Most Important Formula

The heart of neural network learning is:

$$
w = w - learning\ rate \times gradient
$$

Where:

```text
gradient = how much the weight contributed to the error
```

If the prediction is wrong, the neuron changes the weights.

If the prediction is almost correct, the weight changes become very small.

This is the core idea behind training neural networks.

---

## In the Code

The weight update is implemented using:

```python
dw = (X.T @ error) / n
db = torch.mean(error)

w = w - learning_rate * dw
b = b - learning_rate * db
```

Where:

```text
dw = gradient of the weights
db = gradient of the bias
n  = number of training examples
```

This is how the neuron learns from the training examples and improves its predictions over 1500 epochs.
