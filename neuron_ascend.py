import torch

# ============================================================
# 1. Select device: Ascend NPU if available, otherwise CPU
# ============================================================

try:
    import torch_npu
    if torch_npu.npu.is_available():
        torch_npu.npu.set_device("npu:0")
        device = torch.device("npu:0")
        print("Using Ascend NPU:", torch_npu.npu.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("NPU not available. Using CPU.")
except Exception:
    device = torch.device("cpu")
    print("torch_npu not found. Using CPU.")


# ============================================================
# 2. Training data
# ============================================================
# Examples:
# 4 ⊙ 6 ⊙ 8 = 100
# 5 ⊙ 7 ⊙ 9 = 110
# 6 ⊙ 8 ⊙ 10 = 120
# 8 ⊙ 10 ⊙ 12 = 140

X_raw = torch.tensor([
    [4, 6, 8],
    [5, 7, 9],
    [6, 8, 10],
    [8, 10, 12]
], dtype=torch.float32, device=device)

y_raw = torch.tensor([
    [100],
    [110],
    [120],
    [140]
], dtype=torch.float32, device=device)


# ============================================================
# 3. Scale the data
# ============================================================
# Scaling makes training stable.
# Inputs become smaller:
# 4, 6, 8  ->  0.4, 0.6, 0.8
# Output 100 -> 1.0

X = X_raw / 10.0
y = y_raw / 100.0


# ============================================================
# 4. Initialize one neuron
# ============================================================
# Neuron formula:
# y_pred = x1*w1 + x2*w2 + x3*w3 + b

torch.manual_seed(1)

w = torch.randn((3, 1), dtype=torch.float32, device=device) * 0.01
b = torch.tensor(0.0, dtype=torch.float32, device=device)

print("\nInitial weights:")
print(w.detach().cpu().numpy())

print("Initial bias:")
print(b.item())


# ============================================================
# 5. Training settings
# ============================================================

learning_rate = 0.5
epochs = 1500
n = X.shape[0]


# ============================================================
# 6. Train the neuron manually
# ============================================================
# The update formula is:
#
# w_new = w_old - learning_rate * dw
# b_new = b_old - learning_rate * db
#
# where:
#
# dw = average(error * input)
# db = average(error)

for epoch in range(1, epochs + 1):

    # ----------------------------
    # Forward pass
    # ----------------------------
    # y_pred = Xw + b
    y_pred = X @ w + b

    # ----------------------------
    # Calculate error
    # ----------------------------
    # error = predicted - correct
    error = y_pred - y

    # ----------------------------
    # Calculate loss
    # ----------------------------
    # We use 0.5 so the gradient becomes simple:
    # gradient = error * input
    loss = 0.5 * torch.mean(error ** 2)

    # ----------------------------
    # Calculate gradients
    # ----------------------------
    # dw tells how much each weight caused the error
    dw = (X.T @ error) / n

    # db is the average error
    db = torch.mean(error)

    # ----------------------------
    # Update weights and bias
    # ----------------------------
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # Print progress
    if epoch == 1 or epoch % 300 == 0:
        print(f"Epoch {epoch:4d} | Loss = {loss.item():.10f}")


# ============================================================
# 7. Prediction function
# ============================================================

def predict(a, b_input, c):
    x = torch.tensor([[a, b_input, c]], dtype=torch.float32, device=device)

    # scale input
    x = x / 10.0

    # neuron prediction
    result = x @ w + b

    # return to original scale
    result = result * 100.0

    return result.item()


# ============================================================
# 8. Show final learned weights
# ============================================================

print("\nFinal learned weights:")
print(w.detach().cpu().numpy())

print("Final learned bias:")
print(b.item())


# ============================================================
# 9. Test on training examples
# ============================================================

print("\nTraining examples:")

for i in range(len(X_raw)):
    a = int(X_raw[i][0].item())
    b_input = int(X_raw[i][1].item())
    c = int(X_raw[i][2].item())

    answer = predict(a, b_input, c)

    print(f"{a} ⊙ {b_input} ⊙ {c} = {answer:.0f}")


# ============================================================
# 10. Solve new questions
# ============================================================

print("\nNew questions:")

answer1 = predict(14, 16, 18)
answer2 = predict(22, 24, 26)

print(f"14 ⊙ 16 ⊙ 18 = {answer1:.0f}")
print(f"22 ⊙ 24 ⊙ 26 = {answer2:.0f}")