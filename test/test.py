from micrograd.engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f

print(f"g.data: {g.data}")  # Output of the forward pass
g.backward()  # Computes gradients

# Print gradients
print(f"a.grad: {a.grad}")  # Partial derivative of g w.r.t. a
print(f"b.grad: {b.grad}")  # Partial derivative of g w.r.t. b
