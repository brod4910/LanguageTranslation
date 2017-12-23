import numpy as np
import ActivationFns as af

# sanity check for activation functions

Weight_input_forget = np.random.random((10, 5))


forget_gate = np.zeros((10, 5))

print(Weight_input_forget)

forget_gate = af.sigmoid(Weight_input_forget)

print(forget_gate)