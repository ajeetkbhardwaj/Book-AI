import numpy as np

# ------------------------
# Utilities
# ------------------------
def tanh(x):
    return np.tanh(x)

def tanh_deriv_from_tanh(t):
    # input: t = tanh(x)
    return 1.0 - t * t

def softmax_logits_to_probs(logits):
    # logits: (N, T, O) or (N, O)
    # numerically stable softmax across last dim
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)

def cross_entropy_loss_from_logits(logits, targets):
    # logits: (N, T, O), targets: (N, T) integer class indices
    probs = softmax_logits_to_probs(logits)
    N, T, O = logits.shape
    # gather probabilities of target classes
    idx = targets.reshape(-1)
    probs_flat = probs.reshape(-1, O)
    p_t = probs_flat[np.arange(N*T), idx]
    # average negative log-likelihood
    loss = -np.mean(np.log(p_t + 1e-12))
    # gradient wrt logits (average)
    grad = probs_flat.copy()
    grad[np.arange(N*T), idx] -= 1.0
    grad = grad.reshape(N, T, O) / (N * 1.0)  # average over batch (and time included in sum of loss)
    return loss, grad

# ------------------------
# RNNCell: per time-step node (vanilla)
# ------------------------
class RNNCell:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Parameters:
            input_size: F
            hidden_size: H_s
            output_size: O
        Stores params in dict with value and grad entries.
        """
        self.F = input_size
        self.H = hidden_size
        self.O = output_size
        # initialize parameters (use small random)
        scale = 0.1
        self.params = {
            'W_f': {'value': np.random.randn(self.F + self.H, self.H) * scale, 'grad': None},
            'B_f': {'value': np.zeros((1, self.H)), 'grad': None},
            'W_v': {'value': np.random.randn(self.H, self.O) * scale, 'grad': None},
            'B_v': {'value': np.zeros((1, self.O)), 'grad': None}
        }

    def forward(self, x_t, h_prev):
        """
        x_t: (N, F)
        h_prev: (N, H)
        returns: y_t (N, O), h_t (N, H)
        saves intermediates for backward
        """
        self.x_t = x_t
        self.h_prev = h_prev
        # z_t: (N, F+H)
        self.z_t = np.concatenate([x_t, h_prev], axis=1)
        self.a_t = self.z_t.dot(self.params['W_f']['value']) + self.params['B_f']['value']  # (N,H)
        self.h_t = tanh(self.a_t)  # (N,H)
        self.y_t = self.h_t.dot(self.params['W_v']['value']) + self.params['B_v']['value']  # (N,O)
        return self.y_t, self.h_t

    def backward(self, grad_y_t, grad_h_next):
        """
        grad_y_t: (N, O) gradient from loss wrt this time-step output y_t
        grad_h_next: (N, H) gradient from future timestep flowing into this h_t
        Returns:
            grad_x_t: (N, F)
            grad_h_prev: (N, H) gradient to propagate to previous timestep
        Also accumulates parameter gradients in self.params[*]['grad'] (set to zero if None)
        """
        N = grad_y_t.shape[0]
        # Initialize grads storage if None
        for k in self.params:
            if self.params[k]['grad'] is None:
                self.params[k]['grad'] = np.zeros_like(self.params[k]['value'])

        # grad through output linear y_t = h_t W_v + B_v
        # dL/dW_v += h_t^T * grad_y_t summed over batch
        self.params['W_v']['grad'] += self.h_t.T.dot(grad_y_t)  # (H, O)
        self.params['B_v']['grad'] += np.sum(grad_y_t, axis=0, keepdims=True)  # (1, O)

        # gradient to h_t from output
        grad_h_from_y = grad_y_t.dot(self.params['W_v']['value'].T)  # (N,H)

        # total gradient at h_t is grad from output plus gradient from next time-step
        grad_h_total = grad_h_from_y + grad_h_next  # (N,H)

        # backprop through tanh: a_t -> h_t
        grad_a_t = grad_h_total * tanh_deriv_from_tanh(self.h_t)  # (N,H)

        # accumulate param grads for W_f and B_f
        # W_f: (F+H, H) ; grad is z_t^T dot grad_a_t
        self.params['W_f']['grad'] += self.z_t.T.dot(grad_a_t)  # (F+H, H)
        self.params['B_f']['grad'] += np.sum(grad_a_t, axis=0, keepdims=True)  # (1, H)

        # gradient wrt z_t
        grad_z_t = grad_a_t.dot(self.params['W_f']['value'].T)  # (N, F+H)

        # split into grad_x_t and grad_h_prev
        grad_x_t = grad_z_t[:, :self.F]
        grad_h_prev = grad_z_t[:, self.F:]

        return grad_x_t, grad_h_prev

    def zero_grads(self):
        for k in self.params:
            self.params[k]['grad'] = np.zeros_like(self.params[k]['value'])

# ------------------------
# RNNLayer: process sequences (using RNNCell repeated)
# ------------------------
class RNNLayer:
    def __init__(self, input_size, hidden_size, output_size):
        self.cell = RNNCell(input_size, hidden_size, output_size)
        self.hidden_size = hidden_size
        self.first = True
        # start_H: (1, H)
        self.start_H = np.zeros((1, hidden_size))

    def forward(self, x_seq):
        """
        x_seq: (N, T, F)
        returns y_seq: (N, T, O)
        saves intermediates needed for backward
        """
        N, T, F = x_seq.shape
        self.N, self.T, self.F = N, T, F
        # initialize H_in repeated across batch
        H_in = np.repeat(self.start_H, N, axis=0)  # (N, H)
        self.h_list = []
        self.y_list = []
        self.x_list = []
        # forward through time steps
        for t in range(T):
            x_t = x_seq[:, t, :]
            y_t, H_in = self.cell.forward(x_t, H_in)
            # store
            self.x_list.append(x_t)
            self.y_list.append(y_t)
            self.h_list.append(H_in)
        # stack outputs
        y_seq = np.stack(self.y_list, axis=1)  # (N, T, O)
        # update start_H as mean across batch of final H_in (book: mean)
        self.start_H = H_in.mean(axis=0, keepdims=True)
        return y_seq

    def backward(self, grad_y_seq):
        """
        grad_y_seq: (N, T, O) gradient wrt outputs for each time step
        returns grad_x_seq: (N, T, F) gradient wrt inputs
        accumulates parameter gradients in self.cell.params[*]['grad']
        """
        N, T, O = grad_y_seq.shape
        # prepare grad containers
        grad_x_seq = np.zeros((N, T, self.F))
        # zero parameter grads before accumulation
        self.cell.zero_grads()

        # initial grad_h_next is zero for last timestep (no future)
        grad_h_next = np.zeros((N, self.hidden_size))

        # backward through time
        for t in reversed(range(T)):
            grad_y_t = grad_y_seq[:, t, :]  # (N, O)
            grad_x_t, grad_h_prev = self.cell.backward(grad_y_t, grad_h_next)
            grad_x_seq[:, t, :] = grad_x_t
            grad_h_next = grad_h_prev  # to pass to previous time step

        # parameter grads now in self.cell.params[*]['grad']
        return grad_x_seq

    def get_param_list(self):
        # flatten params into list of (value, grad) pairs for optimizer
        p = self.cell.params
        return [(p['W_f']['value'], p['W_f']['grad']),
                (p['B_f']['value'], p['B_f']['grad']),
                (p['W_v']['value'], p['W_v']['grad']),
                (p['B_v']['value'], p['B_v']['grad'])]

    def zero_state(self):
        self.start_H[:] = 0.0

# ------------------------
# Simple model: RNNLayer -> (optionally more) -> loss
# ------------------------
class SimpleRNNModel:
    def __init__(self, input_size, hidden_size, output_size):
        self.rnn = RNNLayer(input_size, hidden_size, output_size)
        # no extra final dense here since RNN layer already gives outputs of size O

    def forward(self, x_seq):
        # x_seq: (N,T,F)
        return self.rnn.forward(x_seq)  # returns (N,T,O)

    def backward(self, grad_y_seq):
        return self.rnn.backward(grad_y_seq)

    def params_and_grads(self):
        return self.rnn.get_param_list()

# ------------------------
# Optimizer: SGD (optionally momentum)
# ------------------------
class SGD:
    def __init__(self, params_and_grads, lr=1e-2, momentum=0.0):
        # params_and_grads: list of tuples (param_array_ref, grad_array_ref)
        self.lr = lr
        self.momentum = momentum
        # create velocity arrays keyed by param id
        self.vels = {}
        for p, g in params_and_grads:
            self.vels[id(p)] = np.zeros_like(p)

    def step(self, params_and_grads):
        for p, g in params_and_grads:
            v = self.vels[id(p)]
            # gradient may be None if not set
            if g is None:
                continue
            v[...] = self.momentum * v - self.lr * g
            p[...] += v

# ------------------------
# Numerical gradient check helper (finite difference)
# ------------------------
def numerical_grad_check(model, x_seq, targets, param_ref, param_name, epsilon=1e-5):
    """
    param_ref: tuple (param_array, grad_array) from model.params_and_grads()
    param_name only for printing
    returns relative error between analytic grad and numerical grad
    """
    p, g_analytic = param_ref
    it = np.nditer(p, flags=['multi_index'], op_flags=['readwrite'])
    numerical_grad = np.zeros_like(p)
    orig = p.copy()
    # iterate few elements (full check is slow); here check random 10 elements
    rng = np.random.default_rng(0)
    indices = [tuple((rng.integers(0, s) for s in p.shape)) for _ in range(10)]
    for idx in indices:
        old = p[idx].item()
        p[idx] = old + epsilon
        y_pos = model.forward(x_seq)
        loss_pos, grad_logits_pos = cross_entropy_loss_from_logits(y_pos, targets)
        p[idx] = old - epsilon
        y_neg = model.forward(x_seq)
        loss_neg, grad_logits_neg = cross_entropy_loss_from_logits(y_neg, targets)
        # numeric grad approx for that element (loss scalar)
        numgrad = (loss_pos - loss_neg) / (2 * epsilon)
        numerical_grad[idx] = numgrad
        # restore
        p[idx] = old
    # compare to analytic grad (which must be computed before calling)
    an = g_analytic
    # compute relative error for checked indices
    diffs = []
    for idx in indices:
        a = an[idx]
        n = numerical_grad[idx]
        rel = abs(a - n) / (max(1e-8, abs(a) + abs(n)))
        diffs.append(rel)
    return np.mean(diffs)

# ------------------------
# Quick runnable example (tiny synthetic classification task)
# ------------------------
if __name__ == "__main__":
    # small synthetic problem: N=4, T=5, F=3, O=2
    N, T, F, Hs, O = 4, 5, 3, 6, 2
    np.random.seed(1)

    model = SimpleRNNModel(input_size=F, hidden_size=Hs, output_size=O)

    # generate random input and integer targets (class 0..O-1) per time step
    X = np.random.randn(N, T, F)
    targets = np.random.randint(0, O, size=(N, T))

    # forward
    logits = model.forward(X)  # (N, T, O)
    loss, grad_logits = cross_entropy_loss_from_logits(logits, targets)
    print("Initial loss:", loss)

    # backward
    model.backward(grad_logits)

    # get params and grads
    params_and_grads = model.params_and_grads()

    # numerical gradient check on first param
    first_param = params_and_grads[0]
    rel_err = numerical_grad_check(model, X, targets, first_param, "W_f", epsilon=1e-5)
    print("Numerical grad check (mean relative error over sampled elements):", rel_err)

    # optimizer step example
    optimizer = SGD(params_and_grads, lr=1e-2, momentum=0.9)
    optimizer.step(params_and_grads)
    print("Optimizer step done.")
