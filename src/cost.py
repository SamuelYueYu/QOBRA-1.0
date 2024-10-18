import time, torch
import warnings

from count import *
from qiskit.quantum_info import Statevector, state_fidelity

def compute_kernel(X, Y, sigma):
    x,y = torch.tensor(X),torch.tensor(Y)
    x_size,y_size = x.size(0),y.size(0)
    d = x.size(1)
    
    x = x.unsqueeze(1) # (x_size, 1, d)
    y = y.unsqueeze(0) # (1, y_size, d)
    tiled_x = x.expand(x_size, y_size, d)
    tiled_y = y.expand(x_size, y_size, d)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(d)
    return torch.exp(-kernel_input/sigma) # (x_size, y_size)

def compute_mmd(x, y, Sigma):
    xy_kernel = compute_kernel(x, y, Sigma)
    mmd = 1 - xy_kernel.mean()
    return mmd.item()

def latent_rep(x, p):
    # Latent space representation
    param_dict = {i_params[j]: x[j] for j in range(num_feature)}
    param_dict.update({e_params[j]: p[j] for j in range(num_encode)})
    q = qc_v.assign_parameters(param_dict)
    psi = Statevector.from_instruction(q)
    return np.real(psi.data)
    
trains_k,tests_k = [],[]
def k_loss(p, train_input, test_input, start):
    n = train_input.shape[0]
    f = partial(latent_rep, p=p)
    s = time.time()
    with Pool(processes = processes) as pool:
        train_encode = np.array(pool.map(f, train_input))
        test_encode = np.array(pool.map(f, test_input))
    
    target = make_target(n)
    k_train = compute_mmd(train_encode, target, Sigma)
    k_test = compute_mmd(test_encode, target, Sigma)
    
    trains_k.append(k_train)
    tests_k.append(k_test)
    if (len(trains_k)-1)%10 == 0:
        print(len(trains_k)-1)
        print(f"mMMD loss: {k_train:.4f}, {k_test:.4f}")
        interval = time.time()-s
        print(f"Interval: {interval:0.2f} s")
        
    plt.figure(figsize=(5,5))
    clear_output(wait=True)
    # Plot the histogram of the data hist
    idx = np.random.randint(n)
    plt.hist(train_encode[idx], color='r', density=True, bins=dim, alpha=1, label='Train')
    idx = np.random.randint(test_input.shape[0])
    plt.hist(test_encode[idx], color='g', density=True, bins=dim, alpha=.8, label='Test')
    idx = np.random.randint(target.shape[0])
    plt.hist(target[idx], color='b', density=True, bins=dim, alpha=.6, label='Reference')
    
    plt.title("Frequency of state amplitudes")
    plt.xlabel("State amplitudes")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"{S}-hist.png", dpi = 300, bbox_inches='tight')
    
    # plotting part
    plt.figure(figsize=(5,5))
    clear_output(wait=True)
    plt.plot(range(len(trains_k)), np.array(trains_k), label='Training')
    plt.plot(range(len(tests_k)), np.array(tests_k), label='Validation')
    
    plt.title("mMMD / iteration")
    plt.xlabel("Iteration")
    plt.ylabel("mMMD")
    plt.legend()
    plt.savefig(f"{S}-mMMD.png",dpi = 300, bbox_inches='tight')
    return k_train
    
def F(x, pd, pe):
    noise = np.random.normal(0, nstd, num_latent)
    
    param_dict = {i_params[j]: x[j] for j in range(num_feature)}
    param_dict.update({e_params[j]: pe[j] for j in range(num_encode)})
    param_dict.update({n_params[j]: noise[j] for j in range(num_latent)})
    param_dict.update({d_params[j]: pd[j] for j in range(num_decode)})
    
    output_qc = train_qc.assign_parameters(param_dict)
    return state_fidelity(Statevector(x), Statevector(output_qc))

trains_f,tests_f = [],[]
trains_g,tests_g = [],[]
def f_loss(pd, pe, train_input, test_input, start):
    f = partial(F, pd=pd, pe=pe)
    s = time.time()
    with Pool(processes = processes) as pool:
        fs_train = pool.map(f, train_input)
        fs_test = pool.map(f, test_input)
        
    f_train = 1 - np.mean(fs_train)
    f_test = 1 - np.mean(fs_test)
    trains_f.append(f_train)
    tests_f.append(f_test)
    
    if (len(trains_f)-1)%10 == 0:
        print(len(trains_f)-1)
        print(f"Infidelity loss: {f_train:.8f}, {f_test:.8f}")
        interval = time.time()-s
        print(f"Interval: {interval:0.2f} s")
    
    # plotting part
    plt.figure(figsize=(5,5))
    clear_output(wait=True)
    plt.plot(range(len(trains_f)), np.array(trains_f), label='Training infidelity')
    plt.plot(range(len(tests_f)), np.array(tests_f), label='Validation infidelity')
    
    plt.title("Loss / iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{S}-infi.png",dpi = 300, bbox_inches='tight')
    return f_train
