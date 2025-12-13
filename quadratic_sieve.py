import math

def gcd(a, b):
    # euclidean algorithm
    if a == 0:
        return b
    return gcd(b % a, a)

def is_prime(n):
    if n <= 1:
        return False

    sqrt_n = n ** (1/2)
    for i in range(2, int(sqrt_n) + 1): # Check divisibility from 2 to √n
        if n % i == 0:
            return False
        
    return True

def check_euler_criterion(n, p):
    # Euler's Criterion: Calculate N^((p-1)/2) mod p
    # If result is 1, N is a quadratic residue.
    euler_val = pow(n, (p - 1) // 2, p)
    if euler_val == 1:
        return True
    return False

def is_smooth(Q, factor_base):
    Q = abs(Q)
    for p in factor_base:
        while Q % p == 0:
            Q = Q // p
    if Q == 1:
        return True
    return False

def compute_bound(N):
    # B = exp((1/2 + o(1)) * sqrt(log(N) * log(log(N)))) where o(1) is assumed 0

    ln_n = math.log(N)       
    ln_ln_n = math.log(ln_n)

    root_term = math.sqrt(ln_n * ln_ln_n)
    coefficient = 0.5
    B = math.exp(coefficient * root_term)

    return int(B)

def compute_Q_function(sqrt_n, step_size, N):
    return ((sqrt_n + step_size)**2) - N

def get_factor_base(beta, N):
    if beta < 2:
        print("B must be at least 2.")
        exit(-1)
    # get primes smaller than the bound B and refine them such that euler criterion is satisfied.
    # euler criterion means that the number can be produced by squaring (quadratic) operation in the mod world
    primes = []
    for num in range(2, beta + 1):
        if is_prime(num) and check_euler_criterion(N, num):
            primes.append(num)
    return primes

def tonelli_shanks(N, p):
    if p == 2:
        root = N % 2
        return (root, root)
    
    Q = p - 1
    S = 0
    while Q % 2 == 0:                       # p - 1 = Q * 2^S
        Q //= 2
        S += 1
    
    z = 2
    while pow(z, (p - 1) // 2, p) != (p - 1):    # quadratic_non_residue 
        z += 1
    
    M = S                                   # exponent depth
    c = pow(z, Q, p)                        # correction seed
    t = pow(N, Q, p)                        # error term (t = 1 means the function is done)
    R = pow(N, (Q + 1) // 2, p)             # best guess for the square root

    while True:
        if t == 1:
            return (R, p - R)
        
        # Find the lowest i (0 < i < M) such that t^(2^i) = 1
        i = 0
        temp_t = t  
        for k in range(1, M):
            temp_t = pow(temp_t, 2, p)
            if temp_t == 1:
                i = k
                break

        b = pow(c, (2**(M - i - 1)), p)
        
        M = i                               # update the variables
        c = pow(b, 2, p)
        t = (t * c) % p
        R = (R * b) % p

def find_roots(N, p, sqrt_n):
    r1, r2 = tonelli_shanks(N % p, p)
    return [
        (r1 - sqrt_n) % p,
        (r2 - sqrt_n) % p
    ]
    
def sieve(sieving_interval_max, N, factor_base):
    sqrt_n = math.ceil(N**(1/2))
    sieving_interval = range(0, sieving_interval_max + 1)
    sieving_table = {}
    for step_size in sieving_interval:
        Q_val = compute_Q_function(sqrt_n, step_size, N)
        sieving_table[step_size] = {"q": Q_val, "log_q": math.log(abs(Q_val))}
    sieving_table_keys = list(sieving_table.keys())
    for p in factor_base:
        roots = set(find_roots(N, p, sqrt_n))
        if len(roots) == 0:
            print(f"Sieving interval is too small for the prime {p} to find any root indexes.")
        update_val = math.log(p)
        for root in roots:
            start = root % p
            for i in range(start, len(sieving_table_keys), p):
                sieving_table[sieving_table_keys[i]]["log_q"] -= update_val
    return sieving_table

def pick_keys_by_Q(sieving_table, factor_base, threshold=10):
    keys = list(sieving_table.keys())
    keys = [key for key in keys if sieving_table[key]["log_q"] < threshold]
    smooth_q_keys = []
    for key in keys:
        q_val = sieving_table[key]["q"]
        if is_smooth(q_val, factor_base):
            smooth_q_keys.append(key)
    return smooth_q_keys
                

if __name__ == '__main__':
    N = 68720000989                             # n to be factored
    beta = compute_bound(N)                     # smoothness factor
    factor_base = get_factor_base(beta, N)
    sieving_table = sieve(1000000, N, factor_base)
    relation_keys = pick_keys_by_Q(sieving_table, factor_base, math.log(beta))
    for key in relation_keys:
        print(f"{sieving_table[key]['log_q']:.2f}", end=' ')
    print()
    print(len(relation_keys))
    print(len(factor_base))    

    