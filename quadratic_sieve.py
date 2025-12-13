import math
from sympy import Matrix, GF

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

def factor_and_check_smoothness(Q, factor_base):
    Q = abs(Q)
    factor_dict = {p: 0 for p in factor_base}
    for p in factor_base:
        while Q % p == 0:
            Q = Q // p
            factor_dict[p] = factor_dict[p] + 1
    return factor_dict, Q == 1

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

def precompute_roots(N, factor_base):
    sqrt_n = math.ceil(N**(1/2))

    root_table = {}

    for p in factor_base:
        r1, r2 = tonelli_shanks(N % p, p)
        roots = {
            (r1 - sqrt_n) % p,
            (r2 - sqrt_n) % p
        }
        root_table[p] = roots

    return root_table

def find_roots(N, p, sqrt_n):
    r1, r2 = tonelli_shanks(N % p, p)
    return [
        (r1 - sqrt_n) % p,
        (r2 - sqrt_n) % p
    ]
    
def sieve_range(x_start, x_end, N, factor_base, root_table):
    sqrt_n = math.ceil(N**0.5)
    sieving_table = {
        x: {
            "q": compute_Q_function(sqrt_n, x, N),
            "log_q": math.log(abs(compute_Q_function(sqrt_n, x, N)))
        }
        for x in range(x_start, x_end)
    }

    for p in factor_base:
        update_val = math.log(p)
        roots = set(root_table[p])

        for root in roots:
            x0 = x_start + ((root - x_start) % p)
            for x in range(x0, x_end, p):
                sieving_table[x]["log_q"] -= update_val

    return sieving_table

def find_and_factor_smooth_numbers(sieving_table, factor_base, threshold=10):
    keys = list(sieving_table.keys())
    keys = [key for key in keys if sieving_table[key]["log_q"] < threshold]
    factored_smooth_numbers = {}
    for key in keys:
        q_val = sieving_table[key]["q"]
        factor_dict, is_smooth = factor_and_check_smoothness(q_val, factor_base)
        if is_smooth:
            factored_smooth_numbers[key] = factor_dict
    return factored_smooth_numbers
                

def try_dependency(dep_vec, factored_smooth_nums, factor_base, N):
    sqrt_n = math.ceil(N ** (1/2))

    X = 1
    prime_exp_sum = {p: 0 for p in factor_base}

    for i, bit in enumerate(dep_vec):
        if bit == 1:
            x_i, factor_dict = factored_smooth_nums[i]

            # Build X
            X = (X * (sqrt_n + x_i)) % N

            # Accumulate prime exponents
            for p in factor_base:
                prime_exp_sum[p] += factor_dict[p]

    # Build Y
    Y = 1
    for p, exp in prime_exp_sum.items():
        Y *= p ** (exp // 2)

    g = gcd(abs(X - Y) % N, N)
    return g


if __name__ == '__main__':
    N = 68720000989                             # n to be factored
    beta = compute_bound(N)                     # smoothness factor
    block_size = 100000                         # sieving block size
    x_start = 0

    factor_base = get_factor_base(beta, N)                                          # factor base
    root_table = precompute_roots(N, factor_base)                                   # sieving roots for each prime
    number_of_needed_relations = len(factor_base) + 15                              # heuristic for the number of needed smooth nums
    threshold = math.log(beta)

    factored_smooth_nums = []
    while len(factored_smooth_nums) < number_of_needed_relations:
        sieving_table = sieve_range(x_start, x_start + block_size, N, factor_base, root_table)
        smooth_number_dict = find_and_factor_smooth_numbers(sieving_table, factor_base, threshold)
        factored_smooth_nums = factored_smooth_nums + [(off_set, factored_dict) for off_set, factored_dict in smooth_number_dict.items()]
        x_start = x_start + block_size

    parity_matrix = []
    for _, factor_dict in factored_smooth_nums:
        row = [factor_dict[p] % 2 for p in factor_base]
        parity_matrix.append(row)    
    
    M = Matrix(parity_matrix)
    print("Solving for Linear Dependencies (Nullspace over GF(2))...")
    # We need the Nullspace of the TRANSPOSE.
    # We want to find a combination of ROWS that sums to zero. 
    # M.T * v = 0  <==>  v * M = 0
    null_basis = M.T.nullspace()
    print(f"Found {len(null_basis)} linear dependencies.")

    for i, basis_vec in enumerate(null_basis):
        # SymPy returns vectors with GF(2) elements, convert to standard int list
        dep_vec = [int(x) for x in basis_vec]
        
        # Check if this dependency yields a factor
        factor = try_dependency(dep_vec, factored_smooth_nums, factor_base, N)
        
        if factor > 1 and factor < N:
            print("---------------------------------------")
            print(f"SUCCESS! N = {N}")
            print(f"Factors found: {factor} * {N // factor}")
            print("---------------------------------------")
            break
    else:
        print("Dependencies found, but all were trivial (X = +/- Y).")
        print("Try increasing the 'number_of_needed_relations' or 'block_size'.")
        
