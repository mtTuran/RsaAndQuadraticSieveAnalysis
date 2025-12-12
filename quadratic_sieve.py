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

def get_factor_base(beta, n):
    if beta < 2:
        print("B must be at least 2.")
        exit(-1)
    primes = [num for num in range(2, beta + 1) if is_prime(num)]       # get primes smaller than B
    # refine primes p such that n has a square root mod p
    refined_primes = []
    for p in primes:
        quadratic_residue = (n % p)**(1/2)
        if quadratic_residue == int(quadratic_residue):
            refined_primes.append(p)
    return refined_primes

if __name__ == '__main__':
    beta = 100              # smoothness factor
    N = 221                 # n to be factored
    factor_base = get_factor_base(beta, N)
    print(factor_base)

    