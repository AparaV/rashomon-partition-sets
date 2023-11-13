import math
import itertools
import numpy as np

from functools import reduce


def prime_factors(n): 
    
    factors = []
    
    while n % 2 == 0:
        factors.append(2)
        n = n // 2
          
    sqrt_n = int(np.sqrt(n))
    for i in range(3, sqrt_n+1, 2):
        print(i)
        while n % i == 0: 
            factors.append(i)
            n = n // i 
              
    if n > 2: 
        factors.append(n)
    
    return factors


def generate_all_factorizations(factors):

    def prod(x):
        res = 1
        for xi in x:
            res *= xi
        return res

    if len(factors) <= 1:
        yield factors
        return

    for f in range(1, len(factors)+1):
        for which_is in itertools.combinations(range(len(factors)), f):
            this_prod = prod(factors[i] for i in which_is)
            rest = [factors[i] for i in range(len(factors)) if i not in which_is]

            for remaining in generate_all_factorizations(rest):
                yield [this_prod] + remaining


def factorizations(factors):
    seen = set()
    # res = []
    for f in generate_all_factorizations(factors):
        f = tuple(sorted(f))
        if f in seen:
            continue
        seen.add(f)
        yield f

# taken from https://stackoverflow.com/a/6800214
def factors(n):    
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))



def num_admissible_poolings(h, m, R):
    if h == 1 or h == (R-1)**m:
        return 1
    
    N = 0
    primes = prime_factors(h)
    for f in factorizations(primes):
        k = len(f)
        if k > m:
            continue
        N_f = math.comb(m, k)
        for fi in f:
            if fi > R - 2:
                N_f = 0
                break
            N_f = N_f * math.comb(R-2, fi-1)
        N = N + N_f
        
    return N