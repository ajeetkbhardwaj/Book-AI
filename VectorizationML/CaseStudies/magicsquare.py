# import all magic square functions

import numpy as np

def magic_square(n):
    """
    
    """
    n = int(n)
    if n < 3 : 
        raise (ValueError("n must be > 2"))
    if n % 2 ==  1:
        return odd_order_magicsquare(n)
    elif n % 4 == 0: 
        return even_order_magicsquare(n)
    else:
        return single_even_order_magicsquare(n)

        