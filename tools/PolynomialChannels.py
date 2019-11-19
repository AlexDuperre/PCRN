import itertools
import numpy as np
import torch
import torch.nn as nn
from functools import reduce
import pretrainedmodels

def get_polynomial(vars, power):
    #   if "c" in vars:
    #     raise Exception("\"c\" cannot be a variable")
    dummy = torch.tensor([123.])
    vars.append(dummy)  # add dummy variable

    # compute all combinations of variables
    terms = []
    output = []
    for x in itertools.combinations_with_replacement(vars, power):
        terms.append(x)
        # print(x)
        List = list(x)
        while any( (dummy == x_).all() for x_ in List):
            List.pop(int(np.nonzero([(dummy == x_).all() for x_ in List])[0][0]))
        if len(List) != 0:
            output.append(reduce(lambda x, y: x*y, List))
    print("len",len(terms))
    return torch.cat(output, dim=1)

from itertools import chain, combinations

# def powerset(iterable):
#     "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
#     s = list(iterable)  # allows duplicate elements
#     return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
#
# stuff = [torch.ones(5, 1, 600, 600) * 1.5, torch.ones(5, 1, 600, 600)  * 2, torch.ones(5, 1, 600, 600)  * 3]#[1, 2, 3]
# List = []
# for i, combo in enumerate(powerset(stuff), 1):
#     print('combo #{}: {}'.format(i, combo))
#     if len(list(combo)) != 0:
#         List.append(reduce(lambda x, y: x * y, list(combo)))

stuff = [torch.ones(1, 1, 2, 2) * 1.5, torch.ones(1, 1, 2, 2) * 2, torch.ones(1, 1, 2, 2) * 3]#[1, 2, 3]
List = []
for L in range(1, 4):
    for subset in itertools.combinations_with_replacement(stuff, L):
        print(subset)
        if len(list(subset)) != 0:
            List.append(reduce(lambda x, y: x * y, list(subset)))
# example
print(len(List))
# terms = get_polynomial([torch.ones(1, 1, 2, 2) * 1.5, torch.ones(1, 1, 2, 2) * 2, torch.ones(1, 1, 2, 2) * 3], 3)
# print(terms)
# for i in range(1,3+1):
#     print("degree", i)
#     terms = get_polynomial([torch.ones(5, 1, 600, 600) * 1.5, torch.ones(5, 1, 600, 600)  * 2, torch.ones(5, 1, 600, 600)  * 3], i)
#     print(terms.shape[1])

