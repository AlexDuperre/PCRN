import itertools
import numpy as np
import torch
from functools import reduce


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



    return torch.cat(output, dim=1)


# example

terms = get_polynomial([torch.ones(1, 1, 2, 2) * 1.5, torch.ones(1, 1, 2, 2) * 2, torch.ones(1, 1, 2, 2) * 3], 3)
print(terms)