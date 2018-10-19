# neldermead

Nelder-Mead implementation

## Getting Started


### Prerequisites

You need only [NumPy](http://www.numpy.org/) that is the package for scientific computing.

### Installing

Please run the following command.

```bash
$ pip install neldermead
```

## Example

This is a simple example that objective function is sphere function.

```python
import numpy as np
from neldermead import NelderMead

dim = 3
f = lambda x: np.sum(x**2)
simplex = np.zeros([dim, dim + 1])
for i in range(dim + 1):
    simplex[:, i] = np.array([np.random.rand() for _ in range(dim)])
nm = NelderMead(dim, f, simplex)

x_best, f_best = nm.optimize(100)
print("x_best:{}, f_best:{}".format(x_best, f_best))
# x_best:[[0.60389863]
# [0.26992676]
# [0.13184699]], f_best:0.45493763222718836
```


## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/nmasahiro/neldermead/tags). 


## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/nmasahiro/neldermead/blob/master/LISENCE) file for details
