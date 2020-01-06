


Import the math package to compute the objective Function

```python
import math
```


Import the superclass (also called base class), which is an abstract class, to implement the subclass AckleyFunction

```python
from ObjectiveFunction import *
```

The subclass that inherits of ObjectiveFunction
```python
class TestProblem(ObjectiveFunction):
```

Create a constructor
```python
    # Constructor
    def __init__(self):

        number_of_dimensions = 2;

        # Store the boundaries
        self.boundaries = [];
        for _ in range(number_of_dimensions):
            self.boundaries.append([-32.768, 32.768]);

        # Call the constructor of the superclass
        super().__init__(number_of_dimensions,
                         self.boundaries,
                         self.objectiveFunction,
                         ObjectiveFunction.MINIMISATION);

        # The name of the function
        self.name = "Ackley Function";

        # Store the global optimum if known
        self.global_optimum = [];
        for _ in range(self.number_of_dimensions):
            self.global_optimum.append(0.0);

        # Typical values: a = 20, b = 0.2 and c = 2pi.
        self.a = 20;
        self.b = 0.2;
        self.c = 2 * math.pi;
        self.global_optimum = [0, 0];


    # objectiveFunction implements the Ackley function
    def objectiveFunction(self, aSolution):

        M = 0;
        N = 0;
        O = 1 / self.number_of_dimensions;

        for i in range(self.number_of_dimensions):
            M += math.pow(aSolution[i], 2);
            N += math.cos(self.c * aSolution[i]);

        return -self.a * math.exp(-self.b * math.sqrt(O * M)) - math.exp(O * N) + self.a + math.e;
```



The whole code:
```python
import math

from ObjectiveFunction import *


class TestProblem(ObjectiveFunction):
    def __init__(self):

        number_of_dimensions = 2;

        self.boundaries = [];
        for _ in range(number_of_dimensions):
            self.boundaries.append([-32.768, 32.768]);

        super().__init__(number_of_dimensions,
                         self.boundaries,
                         self.objectiveFunction,
                         ObjectiveFunction.MINIMISATION);

        self.name = "Ackley Function";

        self.a = 20;
        self.b = 0.2;
        self.c = 2 * math.pi;
        self.global_optimum = [0, 0];


    def objectiveFunction(self, aSolution):

        M = 0;
        N = 0;
        O = 1 / self.number_of_dimensions;

        for i in range(self.number_of_dimensions):
            M += math.pow(aSolution[i], 2);
            N += math.cos(self.c * aSolution[i]);

        return -self.a * math.exp(-self.b * math.sqrt(O * M)) - math.exp(O * N) + self.a + math.e;
```
