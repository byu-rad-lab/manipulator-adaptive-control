# Manipulator Adaptive Control
Python package containing adaptive controllers for use on robotic manipulators.

# Controllers

## Radial Basis Function Neural Network (RBFNN) Controller

Given manipulator dynamics of the form:
 $$M(q) \ddot{q} + C(q, \dot{q}) \dot{q} = \tau - g(q) $$
where $M(q)$ is the mass matrix, $C(q, \dot{q})$ is the Coriolis matrix, $g(q)$ is the gravity vector, and $\tau$ is the control input.

A radial basis function neural network (RBFNN) controller can be used to approximate the dynamics of the manipulator. The radial basis function neural network controller is given by:

$$f(x) = \Theta^T \Phi(x) + \epsilon(x)$$

where $\Theta$ is the weight vector, $\Phi(x)$ is the radial basis function, and $\epsilon(x)$ is the approximation error.

The control law is given by:

$$ \tau = \hat{\Theta}^T \Phi(x) - K_D s $$

and the RBFNN weight update law is given by:

$$ \dot{\hat{\Theta}} = -\Gamma \Phi(x) s^T $$

More details are provided in the paper TODO.

# BibTeX
If you use this package, please cite it using the following:
```
TODO
```

