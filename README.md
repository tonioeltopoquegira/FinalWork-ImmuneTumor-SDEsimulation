# FinalWork: Immune-Tumor interaction SDE model and simulation
Abstract: This undergraduate thesis explores the concepts of Brownian motion, stochastic differential equations (SDEs), and their applications in modeling and simulation. The theory section provides an overview of Brownian motion as a special Markovian stochastic process, along with a discussion of Ito's lemma and stochastic integrals. The focus then shifts to stochastic differential equations, which are used to describe the evolution of systems subject to random fluctuations.

In the application section, a stochastic differential equation model is developed to study the dynamics of the immune-tumor system in a random environment. This model supports the examination of the system's behavior under varying conditions and parameters.

Starting with a qualitative study of the ordinary differential equation, the thesis analyzes the deterministic dynamics, the equilibrium points and the role of parameters. After defining the stochastic model, the thesis showcases two different methods to derive the probability density of the solution of the stochastic differential equation: a discrete numerical method for the solution of the Fokker-Planck PDE and Monte Carlo simulations exploiting the Milstein method and Kernel Density Estimation techniques.

Lastly, the thesis proposes an alternative model formulation treating the number of immune system cells as a random variable.

The report ca be followed running the jupyter notebook Notebook_Thesis.ipynb. The class containing all the methods is defined in TumorTissue_Class.py
