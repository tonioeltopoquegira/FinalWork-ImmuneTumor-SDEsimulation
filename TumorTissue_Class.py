import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.integrate import solve_ivp
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import norm
from scipy.stats import gaussian_kde
from scipy.optimize import minimize_scalar
    

# Palette 

import matplotlib.cm as cm

num_colors = 60
cmap = cm.get_cmap('YlOrRd', num_colors)
#colors = [cmap(i) for i in range(5, num_colors, 5)]
colors = ['limegreen', 'seagreen', 'darkgreen', 'olivedrab', 'darkkhaki', 'darkred']

cmap = cm.get_cmap('Blues', num_colors)
blues = [cmap(i) for i in range(0, num_colors, 6)]

palette = sns.cubehelix_palette(start=240, rot=-0.5, dark=0.6, light=0.8, reverse=True, as_cmap=False)
blue_to_green = palette[10::-1]


# Sheather Jones optimized bandwidth

def sheather_jones_loglikelihood(h, data):
    kde = gaussian_kde(data, bw_method=h)
    return -np.sum(np.log(kde.evaluate(data)))

class tumor_tissue_simulation:

    def __init__(self, params, x0, model, dt, dx, sim_time,  space_grid):
        if model != "generic" and model != "beta":
            raise Exception('Tipologia di modello errata: "generic" o "beta"')

        self.params = params
        self.alpha, self.beta, self.theta, self.sigma = params
        self.x0 = x0
        self.dt = dt
        self.dx = dx
        self.sim_time = sim_time
        self.space_grid = space_grid
        self.model = model # "generic" or "beta"

        self.t_steps = int(self.sim_time/self.dt)
        self.x_steps = int(self.space_grid/self.dx)
        self.x_values = np.arange(0, self.space_grid, self.dx)
        self.t_values = np.arange(0, self.sim_time, self.dt)
        self.densities_FP = np.zeros((self.x_steps, self.t_steps))
        self.sim_mean_var = np.zeros((2,self.t_steps))
    
    def initialize(self):
        self.densities_FP[:,:] = np.zeros((self.x_steps, self.t_steps))
        self.sim_mean_var[:] = np.zeros((2,self.t_steps))
    
    def Dx(self, x, beta = None):
        if beta is not None:
            return self.alpha + (1-self.theta*x)*x - beta*x/(1+x)
        return self.alpha + (1-self.theta*x)*x - self.beta*x/(1+x)

    def d2x(self, x):
        return 1-2*self.theta*x - (self.beta - 1)/(1+x**2+2*x)
    
    def dx_ivp(self, t, x):
        return self.alpha + (1-self.theta*x)*x - self.beta*x/(1+x)
    
    def diff_2ndmodel(self, x):
        return x/(1+x)
    
    def d2diff_2ndmodel(self,x):
        return x/((x+1)**2)

    # Find the steady states of the deterministic model: dX/dt = 0. Bifurcation paramter analysis.

    def ODE_steadystates(self, b_vals, graph):

        b_vals.append((self.theta+1)**2/(4*self.theta))
        b_vals.append(self.beta)
        b_vals.sort()

        if graph:
            
            plt.clf()
            plt.title('Phase portrait')
            plt.xlabel('Tumor cells population')
            plt.ylabel('Population growth')
            plt.plot(self.x_values, np.zeros(self.x_steps), 'k--', linewidth = 1.0)
            

        for b in range(0, len(b_vals)):
            if graph and b_vals[b] == (1+self.theta)**2/(4*self.theta):
                plt.plot(self.x_values, self.Dx(self.x_values, b_vals[b]), '-', color = 'blue', label=f'Bifurcation Boundary: {round(b_vals[b], 3)}')
            if graph and b_vals[b] != (1+self.theta)**2/(4*self.theta):
                plt.plot(self.x_values, self.Dx(self.x_values, b_vals[b]), '-', color = colors[b], label = b_vals[b])

        plt.legend(frameon = True, fontsize = 'x-small')
        plt.ylim(-10, 5)
        plt.xlim(0, 10)
        plt.show()

    # Find and trace the deterministic model solutions for different initial values.

    def ODE_solutions(self, x0_vals = np.linspace(0,7.5,30), format = 'k-', label = False):

        plt.clf()
        for x in x0_vals:
            sol = solve_ivp(fun=self.dx_ivp, y0 = [x], t_span=[0.0, int(self.sim_time)], method = 'BDF') 

            if label:
                plt.plot(sol.t, sol.y[0], format, label = 'ODE solution', linewidth = 2.0)
            else:
                if x == self.x0:
                    plt.plot(sol.t, sol.y[0], 'r-', linewidth = 1.5, label = 'Initial value X = 2')
                else:
                    plt.plot(sol.t, sol.y[0], format, linewidth = 1.5)

        plt.xlabel('Time in days')
        plt.ylabel('Number of tumor cells in unit volume')
        plt.title('Solutions Curves of the ODE')
        plt.legend()
        plt.show()

    # Solve the Fokker Planck PDE for the density with the finite differences method
        
    def solve_Fokk_Planck(self):

        plt.clf()

        # Initialize distribution at 0. Boundary conditions are probability equal to 0

        p = np.zeros((self.x_steps, self.t_steps))
        
        probabilities = np.exp((-0.5 * ((self.x_values-self.x0) / 0.025) ** 2))
        p[:,0] = np.transpose(probabilities / np.sum(probabilities))
        plt.plot(self.x_values, p[:,0], color = 'blue', label = 'Starting distribution')
        
        i = 0
        # Update Rule
        for t in range(0, self.t_steps-1):

            p[0, t+1] = p[0, t] + self.dt/self.dx *  ((- p[1, t]*self.Dx(self.x_values[1]) + p[0, t]*self.Dx(self.x_values[0])) + self.sigma**2 * (p[0,t] - 2*p[1,t] + p[2,t])/self.dx)

            for m in range(1, self.x_steps-1):

                p[m, t+1] = p[m, t] + self.dt/(2*self.dx) * (-p[m+1,t]*self.Dx(self.x_values[m+1]) + p[m-1, t]*self.Dx(self.x_values[m-1]) + self.sigma**2 * (p[m+1,t] - 2*p[m,t] + p[m-1,t])/self.dx)

            p[:,t+1] = np.clip(p[:,t+1], 0, None)
            p[:,t+1] = p[:,t+1] /np.sum(p[:,t+1])

            
            if t+1 in (100, 300, 500, 700, 3000, 5500):
                plt.plot(self.x_values, p[:,t+1], color = colors[i], label = f'After {int((t+1)*self.dt)} days')
                i+=1

        p[:,-1] = np.clip(p[:,t+1], 0, None)
        p[:,-1] = p[:,t+1] /np.sum(p[:,t+1])

        self.densities_FP[:,:] = p

        plt.plot(self.x_values, p[:,-2], color ='red', label='Stationary')
        plt.xlim(0, self.x0+0.5)
        plt.title('PDF with Fokker Planck')
        plt.xlabel('Tumor cells population')
        plt.legend()
        plt.show() 
    
    def simulate_SDE_Milstein(self, n_run, graph = True, show_ODE = False, percentiles = (90, 10)):

        x = np.zeros((n_run+len(percentiles)+2, self.t_steps))
        x[:, 0] = self.x0
        for _ in range(n_run):
            dBt = np.random.normal(0, self.sigma, self.t_steps)
            for i in range(self.t_steps-1):

                if self.model == 'generic':
                    x[_, i+1] = x[_, i] + self.Dx(x[_, i])*self.dt + dBt[i] + 0.5 * self.Dx(x[_, i]) * self.d2x(x[_, i]) * (dBt[i]**2 - self.dt)
                    
                if self.model == 'beta':
                    x[_, i+1] = x[_, i] + self.Dx(x[_, i])*self.dt + self.diff_2ndmodel(x[_,i])*dBt[i] + 0.5 * self.diff_2ndmodel(x[_, i]) * self.d2diff_2ndmodel(x[_, i]) * (dBt[i]**2)
                    

        x[-1,:] = [np.mean(x[:,_]) for _ in range(self.t_steps)]
        x[-2,:] = [np.std(x[:,_]) for _ in range(self.t_steps)]

        for p in range(len(percentiles)):
            x[-3-p,:] = [np.percentile(x[:,_], percentiles[p]) for _ in range(self.t_steps)]

        if graph:

            """ for _ in range(0,n_run,250):
                plt.plot(self.t_values, x[_,:], 'k--', linewidth = 0.75)
             """
            plt.plot(self.t_values, x[-1,:], 'r-', label = 'mean')

            for p in range(len(percentiles)):
                plt.plot(self.t_values, x[-3-p,:], color = colors[p], label =f'{percentiles[p]}th percentile')

            
            plt.xlabel('Time in days')
            plt.ylabel('Tumor cells population size')
            if show_ODE:
                sol = solve_ivp(fun=self.dx_ivp, y0 = [self.x0], t_span=[0.0, int(self.sim_time)], method = 'BDF') 
                plt.plot(sol.y[0], 'r--', label = 'ODE trajectory')

            plt.xlim(0, self.sim_time)
            plt.legend()
            plt.title('Second Model Simulations Statistics')
            plt.show() 

        self.sim_mean_var[:,:] = x[-2:-1, :]
        
        return x[0:n_run+1,:]

    def MC_density_estimation(self, t, mc_sim):

        tt = int(t / self.dt) 

        # CLT approximation
        m = self.sim_mean_var[0,tt] 
        std = self.sim_mean_var[1,tt]
        pdf_est = norm.pdf(self.x_values, m, std)
        plt.plot(self.x_values, pdf_est, 'k--', label = 'PDF approximated to Gaussian (CLT)')

        # SNS KDE
        sns.kdeplot(data = mc_sim[:,tt], color = 'red', label = 'PDF estimated with KDE', clip = [0, 15]) 
        plt.title(f'Probability density function after {t} days')

        # SCIPY 

        result = minimize_scalar(sheather_jones_loglikelihood, args=(mc_sim[:,tt],), method='bounded', bounds=(0.01, 1))
        kde = gaussian_kde(mc_sim[:,tt], bw_method=result.x)
        scipy_pdf = kde.evaluate(self.x_values)
        plt.plot(self.x_values, scipy_pdf, color = 'blue', linewidth = 1.0, label='KDE with Sheater-Jones bandwidth')

        plt.xlim(0, 15)
        plt.legend()
        plt.show()
    
    def density_evolution(self, instants, mc_sim):
        
        for t in range(len(instants)):
            tt = int(instants[t] / self.dt) - 1

            if instants[t] == 0:
                result = minimize_scalar(sheather_jones_loglikelihood, args=(mc_sim[:,0],), method='bounded', bounds=(0.01, 1))
                kde = gaussian_kde(mc_sim[:,0], bw_method=result.x)
                scipy_pdf = kde.evaluate(self.x_values)
                plt.plot(self.x_values, scipy_pdf, color = colors[t], linewidth = 1.0, label=f'After {instants[t]} days')
                continue
            
            # SCIPY AND SJ bandwidth
            result = minimize_scalar(sheather_jones_loglikelihood, args=(mc_sim[:,tt],), method='bounded', bounds=(0.01, 1))
            kde = gaussian_kde(mc_sim[:,tt], bw_method=result.x)
            scipy_pdf = kde.evaluate(self.x_values)
            
            plt.plot(self.x_values, scipy_pdf, color = colors[t], linewidth = 1.0, label=f'After {instants[t]} days')

            
        plt.xlim(0, np.max(mc_sim[:,tt])+3)
        plt.title(f'Evolution of the pdf estimated with KDE')
        plt.legend()
        plt.show()

