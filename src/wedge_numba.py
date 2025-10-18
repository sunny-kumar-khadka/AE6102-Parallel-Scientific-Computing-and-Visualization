import numpy as np
import math 
import matplotlib.pyplot as plt
from numba import jit, float64, prange 
import  numba 
import time
from argparse import ArgumentParser
global R, gm, Cv
R = 287.0
gm = 1.4
Cv = R/(gm-1)

#---------Grid Generation ------------------------
def grid_generation(Ny, Nx, L1, L2, L3, h1, theta):
    L = L1 + L2 + L3
    h1_2 = np.tan(math.radians(theta))*L2
    x = np.zeros([Ny, Nx])
    y = np.zeros([Ny, Nx])
    for i in range(Ny):
          x[i, :] = np.linspace(0, L, Nx)
    for j in range(Nx):
          if x[0, j] <=L1:
             y[:, j] = np.linspace(0, h1, Ny)
          elif (x[0, j] > L1) & (x[0, j] <=L1+L2):
             y[:, j] = np.linspace(np.tan(math.radians(theta))* (x[0, j] - L1), h1, Ny)
          else:
             y[:, j] = np.linspace(h1_2, h1, Ny)  
    return x, y


#------------Edge length and normal calculations--------
def vertical_edge_calculation(grid_x, grid_y, Nx, Ny):
    vertical_edge_len = np.zeros((Ny-1, Nx))
    vertical_edge_nx = np.zeros_like(vertical_edge_len)
    vertical_edge_ny = np.zeros_like(vertical_edge_len)
    x0 = grid_x[ :-1, :]
    x1 = grid_x[1: , :]
    y0 = grid_y[ :-1, :]
    y1 = grid_y[1 :, :]
    vertical_edge_len = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    vertical_edge_nx = (y1 - y0)  / vertical_edge_len
    vertical_edge_ny = (x0- x1)  / vertical_edge_len
    return vertical_edge_len, vertical_edge_nx,  vertical_edge_ny

def horizontal_edge_calculation(grid_x, grid_y, Nx, Ny):
    horizontal_edge_len = np.zeros((Ny, Nx-1))
    horizontal_edge_nx = np.zeros_like(horizontal_edge_len)
    horizontal_edge_ny = np.zeros_like(horizontal_edge_len)
    x0 = grid_x[ : , : -1]
    x1 = grid_x[ : , 1: ]
    y0 = grid_y[ : , : -1]
    y1 = grid_y[ : , 1: ]
    horizontal_edge_len = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    horizontal_edge_nx = (y1 - y0)  / horizontal_edge_len
    horizontal_edge_ny = (x0- x1)  / horizontal_edge_len
    return horizontal_edge_len, horizontal_edge_nx, horizontal_edge_ny


#----------------Cell Geometry Calculation --------------
def center_and_volume(grid_x, grid_y):
    x1 = grid_x[ :-1, :-1]
    x2 = grid_x[ :-1,  1:]
    x3 = grid_x[1: ,  1: ]
    x4 = grid_x[1: ,  :-1]
    y1 = grid_y[ :-1, :-1]
    y2 = grid_y[ :-1,  1:]
    y3 = grid_y[1: ,  1: ]
    y4 = grid_y[1: ,  :-1]
    cell_x = 0.25*(x1 + x2 + x3 + x4)
    cell_y = 0.25*(y1 + y2 + y3 + y4)
    cell_volume = 0.5*abs((x1 - x3)*(y2 - y4) + (x4 - x2)*(y1 - y3))
    return cell_x, cell_y, cell_volume


#--------Helper function to calculate flux --------------
def conservative_var_from_input(u, v, Temp, rho):
    U = np.array([rho, rho*u, rho*v, rho*(Cv*Temp + 0.5*(u*u + v*v))])
    return U


@jit(nopython = True)
def normal_and_tangential(u, v, n):
    un = u*n[0] + v*n[1]
    ut = u*(-n[1]) + v*n[0]
    return un, ut

@jit(nopython = True)
def flux_from_variables(rho, un, ut, T, P):
    e = Cv*T
    F = np.array([rho*un, rho*un*un + P, rho*ut*un, rho*(e + 0.5*(un*un + ut*ut) + P/rho)*un])
    return F

@jit(nopython = True)
def tilda_quantity(var_L, var_R, rhoL, rhoR):
    var_tilda = (var_L*np.sqrt(rhoL) + var_R*np.sqrt(rhoR))/(np.sqrt(rhoL) + np.sqrt(rhoR))
    return var_tilda


#------------Calculation of primitive variables -------------
@jit(nopython = True, parallel = True)
def primitive_from_conservative(U):
    Prim_var = np.zeros((U.shape[0], U.shape[1], 5))
    for i in prange(U.shape[0]):
        for j in prange(U.shape[1]):
            Prim_var[i, j, 0] = U[i, j, 0]
            Prim_var[i, j, 1] = U[i, j, 1]/U[i, j, 0]
            Prim_var[i, j, 2] = U[i, j, 2]/U[i, j, 0]
            Prim_var[i, j, 4] = ((U[i, j, 3]/U[i, j, 0]) - 0.5*( Prim_var[i, j, 1]* Prim_var[i, j, 1] +Prim_var[i, j, 2]*Prim_var[i, j, 2]))/Cv
            Prim_var[i, j, 3] = Prim_var[i, j, 0]*Prim_var[i, j, 4]*R
    return Prim_var


#-------------Calculation of flux----------------------
def Roe_Flux(Prim_L, Prim_R, nx, ny):
    n = np.array([nx, ny])
    [rhoL, uL, vL, PL, TL] = Prim_L
    [rhoR, uR, vR, PR, TR] = Prim_R
    unL,utL = normal_and_tangential(uL, vL, n)
    unR, utR = normal_and_tangential(uR, vR, n)

    HL = Cv*TL + 0.5*(unL*unL + utL*utL) + PL/rhoL
    HR = Cv*TR + 0.5*(unR*unR + utR*utR) + PR/rhoR

    rho_tilda = np.sqrt(rhoL*rhoR)
    un_tilda = tilda_quantity(unL, unR, rhoL, rhoR)
    ut_tilda = tilda_quantity(utL, utR, rhoL, rhoR)
    H_tilda = tilda_quantity(HL, HR, rhoL, rhoR)
    a_tilda = math.sqrt((gm - 1)*(H_tilda-0.5*(un_tilda**2 + ut_tilda**2)))
    
    alpha1_tilda = ((PR - PL) - rho_tilda*a_tilda*(unR - unL))/(2*a_tilda*a_tilda)
    alpha2_tilda = (rhoR - rhoL) - (PR - PL)/(a_tilda*a_tilda)
    alpha3_tilda = rho_tilda*(utR - utL)
    alpha4_tilda = ((PR - PL) + rho_tilda*a_tilda*(unR - unL))/(2*a_tilda*a_tilda)

    K1 = np.array([1, un_tilda - a_tilda, ut_tilda, H_tilda - un_tilda*a_tilda])
    K2 = np.array([1, un_tilda, ut_tilda, 0.5*(un_tilda**2 + ut_tilda**2)])
    K3 = np.array([0, 0, 1, ut_tilda])
    K4 = np.array([1, un_tilda + a_tilda, ut_tilda, H_tilda + un_tilda*a_tilda])

    lemda1 = un_tilda - a_tilda
    lemda2 = un_tilda 
    lemda3 = un_tilda
    lemda4 = un_tilda + a_tilda
    
    Flux_L = flux_from_variables(rhoL, unL, utL, TL, PL)
    Flux_R = flux_from_variables(rhoR, unR, utR, TR, PR)
    
    F = 0.5*((Flux_L + Flux_R) - (alpha1_tilda*abs(lemda1)*K1 + alpha2_tilda*abs(lemda2)*K2 + alpha3_tilda*abs(lemda3)*K3 + alpha4_tilda*abs(lemda4)*K4))
    
    Flux = np.array([F[0], F[1]*n[0] - F[2]*n[1],  F[1]*n[1] + F[2]*n[0], F[3] ])
    return Flux

@jit(nopython = True)
def wall_flux(Prim_L, nx, ny):
    [rhoL, uL, vL, PL, TL] = Prim_L
    Flux = np.array([0, PL*nx, PL*ny, 0])
    return Flux


#-----------RSS calculation and visualization --------------
def RSS(change, original):
    relative_change = (change /original)**2
    Rss = math.sqrt(np.sum(relative_change)/(relative_change.shape[0]*relative_change.shape[1]))
    return Rss


def max_and_min(variables):
    max_rows = np.ndarray((variables.shape[0], 1))
    min_rows = np.ndarray((variables.shape[0], 1))
    for i in range(variables.shape[0]):
        max_rows[i, 0] = max(variables[i, :])
        min_rows[i, 0] = min(variables[i, :])
    max_variables = max(max_rows)
    min_variables = min(min_rows)
    return max_variables[0], min_variables[0]


def Visualization(Prim_var, cell_x, cell_y): 
    max_density, min_density = max_and_min(Prim_var[:, :, 0])
    max_T, min_T = max_and_min(Prim_var[:, :, 4])
    max_P, min_P = max_and_min(Prim_var[:, :, 3])
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].contourf(cell_x, cell_y, Prim_var[:, :, 0], levels = np.linspace(min_density, 1.1*max_density, 40))  
    ax[0, 1].contourf(cell_x, cell_y, Prim_var[:, :, 4], levels = np.linspace(min_T, 1.1*max_T, 40))
    ax[1, 0].contourf(cell_x, cell_y, Prim_var[:, :, 3], levels = np.linspace(min_P, 1.1*max_P, 40))     
    ax[1, 1].quiver(cell_x, cell_y, Prim_var[:, :, 1], Prim_var[:, :, 2])
    plt.show()


#---------Fluc calculations at all edges-----------------
#@jit(nopython = True, parallel = True)
#@numba.njit(parallel=True)
def vertical_flux_calculations(Prim_var, Prim_var_inlet, vertical_edge_len, vertical_edge_nx, vertical_edge_ny, Nx, Ny):
    vertical_flux = np.zeros((Ny-1, Nx, 4))
    for i in range(Ny-1):
        for j in range(Nx):
                if (j == 0):
                    vertical_flux[i, j, :] = Roe_Flux(Prim_var_inlet, Prim_var[i, j, :], vertical_edge_nx[i, j], vertical_edge_ny[i, j])*vertical_edge_len[i, j]
                elif (j == vertical_edge_len.shape[1] - 1):
                    vertical_flux[i, j, :] = Roe_Flux(Prim_var[i, j-1, :], Prim_var[i, j-1, :], vertical_edge_nx[i, j], vertical_edge_ny[i, j])*vertical_edge_len[i, j]
                else:
                    vertical_flux[i, j, :] = Roe_Flux(Prim_var[i, j-1, :], Prim_var[i, j, :], vertical_edge_nx[i, j], vertical_edge_ny[i, j])*vertical_edge_len[i, j]
    return vertical_flux


#@jit(nopython = True, parallel = True)
def horizontal_flux_calculations(Prim_var, Prim_var_inlet, horizontal_edge_len, horizontal_edge_nx, horizontal_edge_ny, Nx, Ny):
     horizontal_flux = np.zeros((Ny, Nx-1, 4))
     for i in range(Ny):
           for j in range(Nx-1):
                if (i == 0):
                        horizontal_flux[i ,j, :] = wall_flux(Prim_var[i, j, :], horizontal_edge_nx[i, j], horizontal_edge_ny[i, j])*horizontal_edge_len[i, j]  
                elif (i == horizontal_edge_len.shape[0] - 1):
                        horizontal_flux[i, j, :] = wall_flux(Prim_var[i-1, j, :], horizontal_edge_nx[i, j], horizontal_edge_ny[i, j])*horizontal_edge_len[i, j]
                else:     
                        horizontal_flux[i, j, :] = Roe_Flux(Prim_var[i, j, :], Prim_var[i-1, j, :], horizontal_edge_nx[i, j], horizontal_edge_ny[i, j])*horizontal_edge_len[i, j]
     return horizontal_flux


#-----------Cell Variables Update----------------
def cell_variables_update(horizontal_flux, vertical_flux, U, dt, cell_volume):
    U[:, :, :] =  U[:, :, :] - (dt/cell_volume[:, :, :])*(vertical_flux[ : , 1 : , :] - vertical_flux[ : ,  : -1, :] + horizontal_flux[ : -1, :] - horizontal_flux[1 : , :])
    return U


def wedge_main_numba(Ny, Nx, L1, L2, L3, h1, theta, plot):
    Ny, Nx, L1, L2, L3, h1, theta = 36, 78, 1, 9, 3, 6, 15
    grid_x, grid_y = grid_generation(Ny, Nx, L1, L2, L3, h1, theta)

    vertical_edge_len, vertical_edge_nx,  vertical_edge_ny =  vertical_edge_calculation(grid_x, grid_y, Nx, Ny)
    horizontal_edge_len, horizontal_edge_nx, horizontal_edge_ny = horizontal_edge_calculation(grid_x, grid_y, Nx, Ny)

    cell_volume = np.zeros((Ny-1, Nx-1))
    cell_x = np.zeros_like(cell_volume)
    cell_y = np.zeros_like(cell_volume) 
    
    cell_x, cell_y, cell_volume = center_and_volume(grid_x, grid_y)
    cell_volume = np.expand_dims(cell_volume, axis = 2)
    cell_volume_ = np.repeat(cell_volume, 4, axis = 2)

    U = np.ndarray((Ny-1, Nx-1, 4))
    Residual = np.ndarray((Ny-1, Nx-1, 4))
    Prim_var = np.ndarray((Ny-1, Nx-1, 5)) # rho, u, v, P, T
    vertical_flux = np.zeros((Ny-1, Nx, 4))
    horizontal_flux = np.zeros((Ny, Nx-1, 4))

    dt = 0.00005
    U_initial = conservative_var_from_input(800, 0, 300, 1.2)
    U_inlet  = conservative_var_from_input(800, 0, 300, 1.2)
    Prim_var_inlet = np.array([1.2, 800, 0, 103320, 300])
    #initialization
    U[:, :, :] = U_initial
    Prim_var = primitive_from_conservative(U)

    for nt in range(3000):
        vertical_flux  = vertical_flux_calculations(Prim_var, Prim_var_inlet, vertical_edge_len, vertical_edge_nx, vertical_edge_ny, Nx, Ny)
        horizontal_flux = horizontal_flux_calculations(Prim_var, Prim_var_inlet, horizontal_edge_len, horizontal_edge_nx, horizontal_edge_ny, Nx, Ny)
        density_0 = np.ndarray((Ny - 1, Nx -1))
        density_0[:, :] = Prim_var[:, :, 0]
        U = cell_variables_update(horizontal_flux, vertical_flux, U, dt, cell_volume)
        Prim_var = primitive_from_conservative(U)
        change_density = Prim_var[:, :, 0] - density_0
        Rss_density = RSS(change_density, density_0)
        print(nt, Rss_density) 
        if Rss_density < 1e-05:
            break

    if plot == True:
        Visualization(Prim_var, cell_x, cell_y)



if __name__ == '__main__':
    Ny, Nx, L1, L2, L3, h1, theta = 36, 78, 1, 9, 3, 6, 15
    p = ArgumentParser(description='Jacobi method for Laplace equation')
    p.add_argument('--nx', type=int, default=36, help='num of discretized points in X')
    p.add_argument('--ny', type=int, default=78, help='num of discretized points in Y')
    p.add_argument('--L1', type=float, default=1, help='length of first segment')
    p.add_argument('--L2', type=float, default=9, help='length of second segment')
    p.add_argument('--L3', type=float, default=3, help='length of third segment')
    p.add_argument('--h1', type=float, default=6, help='height')
    p.add_argument('--theta', type=float, default=15, help='Angle of bend in degrees')
    p.add_argument('--plot', choices=[True, False],
               default=True, help='Plot the results')
    p.add_argument(
        '--output-dir', type=str, default='.',
        help='Output directory to generate file.'
    )
    args = p.parse_args()
    wedge_main_numba(args.ny, args.nx, args.L1, args.L2, args.L3, args.h1, args.theta, plot=args.plot)  