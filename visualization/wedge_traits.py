from traits.api import HasTraits, Instance, Int, Enum, Range
from traitsui.api import View, Item, Group 
import numpy as np
import math 
import matplotlib.pyplot as plt 
from chaco.api import ArrayPlotData, Plot
from enable.api import ComponentEditor


class ActorViewer(HasTraits):

    plot = Instance(Plot)
    Ny = Int(36)
    Nx = Int(78)
    L1 = Range(1, 5, 1)
    L2 = Range(8, 15, 9)
    L3 = Range(1, 10, 3)
    h1 = Enum(6,7,8,9)
    theta = Range(5, 25, 15)

    view = View(
                Group(
                    Item('plot', editor=ComponentEditor(),show_label=False, width=1200, height=500, resizable=True),
                    Item('Ny', label="Ny"),
                    Item('Nx', label="Nx"),
                    Item('L1', label="L1"),
                    Item('L2', label="L2"),
                    Item('L3', label="L3"),
                    Item('h1', label="h1"),
                    Item('theta', label="theta"),
                # resizable=True,
                #title = "Wedge"
                )
    )

    
    def grid_generation(self, Ny, Nx, L1, L2, L3, h1, theta):
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


    def _plot_default(self):
        self.grid_x, self.grid_y =self.grid_generation(self.Ny, self.Nx, self.L1, self.L2, self.L3, self.h1, self.theta)
        self.plotdata = ArrayPlotData(x = self.grid_x.flatten(), y = self.grid_y.flatten())
        self.plot = Plot(self.plotdata)
        self.plot.plot(("x", "y"), type="scatter", color="red") 
        return self.plot


    def _Ny_changed(self):
        self.grid_x, self.grid_y =self.grid_generation(self.Ny, self.Nx, self.L1, self.L2, self.L3, self.h1, self.theta)
        self.plotdata.set_data("x", self.grid_x.flatten()) 
        self.plotdata.set_data("y", self.grid_y.flatten())

    def _Nx_changed(self):
        self.grid_x, self.grid_y =self.grid_generation(self.Ny, self.Nx, self.L1, self.L2, self.L3, self.h1, self.theta)
        self.plotdata.set_data("x", self.grid_x.flatten()) 
        self.plotdata.set_data("y", self.grid_y.flatten())

    def _L1_changed(self):
        self.grid_x, self.grid_y =self.grid_generation(self.Ny, self.Nx, self.L1, self.L2, self.L3, self.h1, self.theta)
        self.plotdata.set_data("x", self.grid_x.flatten()) 
        self.plotdata.set_data("y", self.grid_y.flatten())

    def _L2_changed(self):
        self.grid_x, self.grid_y =self.grid_generation(self.Ny, self.Nx, self.L1, self.L2, self.L3, self.h1, self.theta)
        self.plotdata.set_data("x", self.grid_x.flatten()) 
        self.plotdata.set_data("y", self.grid_y.flatten())

    def _L3_changed(self):
        self.grid_x, self.grid_y =self.grid_generation(self.Ny, self.Nx, self.L1, self.L2, self.L3, self.h1, self.theta)
        self.plotdata.set_data("x", self.grid_x.flatten()) 
        self.plotdata.set_data("y", self.grid_y.flatten())

    def _h1_changed(self):
        self.grid_x, self.grid_y =self.grid_generation(self.Ny, self.Nx, self.L1, self.L2, self.L3, self.h1, self.theta)
        self.plotdata.set_data("x", self.grid_x.flatten()) 
        self.plotdata.set_data("y", self.grid_y.flatten())
    
    def _theta_changed(self):
        self.grid_x, self.grid_y =self.grid_generation(self.Ny, self.Nx, self.L1, self.L2, self.L3, self.h1, self.theta)
        self.plotdata.set_data("x", self.grid_x.flatten()) 
        self.plotdata.set_data("y", self.grid_y.flatten())


if __name__ == '__main__':
    a = ActorViewer()
    a.configure_traits() 