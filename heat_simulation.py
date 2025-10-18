import numpy as np
from scipy.linalg import norm
from matplotlib import colormaps as cm
from matplotlib import pyplot as plt


#TASK23


# Constants

x0 = 0
xe = 8
Nx = 100  

y0 = 0
ye = 6
Ny = 90

x = np.linspace(x0, xe, Nx)
y = np.linspace(y0, ye, Ny)

X,Y = np.meshgrid(x,y,indexing='ij')

k = 2 # Thermal conductivity
 
dx = x[1] - x[0]  # x step size
dy = y[1] - y[0]  # y step size
lambda1 = 1  # for T1
lambda2 = 1  # for T2

# Coefficients for boundary conditions
A1 = k / lambda1  # coeff for T1
A2 = k / lambda2  # coeff for T2
const1 = 1 / (1 + A1)  
const2 = 1 / (1 + A2)  

#initial temperature
T0 = 0
T = np.ones_like(X)*T0  #Temperature matrix initialized to T0

T1 = 100  # Temperature of opposite faces when heated
T2 = -100  # Temperature of opposite faces when cooled

# Setting boundary conditions
T[:, 0] = T2   #Left boundary
T[:, -1] = T2  #Right boundary
T[0, :] = T1   #Top boundary
T[-1, :] = T1  #Bottom boundary


# temperature at corners - setting to average of neighboring cells
T[0, 0] = (T[0, 1] + T[1, 0]) / 2
T[-1, -1] = (T[-2, -1] + T[-1, -2]) / 2
T[-1, 0] = (T[-2, 0] + T[-1, 1]) / 2
T[0, -1] = (T[0, -2] + T[1, -1]) / 2

# Function to apply boundary conditions
def bc2_low(T):
    T[0, :] = const1 * (T[1, :] * A1 + T1) #top
    T[:, 0] = const2 * (T[:, 1] * A2 + T2) #left
    T[-1, :] = const1 * (T[-2, :] * A1 + T1) #bottom
    T[:, -1] = const2 * (T[:, -2] * A2 + T2) #right


Tc = T.copy()

# Function to perform a simulation step
def step(TT):
    TT[1:-1, 1:-1] = 0.25 * (TT[:-2, 1:-1] + TT[2:, 1:-1] + TT[1:-1, 2:] + TT[1:-1, :-2])


# Variables to control the iteration process
max_increment = []
more = True
success = 1e-9 #max increment between loops
too_many_loops = 1e5 #max number of iterations
count = 0
Tc = T.copy()

# Function to solve for the temperature distribution
def sol(more, count, Tc):
    while more:
        count += 1
        step(T)
        bc2_low(T)
        inc = np.max(np.abs(T - Tc)) # Calculate maximum increment
        max_increment.append(inc)
        Tc = T.copy()
        if  inc < success or count >= too_many_loops :
            # print(count)
            more = False
            
        
        
    return max_increment,count

max_increment,loops_performed = sol(more, count, Tc)

sol(more,count,Tc)

# Plotting temperature field 3d

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# tytul = "Temperature distribution"
# ax.plot_surface(X, Y, T, cmap='plasma', lw=0.5, edgecolor="k")
# ax.set(xlabel='x', ylabel='y', zlabel='z')
# ax.set_title(tytul)
# plt.show()

# Plotting the 2D temperature field
plt.figure()
plt.contourf(X, Y, T, cmap='plasma')
plt.colorbar(label='Temperature (°C)')
plt.xlabel('x (meters)')
plt.ylabel('y (meters)')
plt.title(f'Temperature Field T(x, y) after {loops_performed} Loops')
plt.show()

# Calculating the heat flux
gx,gy = np.gradient(T,dy,dx)
l = np.sqrt(gx**2+gy**2)
llim = 2
ratio = l/llim
ratio[ratio<1]=1
gxc = gx/ratio
gyc = gy/ratio

ev = 4 # Vector field sparsity factor

# Plotting the heat flux field
fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax1.quiver(X[::ev,::ev],Y[::ev,::ev],-gxc[::ev,::ev],-gyc[::ev,::ev], l[::ev,::ev], cmap='gist_ncar', units='dots')
ax1.set_title("Heat flux")
ax1.set(xlabel='x', ylabel='y')


plt.show()

#Performing the Gauss test

# Function to translate coordinates to grid indices
def translate_coordinates(x, y):
    i, j = round(x/dx), round(y/dy)
    return i, j

# Specifying the region for Gauss' theorem application
xg0 = 5
xge = 7
yg0 = 2
yge = 5

ig0, jg0 = translate_coordinates(xg0, yg0)
ige, jge = translate_coordinates(xge, yge)

# Creating the boundary coordinates of the region
xr = np.array([x[ig0], x[ig0], x[ige], x[ige], x[ig0]])
yr = np.array([y[jg0], y[jge], y[jge], y[jg0], y[jg0]])


# Calculating the flux through the boundaries of the region
botr = gy[ig0:ige + 1, jg0] * dx
botr[0] /= 2
botr[-1] /= 2
bot = np.sum(botr)

topr = gy[ig0:ige+1, jge]*dx
topr[0] /=2
topr[-1] /=2
top = np.sum(topr)

leftr = gx[ig0, jg0:jge+1]*dy
leftr[0] /=2
leftr[-1] /=2
left = np.sum(leftr)

rightr = gx[ige, jg0:jge+1]*dy
rightr[0] /=2
rightr[-1] /=2
right = np.sum(rightr)

total_flux = left - right + bot - top

print("total flux: ",total_flux)

# Plotting the temperature field and the region of interest
fig2 = plt.figure()
ax2 = fig2.add_subplot()
pcm = ax2.pcolormesh(X,Y,T,cmap='jet')
fig2.colorbar(pcm)
ax2.plot(xr,yr,'-k')
ax2.set_title(f'Temperature Field with Gauss Region after {loops_performed} Loops')
ax2.set_xlabel('x (meters)')
ax2.set_ylabel('y (meters)')

plt.show()


# DISCUSSION OF CORRECTNESS 
# For above situation the output value:2.7318626223404863e-06 represents 
# the total flux calculated using Gauss' theorem over the specified region. 
# This flux value being close to zero indicates that the numerical solution 
# for the heat equation has achieved a state where the net heat flux 
# into the region equals the net heat flux out of the region. 
# This is consistent with the steady-state solution of the heat equation, 
# where the temperature distribution no longer changes over time.


