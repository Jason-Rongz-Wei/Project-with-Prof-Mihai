import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Load the uploaded CSV file
file_path = '/Users/weirongze/Documents/芝加哥/课件/Research/Mihai Anitescu/U_optimized.csv'
data = pd.read_csv(file_path, header=None)

# Convert DataFrame to NumPy array for plotting
U = data.values
x = np.linspace(0, 1, U.shape[1])
t = np.linspace(0, 1, U.shape[0])
X, T = np.meshgrid(x, t)

# Plot the surface
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, U, cmap='viridis')

ax.set_xlabel('Space (x)')
ax.set_ylabel('Time (t)')
ax.set_zlabel('U(x,t)')
ax.set_title('Surface Plot of U(x,t)')

plt.show()
