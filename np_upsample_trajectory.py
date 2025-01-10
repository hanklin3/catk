# %%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define 3 points (x, y) as a trajectory
x_points = torch.tensor([0.0, 0.5, 1.0])
y_points = torch.tensor([0.0, 0.8, 0.2])

# Combine x and y into a 2D tensor (1, 2, L) for interpolation, 2 for x and y, L for number of points
trajectory = torch.stack([x_points, y_points], dim=0).unsqueeze(0)  # Shape: (1, 2, 3)

# Interpolate to 5 points using linear interpolation
upsampled_trajectory = F.interpolate(trajectory, size=5, mode='linear', align_corners=True)

# Add curvature by applying a non-linear transformation (e.g., sine)
curvature_factor = 0.5  # Adjust curvature intensity
upsampled_trajectory[0, 1, :] += curvature_factor * torch.sin(torch.linspace(0, 3.14, 5))

# Extract the upsampled x and y coordinates
upsampled_x = upsampled_trajectory[0, 0, :].detach()  # Detach for plotting
upsampled_y = upsampled_trajectory[0, 1, :].detach()

# Plot the original and upsampled points
plt.figure(figsize=(6, 6))
plt.plot(x_points, y_points, 'ro-', label='Original Points')  # Original points in red
plt.plot(upsampled_x, upsampled_y, 'bo-', label='Upsampled Points (Curved)')  # Upsampled points in blue
plt.title('Trajectory Upsampling with Curvature')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()


# %%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Generate training data
x_train = torch.linspace(-2 * 3.14, 2 * 3.14, 1000).unsqueeze(1)  # Inputs: Shape (1000, 1)
y_train = torch.sin(x_train)  # Targets: sin(x)

# Define MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),  # Input layer (1 -> 64 neurons)
            nn.ReLU(),         # Activation function
            nn.Linear(64, 64), # Hidden layer (64 -> 64 neurons)
            nn.ReLU(),         # Activation function
            nn.Linear(64, 1)   # Output layer (64 -> 1 neuron)
        )

    def forward(self, x):
        return self.layers(x)

# Initialize the model, loss function, and optimizer
model = MLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(x_train)  # Forward pass
    loss = criterion(y_pred, y_train)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

# Test the model on new data
x_test = torch.linspace(-2 * 3.14, 2 * 3.14, 500).unsqueeze(1)
y_test_pred = model(x_test).detach()

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(x_test, torch.sin(x_test), label='True sin(x)', color='blue')
plt.plot(x_test, y_test_pred, label='MLP Approximation', color='red', linestyle='--')
plt.legend()
plt.title('MLP Approximation of sin(x)')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid()
plt.show()

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define 3 points (x, y) as a trajectory
x_points = torch.tensor([0.0, 0.5, 1.0])
y_points = torch.tensor([0.0, 0.8, 0.2])

# Combine x and y into a 2D tensor (1, 2, L) for interpolation
trajectory = torch.stack([x_points, y_points], dim=0).unsqueeze(0)  # Shape: (1, 2, 3)

# Interpolate to 5 points using linear interpolation
upsampled_trajectory = F.interpolate(trajectory, size=5, mode='linear', align_corners=True)

# Define an MLP for trajectory refinement
class TrajectoryRefiner(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=2):
        super(TrajectoryRefiner, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

# Initialize the MLP
mlp = TrajectoryRefiner()

# Prepare input for the MLP
# Flatten the (1, 2, 5) tensor to (5, 2) for processing
mlp_input = upsampled_trajectory.squeeze(0).T  # Shape: (5, 2)

# Pass through the MLP
refined_trajectory = mlp(mlp_input)  # Shape: (5, 2)

# Plot the original, interpolated, and refined points
plt.figure(figsize=(6, 6))
plt.plot(x_points, y_points, 'ro-', label='Original Points')  # Original points in red
plt.plot(upsampled_trajectory[0, 0], upsampled_trajectory[0, 1], 'bo--', label='Interpolated Points')  # Interpolated points in blue
plt.plot(refined_trajectory[:, 0].detach(), refined_trajectory[:, 1].detach(), 'go-', label='Refined Points')  # Refined points in green
plt.title('Trajectory Upsampling and Refinement')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()

# %%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Generate random points for a batch
batch_size = 3
num_original_and_upsampled_points = [(1, 2), (2, 3), (3, 5), (5, 8), (8, 10)]
num_original_and_upsampled_points = [(2, 10), (4, 10), (7, 10), (8, 10), (9, 10)]
num_original_points, num_upsampled_points = num_original_and_upsampled_points[1]
# torch.manual_seed(0)  # For reproducibility

################## Method 1: Random points
# Generate random (x, y) points for each batch
x_points = torch.rand(batch_size, num_original_points)
y_points = torch.rand(batch_size, num_original_points)

# sort x_points in ascending order, no overlap
x_points, _ = torch.sort(torch.rand(batch_size, num_original_points), dim=1)


# ################## Method 2: U-shape parabola
# Generate x-points
x_points = torch.linspace(-1, 1, num_original_points).repeat(batch_size, 1)  # Symmetric around 0

# Generate y-points for a perfect U-shape using a parabola
# y = -a * x^2 + k, where a controls sharpness and k is the height of the U
a = 2.0  # Sharpness
k = 1.0  # Height of the U
y_points = -a * x_points**2 + k  # U-shape parabola


# Combine x and y into a trajectory tensor of shape (batch_size, 2, num_original_points)
trajectory = torch.stack([x_points, y_points], dim=1)  # Shape: (batch_size, 2, num_original_points)

# Interpolate to 5 points using linear interpolation
upsampled_trajectory = F.interpolate(trajectory, size=num_upsampled_points, mode='linear', align_corners=True)

# Add a sine-based curvature to the y-coordinates of the upsampled trajectory
curvature_factor = 0.3  # Adjust curvature intensity
sine_wave = curvature_factor * torch.sin(torch.linspace(0, 3.14, num_upsampled_points))
# upsampled_trajectory[:, 1, :] += sine_wave  # Apply sine adjustment to y-coordinates

# Plot the original, interpolated, and curved trajectories for each batch
plt.figure(figsize=(10, 10))
for i in range(batch_size):
    plt.subplot(batch_size, 1, i + 1)
    # Original points
    plt.plot(x_points[i], y_points[i], 'ro-', label='Original Points')
    # Interpolated points with curvature
    plt.plot(upsampled_trajectory[i, 0, :], upsampled_trajectory[i, 1, :], 'bo--', label='Curved Points')
    plt.title(f"Batch {i+1}")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid()
    plt.axis('equal')

plt.tight_layout()
plt.show()

# %%
