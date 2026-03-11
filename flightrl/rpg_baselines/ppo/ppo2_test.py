import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import yaml
import pyquaternion
import os
from scipy.spatial.transform import Rotation as R


# Gate class for better visualization
class Gate:
    def __init__(self, pos, att, size=1.45):
        self.pos = np.array(pos)
        self.att = pyquaternion.Quaternion(w=att[0], x=att[1], y=att[2], z=att[3])
        self.gate_dim = size
        self.w_border = 0.3
        self.yaw = 2.0 * np.arcsin(self.att.z)
        self.tl_corner = self.pos + self.att.rotate(
            np.array([0.0, 1.0, 1.0]) * self.gate_dim / 2.0
        )
        self.tr_corner = self.pos + self.att.rotate(
            np.array([0.0, -1.0, 1.0]) * self.gate_dim / 2.0
        )
        self.bl_corner = self.pos + self.att.rotate(
            np.array([0.0, 1.0, -1.0]) * self.gate_dim / 2.0
        )
        self.br_corner = self.pos + self.att.rotate(
            np.array([0.0, -1.0, -1.0]) * self.gate_dim / 2.0
        )

    def is_passed(self, pos: np.ndarray) -> bool:
        drone_pos_in_gate_frame = self.att.inverse.rotate(pos - self.pos)
        return (
            drone_pos_in_gate_frame[0] > 0.05
            and abs(drone_pos_in_gate_frame[2]) < 1.0
            and abs(drone_pos_in_gate_frame[1]) < 1.0
        )

    def __repr__(self):
        return "Gate at [%.2f, %.2f, %.2f] with yaw %.2f deg." % (
            self.pos[0],
            self.pos[1],
            self.pos[2],
            180.0 / np.pi * self.yaw,
        )

    def draw(self, ax, color='black'):
        # helper corners for drawing
        tl_outer = self.pos + self.att.rotate(
            np.array([0.0, 1.0, 1.0]) * self.gate_dim / 2.0
            + np.array([0.0, 1.0, 1.0]) * self.w_border
        )
        tr_outer = self.pos + self.att.rotate(
            np.array([0.0, -1.0, 1.0]) * self.gate_dim / 2.0
            + np.array([0.0, -1.0, 1.0]) * self.w_border
        )
        tl_lower = self.pos + self.att.rotate(
            np.array([0.0, 1.0, 1.0]) * self.gate_dim / 2.0
            + np.array([0.0, 1.0, 0.0]) * self.w_border
        )
        tr_lower = self.pos + self.att.rotate(
            np.array([0.0, -1.0, 1.0]) * self.gate_dim / 2.0
            + np.array([0.0, -1.0, 0.0]) * self.w_border
        )
        bl_outer = self.pos + self.att.rotate(
            np.array([0.0, 1.0, -1.0]) * self.gate_dim / 2.0
            + np.array([0.0, 1.0, -1.0]) * self.w_border
        )
        br_outer = self.pos + self.att.rotate(
            np.array([0.0, -1.0, -1.0]) * self.gate_dim / 2.0
            + np.array([0.0, -1.0, -1.0]) * self.w_border
        )
        bl_upper = self.pos + self.att.rotate(
            np.array([0.0, 1.0, -1.0]) * self.gate_dim / 2.0
            + np.array([0.0, 1.0, 0.0]) * self.w_border
        )
        br_upper = self.pos + self.att.rotate(
            np.array([0.0, -1.0, -1.0]) * self.gate_dim / 2.0
            + np.array([0.0, -1.0, 0.0]) * self.w_border
        )

        x = [tl_outer[0], tr_outer[0], tr_lower[0], tl_lower[0]]
        y = [tl_outer[1], tr_outer[1], tr_lower[1], tl_lower[1]]
        z = [tl_outer[2], tr_outer[2], tr_lower[2], tl_lower[2]]
        verts = [list(zip(x, y, z))]
        poly = Poly3DCollection(verts)
        poly.set_color(color)
        ax.add_collection3d(poly)
        
        x = [tl_lower[0], self.tl_corner[0], self.bl_corner[0], bl_upper[0]]
        y = [tl_lower[1], self.tl_corner[1], self.bl_corner[1], bl_upper[1]]
        z = [tl_lower[2], self.tl_corner[2], self.bl_corner[2], bl_upper[2]]
        verts = [list(zip(x, y, z))]
        poly = Poly3DCollection(verts)
        poly.set_color(color)
        ax.add_collection3d(poly)
        
        x = [self.tr_corner[0], tr_lower[0], br_upper[0], self.br_corner[0]]
        y = [self.tr_corner[1], tr_lower[1], br_upper[1], self.br_corner[1]]
        z = [self.tr_corner[2], tr_lower[2], br_upper[2], self.br_corner[2]]
        verts = [list(zip(x, y, z))]
        poly = Poly3DCollection(verts)
        poly.set_color(color)
        ax.add_collection3d(poly)
        
        x = [bl_upper[0], br_upper[0], br_outer[0], bl_outer[0]]
        y = [bl_upper[1], br_upper[1], br_outer[1], bl_outer[1]]
        z = [bl_upper[2], br_upper[2], br_outer[2], bl_outer[2]]
        verts = [list(zip(x, y, z))]
        poly = Poly3DCollection(verts)
        poly.set_color(color)
        ax.add_collection3d(poly)


# Track class to manage gates and draw the full track
class Track:
    def __init__(self, gates):
        self.gates = gates
    
    @classmethod
    def from_yaml(cls, yaml_path):
        """Load track from a YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        gates = []
        for k, v in data['gates'].items():
            if k == 'N': 
                continue
            
            # Get gate position, rotation, and scale
            position = np.array(v['position'])
            rotation = np.array(v['rotation'])
            scale = np.array(v['scale'])[0] if 'scale' in v else 1.0
            
            # Create gate object
            gates.append(Gate(position, rotation, size=scale))
        
        return cls(gates)
    
    def draw(self, ax, show_numbers=True):
        """Draw the track (all gates) on the given 3D axes."""
        # Draw all gates with unique color and alpha settings
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.gates)))
        
        for i, (gate, color) in enumerate(zip(self.gates, colors)):
            gate.draw(ax, color=color)
            
            # Add a small marker at the gate center
            ax.scatter(gate.pos[0], gate.pos[1], gate.pos[2], color=color, s=30, alpha=1.0)
            
            # Add gate number if requested
            if show_numbers:
                ax.text(gate.pos[0], gate.pos[1], gate.pos[2] + gate.gate_dim/2 + 0.3, 
                       f'Gate {i+1}', 
                       fontsize=10, color='black', ha='center', va='center')
        
        # Draw connecting lines between gates to show the racing path
        gate_centers = np.array([gate.pos for gate in self.gates])
        
        # Connect the gates in order, with the last gate connecting back to the first
        ax.plot(gate_centers[:, 0], gate_centers[:, 1], 
                gate_centers[:, 2], 'k--', alpha=0.5, linewidth=1)


# Quaternion utilities for proper gate orientation
def quaternion_to_rotation_matrix(quat):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = quat
    
    # Precompute common factors
    xx = x * x
    xy = x * y
    xz = x * z
    xw = x * w
    yy = y * y
    yz = y * z
    yw = y * w
    zz = z * z
    zw = z * w
    
    # Build rotation matrix
    rot_matrix = np.zeros((3, 3))
    rot_matrix[0, 0] = 1 - 2 * (yy + zz)
    rot_matrix[0, 1] = 2 * (xy - zw)
    rot_matrix[0, 2] = 2 * (xz + yw)
    rot_matrix[1, 0] = 2 * (xy + zw)
    rot_matrix[1, 1] = 1 - 2 * (xx + zz)
    rot_matrix[1, 2] = 2 * (yz - xw)
    rot_matrix[2, 0] = 2 * (xz - yw)
    rot_matrix[2, 1] = 2 * (yz + xw)
    rot_matrix[2, 2] = 1 - 2 * (xx + yy)
    
    return rot_matrix

def quaternion_ros_to_unity(quat):
    """
    Convert a quaternion from ROS to Unity coordinate system.
    This mimics flightlib's quaternionRos2Unity operation.
    """
    # Create coordinate transform matrix (like in flightlib's math.cpp)
    rot_mat = np.zeros((3, 3))
    rot_mat[0, 0] = 1.0
    rot_mat[1, 2] = 1.0
    rot_mat[2, 1] = 1.0
    
    # Convert quaternion to rotation matrix in ROS frame
    ros_rot_mat = quaternion_to_rotation_matrix(quat)
    
    # Apply coordinate transformation
    unity_rot_mat = rot_mat @ ros_rot_mat @ rot_mat.T
    
    # Convert back to quaternion
    # Note: This is a simplification; in practice we just use the matrix directly
    return quat  # We'll use unity_rot_mat directly in our operations

def rotate_points(points, rotation_matrix, center):
    """Rotate points around center using rotation matrix."""
    translated = points - center
    rotated = np.array([rotation_matrix @ point for point in translated])
    return rotated + center

# Helper to load gate positions and rotations from SplitS.yaml
def load_gates(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    gates = []
    for k, v in data['gates'].items():
        if k == 'N': continue
        # Store both position and rotation for each gate
        gates.append({
            'position': np.array(v['position']),
            'rotation': np.array(v['rotation']),
            'scale': np.array(v['scale']) if 'scale' in v else np.array([0.604, 0.604, 0.604])
        })
    return gates


def set_axes_equal(ax):
    '''Set 3D plot axes to equal scale.'''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max([x_range, y_range, z_range])
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])


def plot_gates_3d(ax, gates, r=1.0):
    """Plot gates with correct orientations based on quaternion rotations."""
    # This function is now commented out to disable gate visualization
    return  # Early return - don't draw any gates
    
    # The rest of the function is preserved but will not be executed
    """
    # Define the conversion for coordinates like Unity does
    # Unity uses left-handed coordinate system: (x, z, y) while ROS uses (x, y, z)
    def to_unity_coords(v):
        return np.array([v[0], v[2], v[1]])
    
    # Conversion back from Unity to ROS coords for visualization
    def from_unity_coords(v):
        return np.array([v[0], v[2], v[1]])
    
    for i, gate in enumerate(gates):
        pos = gate['position']
        quat = gate['rotation']  # [w, x, y, z]
        
        # Apply coordinate system conversion like in unity_bridge.cpp
        rot_mat = np.zeros((3, 3))
        rot_mat[0, 0] = 1.0
        rot_mat[1, 2] = 1.0
        rot_mat[2, 1] = 1.0
        
        # Get rotation matrix from quaternion
        ros_rot_mat = quaternion_to_rotation_matrix(quat)
        
        # Convert to Unity-like coordinate system
        unity_rot_mat = rot_mat @ ros_rot_mat @ rot_mat.T
        
        # Gate center in original coordinates
        center = np.array(pos)
        half = r
        
        # Different handling for each gate type based on Unity's representation
        # Gate pattern identification based on quaternion values
        gate_type = -1
        
        # Identify gate types based on quaternion values and gate index
        if abs(quat[0] - 0.7071068) < 1e-6 and abs(quat[3] - 0.7071068) < 1e-6:
            # Gates 1, 5 pattern: [0.7071068, 0, 0, 0.7071068]
            gate_type = 1
        elif abs(quat[0] - 0.7071068) < 1e-6 and abs(quat[3] + 0.7071068) < 1e-6:
            # Gate 4 pattern: [0.7071068, 0, 0, -0.7071068]
            gate_type = 2
        elif i == 2:  # Gate 3 (index 2) has a special rotation handling
            gate_type = 3
        elif i == 5:  # Gate 6 (index 5) has a special rotation handling
            gate_type = 4
        elif i == 6:  # Gate 7 (index 6) has a special rotation handling
            gate_type = 5
        
        # Create a standard gate in the XY plane (at the origin)
        std_corners = np.array([
            [-half, -half, 0],
            [ half, -half, 0],
            [ half,  half, 0],
            [-half,  half, 0],
            [-half, -half, 0]
        ])
        
        # Apply specific transformations based on gate type
        if gate_type == 1:  # Gates 1, 5
            # Rotate 90° around X axis to match Unity's representation
            pre_rot = np.array([
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0]
            ])
            rotated_corners = rotate_points(std_corners, pre_rot @ unity_rot_mat, np.zeros(3)) + center
            
        elif gate_type == 2:  # Gate 4
            # Rotate 90° around X axis + 180° around Z for Gate 4
            pre_rot = np.array([
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0]
            ])
            extra_rot = np.array([
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
            ])
            rotated_corners = rotate_points(std_corners, pre_rot @ extra_rot @ unity_rot_mat, np.zeros(3)) + center
            
        elif gate_type == 3:  # Gate 3 (tilted)
            # Special transformation for Gate 3
            pre_rot = np.array([
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0]
            ])
            # Compensate for the specific quaternion of Gate 3 to correct tilt
            angle = np.pi/6  # Adjust this angle if needed
            extra_rot = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            rotated_corners = rotate_points(std_corners, pre_rot @ extra_rot @ unity_rot_mat, np.zeros(3)) + center
            
        elif gate_type == 4:  # Gate 6 - special case with 90-degree rotation
            # For Gate 6, we need a specific orientation (90° left or right)
            # First define a standard gate in XY plane
            rot90z = np.array([
                [0, -1, 0],  # 90° around Z-axis
                [1, 0, 0],
                [0, 0, 1]
            ])
            
            # This rotation makes it stand "upright" along the Y-axis
            rot90x = np.array([
                [1, 0, 0],
                [0, 0, -1], # 90° around X-axis
                [0, 1, 0]
            ])
            
            # This final rotation orients it properly in the scene
            rot90y = np.array([
                [0, 0, 1], # 90° around Y-axis
                [0, 1, 0],
                [-1, 0, 0]
            ])
            
            # Combine rotations - apply in the right order to get proper orientation
            # Rotate first around X to make gate vertical, then around Y for proper facing
            total_rot = rot90y @ rot90x
            
            # Apply the combined rotation to the standard corners
            rotated_corners = rotate_points(std_corners, total_rot, np.zeros(3)) + center
            
        elif gate_type == 5:  # Gate 7
            # Special transformation for Gate 7 to fix flattening
            pre_rot = np.array([
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0]
            ])
            rotated_corners = rotate_points(std_corners, pre_rot @ unity_rot_mat, np.zeros(3)) + center
            
        else:  # Standard gates (the ones that were already working)
            # Use the Unity rotation matrix directly
            rotated_corners = rotate_points(std_corners, unity_rot_mat, np.zeros(3)) + center
        
        # Draw the gate
        segs = [[rotated_corners[j], rotated_corners[j+1]] for j in range(4)]
        lc = Line3DCollection(segs, colors="#333333", linewidths=5)
        ax.add_collection3d(lc)
        
        # Add a marker for the gate center
        ax.scatter(pos[0], pos[1], pos[2], color='red', s=40)
        
        # Add small arrows to show gate orientation
        arrow_len = 0.2  # Smaller arrows for clarity
        
        # Get the appropriate rotation matrix for this gate
        if gate_type == 1:
            rot_matrix = pre_rot @ unity_rot_mat
        elif gate_type == 2:
            rot_matrix = pre_rot @ extra_rot @ unity_rot_mat
        elif gate_type == 3:
            rot_matrix = pre_rot @ extra_rot @ unity_rot_mat
        elif gate_type == 4:
            # Use the same rotation computed for Gate 6
            rot_matrix = total_rot
        elif gate_type == 5:
            rot_matrix = pre_rot @ unity_rot_mat
        else:
            rot_matrix = unity_rot_mat
        
        # Draw orientation axes
        # Forward direction (X-axis in local frame)
        forward = center + arrow_len * rot_matrix @ np.array([1, 0, 0])
        ax.quiver(center[0], center[1], center[2], 
                 forward[0]-center[0], forward[1]-center[1], forward[2]-center[2], 
                 color='red', arrow_length_ratio=0.3)
        
        # Up direction (Y-axis in local frame)
        up = center + arrow_len * rot_matrix @ np.array([0, 1, 0])
        ax.quiver(center[0], center[1], center[2], 
                 up[0]-center[0], up[1]-center[1], up[2]-center[2], 
                 color='green', arrow_length_ratio=0.3)
        
        # Right direction (Z-axis in local frame)
        right = center + arrow_len * rot_matrix @ np.array([0, 0, 1])
        ax.quiver(center[0], center[1], center[2], 
                 right[0]-center[0], right[1]-center[1], right[2]-center[2], 
                 color='blue', arrow_length_ratio=0.3)
        
        # Add gate number text
        ax.text(pos[0], pos[1], pos[2]+half+0.2, f'Gate {i+1}', 
               fontsize=10, color='black', ha='center', va='center')
    """


def draw_translucent_gate_cuboids(ax, yaml_path, alpha=0.25, edge_color='k'):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    for gate in data['gates'].values():
        if isinstance(gate, dict) and 'position' in gate and 'rotation' in gate and 'scale' in gate:
            pos = np.array(gate['position'])
            scale = np.array(gate['scale'])
            # 8 corners of the cuboid in local frame
            # Increase the length (z-dimension) of the gate cuboid for better visibility
            width_factor = 4.5
            length_factor = 4.5
              # Increase this value for even longer gates
            w, h, d = scale[0]* width_factor, scale[1]* 0.5, scale[2]*length_factor 
            corners = np.array([
                [-w/2, -h/2, -d/2],
                [ w/2, -h/2, -d/2],
                [ w/2,  h/2, -d/2],
                [-w/2,  h/2, -d/2],
                [-w/2, -h/2,  d/2],
                [ w/2, -h/2,  d/2],
                [ w/2,  h/2,  d/2],
                [-w/2,  h/2,  d/2],
            ])
            quat = gate['rotation']  # [w, x, y, z]
            rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
            rotated_corners = rot.apply(corners) + pos
            # 12 edges of the cuboid
            edges = [
                [0,1],[1,2],[2,3],[3,0], # bottom
                [4,5],[5,6],[6,7],[7,4], # top
                [0,4],[1,5],[2,6],[3,7]  # sides
            ]
            for e in edges:
                ax.plot(*zip(rotated_corners[e[0]], rotated_corners[e[1]]), color=edge_color, alpha=0.6, linewidth=3)
            # Optionally, add a very transparent face for visual effect
            faces = [
                [0,1,2,3], [4,5,6,7], [0,1,5,4], [2,3,7,6], [1,2,6,5], [0,3,7,4]
            ]
            for f in faces:
                verts = [rotated_corners[f]]
                poly = Poly3DCollection(verts, alpha=alpha, facecolor='white', edgecolor='none')
                ax.add_collection3d(poly)


def test_model(env, model, render=False):
    # Create a figure for trajectory visualization
    fig = plt.figure(figsize=(24, 18))
    
    # Main 3D plot
    ax_3d = fig.add_subplot(111, projection='3d')
    ax_3d.set_title('', fontsize=24, weight='bold')
    ax_3d.set_xlabel('X', fontsize=10)
    ax_3d.set_ylabel('Y', fontsize=10)
    ax_3d.set_zlabel('Z', fontsize=10)
    ax_3d.grid(True)
    # Draw translucent cuboid gates from CircularLoop.yaml
    draw_translucent_gate_cuboids(ax_3d, '/home/golgapha/Desktop/flightmare/flightlib/configs/CircularLoop.yaml')
    max_ep_length = env.max_episode_steps
    num_rollouts = 10
    
    if render:
        env.connectUnity()
    
    best_num_gates = 0
    best_rollout_data = None
    
    for n_roll in range(num_rollouts):
        pos, vel = [], []
        gates_passed = []
        current_gate = 0
        actions = []
        value_profile = []  # Store value function at each step
        obs, done, ep_len = env.reset(), False, 0
        
        while not (np.all(done) or (ep_len >= max_ep_length)):
            act, _ = model.predict(obs, deterministic=True)
            # Get value function for current obs (try model.value or model.critic or model.predict_value)
            try:
                value = model.value(obs)
            except AttributeError:
                try:
                    value = model.critic(obs)
                except AttributeError:
                    value = None
            if value is not None:
                # Robustly extract scalar
                if isinstance(value, np.ndarray):
                    value_profile.append(float(np.ravel(value)[0]))
                else:
                    value_profile.append(float(value))
            else:
                value_profile.append(np.nan)
            prev_obs = obs.copy()
            obs, rew, done, infos = env.step(act)
            
            # Extract gate information from observation
            # Assume obs[0, 12:15] contains relative gate position
            ep_len += 1
            pos.append(obs[0, 0:3].tolist())
            
            # Calculate velocity from position changes
            if len(pos) > 1:
                curr_pos = np.array(pos[-1])
                prev_pos = np.array(pos[-2])
                velocity = (curr_pos - prev_pos) / env.dt  # Assuming env.dt is available
                vel.append(np.linalg.norm(velocity))
            else:
                vel.append(0.0)
            
            # Detect if we passed a gate by analyzing changes in the relative gate position
            # When there's a sudden jump in relative position, we likely passed a gate
            if ep_len > 1:
                prev_gate_rel_pos = prev_obs[0, 12:15]
                curr_gate_rel_pos = obs[0, 12:15]
                if np.linalg.norm(curr_gate_rel_pos - prev_gate_rel_pos) > 5.0:  # Threshold for detecting gate change
                    current_gate += 1
                    gates_passed.append(ep_len)
        pos = np.asarray(pos)
        num_gates_passed = current_gate
        
        # Track best rollout for more detailed plotting
        if num_gates_passed > best_num_gates:
            best_num_gates = num_gates_passed
            best_rollout_data = {
                'pos': pos,
                'vel': vel,
                'gates_passed': gates_passed,
                'ep_len': ep_len,
                'value_profile': value_profile
            }
        
        # Create colormap that changes from blue to red as drone progresses
        num_points = pos.shape[0]
        colors = plt.cm.plasma(np.linspace(0, 1, num_points))
        
        # Plot trajectory with color gradient to show time progression using a smooth curve
        from scipy import interpolate
        
        # If we have enough points for spline interpolation
        if num_points > 3:
            # Create parameter for spline
            t = np.linspace(0, 1, num_points)
            
            # Create the spline representation for smooth trajectory
            tck, u = interpolate.splprep([pos[:, 0], pos[:, 1], pos[:, 2]], s=0.1, k=min(3, num_points-1))
            
            # Evaluate the spline at more points for a smoother curve
            u_fine = np.linspace(0, 1, num_points * 5)
            x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
            
            # Get colors for the interpolated trajectory
            interp_colors = plt.cm.plasma(np.linspace(0, 1, len(u_fine)))
            
            # Plot the smooth trajectory
            for i in range(len(u_fine) - 1):
                ax_3d.plot([x_fine[i], x_fine[i+1]], 
                          [y_fine[i], y_fine[i+1]], 
                          [z_fine[i], z_fine[i+1]], 
                          linewidth=2.5, alpha=0.8, color=interp_colors[i])
        else:
            # Fallback to original segmented plotting if not enough points
            for i in range(num_points - 1):
                ax_3d.plot(pos[i:i+2, 0], pos[i:i+2, 1], pos[i:i+2, 2], 
                          linewidth=1.5, alpha=0.8, color=colors[i])
        
        # Plot trajectory with color gradient to show velocity (m/s)
        if num_points > 3:
            t = np.linspace(0, 1, num_points)
            tck, u = interpolate.splprep([pos[:, 0], pos[:, 1], pos[:, 2]], s=0.1, k=min(3, num_points-1))
            u_fine = np.linspace(0, 1, num_points * 5)
            x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
            # Interpolate velocity for color mapping
            vel_arr = np.array(vel)
            from scipy.interpolate import interp1d
            vel_interp = interp1d(np.linspace(0, 1, len(vel_arr)), vel_arr, kind='linear', fill_value='extrapolate')
            vel_fine = vel_interp(u_fine)
            # Normalize velocity for colormap
            vmin, vmax = np.min(vel_fine), np.max(vel_fine)
            norm = plt.Normalize(vmin, vmax)
            cmap = plt.cm.jet  # blue-green-yellow-red
            for i in range(len(u_fine) - 1):
                ax_3d.plot([x_fine[i], x_fine[i+1]],
                          [y_fine[i], y_fine[i+1]],
                          [z_fine[i], z_fine[i+1]],
                          linewidth=1.5, alpha=0.8, color=cmap(norm(vel_fine[i])))
        else:
            vel_arr = np.array(vel)
            vmin, vmax = np.min(vel_arr), np.max(vel_arr)
            norm = plt.Normalize(vmin, vmax)
            cmap = plt.cm.jet
            for i in range(num_points - 1):
                ax_3d.plot(pos[i:i+2, 0], pos[i:i+2, 1], pos[i:i+2, 2],
                          linewidth=2.5, alpha=0.8, color=cmap(norm(vel_arr[i])))
        
        # Add a label for the legend
        # ax_3d.plot([], [], [], label=f'Rollout {n_roll+1}: Passed {num_gates_passed} gates', 
        #           linewidth=3, color=plt.cm.tab10(n_roll % 10))
        
        # Start and end points removed as requested
    
    # Mark gate passing events on trajectory if available
    # if best_rollout_data and best_rollout_data['gates_passed']:
    #     best_pos = best_rollout_data['pos']
    #     for i, gate_step in enumerate(best_rollout_data['gates_passed']):
    #         if gate_step < len(best_pos):
    #             pos_at_gate = best_pos[gate_step]
    #             # Create a marker at each gate passing point
    #             ax_3d.scatter(pos_at_gate[0], pos_at_gate[1], pos_at_gate[2],
    #                         color='lime', s=150, alpha=0.8, marker='*')
    
    if render:
        env.disconnectUnity()
        
    set_axes_equal(ax_3d)
    ax_3d.legend(fontsize=12, loc='upper right')
    
    # # Add color bar to show time progression
    # sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma)
    # sm.set_array([])
    # cbar = plt.colorbar(sm, ax=ax_3d, pad=0.05)
    # cbar.set_label('Time progression', rotation=270, labelpad=20, fontsize=14)
    
    # Add color bar to show velocity (m/s)
    sm_vel = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    sm_vel.set_array([])
    cbar_vel = plt.colorbar(sm_vel, ax=ax_3d, pad=0.12)
    cbar_vel.set_label('Velocity (m/s)', rotation=270, labelpad=20, fontsize=14)
    
    # Add summary text
    # if best_rollout_data:
    #     fig.text(0.02, 0.01, 
    #             f"Best Rollout: Passed {best_num_gates} gates",
    #             fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/home/golgapha/Desktop/flightmare/drone_racing_results.png', dpi=480)
    plt.show()
    
    # --- Velocity and Value Function Profiles for Best Rollout ---
    if best_rollout_data is not None:
        pos = best_rollout_data['pos']
        vel = np.array(best_rollout_data['vel'])
        values = np.array(best_rollout_data['value_profile'])
        # Plot velocity and value function profiles
        fig2, axarr = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        axarr[0].plot(vel, label='Velocity (m/s)', color='tab:blue')
        axarr[0].set_ylabel('Velocity (m/s)')
        axarr[0].set_title('Velocity Profile')
        axarr[0].grid(True)
        axarr[1].plot(values, label='Value Function', color='tab:orange')
        axarr[1].set_ylabel('Value')
        axarr[1].set_title('Value Function Profile')
        axarr[1].set_xlabel('Timestep')
        axarr[1].grid(True)
        plt.tight_layout()
        plt.savefig('/home/golgapha/Desktop/flightmare/drone_racing_value_velocity.png', dpi=200)
        plt.show()