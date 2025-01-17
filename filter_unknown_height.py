import pickle
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import matplotlib.pyplot as plt
from copy import deepcopy

### In Camera Coordinate System ###
# x = x y = -z z = y
np.set_printoptions(precision=2, suppress=True)

with open('blender_data.pickle', 'rb') as fd:
    blender_data = pickle.load(fd)

with open('tracking_data.pickle','rb') as fd:
    tracking_data = pickle.load(fd)

camera_data = blender_data['camera_data']
A = camera_data['A']
RT = camera_data['RT']
R = RT[:, :3] # Camera Rotation 
fps = camera_data['fps']
cam_positions = blender_data['car_pos']
real_ped_position = blender_data['pedestrian_pos']
pedestrian_height = blender_data['pedestrian_height'] 
dt = 1/fps

center_points = tracking_data['center_points']
heights = tracking_data['heights']

assert(len(center_points) == len(heights) and len(cam_positions) == len(center_points))
velocities = [[0,0,0]]
std_c = 0.8 # ca. 3km/h
for i in range(1, len(cam_positions)):
    x, y, z = cam_positions[i]
    x_prev, y_prev, z_prev = cam_positions[i-1]
    vx = (x - x_prev) / dt + np.random.normal(0,std_c)
    vy = (y - y_prev) / dt + np.random.normal(0,std_c)
    vz = (z - z_prev) / dt + np.random.normal(0,std_c)
    vel = np.array([vx, vy, vz])
    velocities.append(vel)

real_ped_vel = [[0,0,0]]
for i in range(1, len(real_ped_position)):
    x, y, z = real_ped_position[i]
    x_prev, y_prev, z_prev = real_ped_position[i-1]
    vx = (x - x_prev) / dt
    vy = (y - y_prev) / dt
    vz = (z - z_prev) / dt
    vel = np.array([vx,vy,vz])
    real_ped_vel.append(vel)


def f_cv(x,dt,control_input):
    # position and velocity relative to the camera
    px,vx,py,vy,pz,vz,h,cx,cy,cz = x
    cx = control_input[0]
    cy = control_input[1]
    cz = control_input[2]
    # distance car moved in the time dt
    car_x = cx * dt 
    car_y = cy * dt 
    car_z = cz * dt
    px = px + dt * vx - car_x
    py = py + dt * vy - car_y 
    pz = pz + dt * vz - car_z

    return np.array([px,vx,py,vy,pz,vz,h,cx,cy,cz])
 
def calcHeightOfPedestrian(p_camera, pedestrian_height, A, R):
    # Corrdintes flipped by R
    assert(R.shape == (3, 3))  
    assert(A.shape == (3, 3))  

    top_p = np.array([p_camera[0], p_camera[1] + pedestrian_height/2 , p_camera[2] ])  
    bottom_p = np.array([p_camera[0], p_camera[1] - pedestrian_height/2 ,p_camera[2] ])   
    #print(f"top: {top_p} bot: {bottom_p}")
    top_p_2D = A @ top_p
    bottom_p_2D = A @ bottom_p

    top_p_2D /= top_p_2D[2]
    bottom_p_2D /= bottom_p_2D[2]
    #print(top_p_2D, bottom_p_2D)
    height_2D = np.sqrt(np.sum((top_p_2D[:2] - bottom_p_2D[:2]) ** 2)) 
    #print(f"h: {height_2D}")
    return height_2D
    
def h_cv(x):    
    px,vxp,py,vyp,pz,vzp,h, cx,cy,cz = x
    # world is in reality Camera world coordinates
    p_camera = R @ np.array([px,py,pz]) 
    p0_2D = A @ p_camera

    p0_2D = p0_2D / p0_2D[2]
    cx = p0_2D[0]
    cy = p0_2D[1]
    #print(f"cx: {cx}, cy: {cy}")
    ph = calcHeightOfPedestrian(p_camera,h,A,R)
    return np.array([cx,cy,ph])


# test = np.array([7,0,90,0,0,0,1.8])
# print(f"test:{h_cv(test)}")

sigmas = MerweScaledSigmaPoints(10,alpha=.1, beta=2, kappa=-4)
ukf = UKF(dim_x=10,dim_z=3,fx=f_cv, hx=h_cv, dt=dt, points=sigmas)
ukf.R = np.diag([2**2,2**2,8**2])

std_pos = 0.5 
std_vel = 0.5
std_h = 0.2
std_c = 0.8

Q = np.zeros((10,10))
Q[0,0] = std_pos**2
Q[1,1] = std_vel**2
Q[2,2] = std_pos**2
Q[3,3] = std_vel**2
Q[4,4] = std_pos**2
Q[5,5] = std_vel**2
Q[6,6] = std_h**2
Q[7,7] = std_c**2
Q[8,8] = std_c**2
Q[9,9] = std_c**2

P = np.zeros((10,10))
P[0,0] = 10**2
P[1,1] = 0.5**2
P[2,2] = 50**2
P[3,3] = 0.5**2
P[4,4] = 5**2
P[5,5] = 0.5**2
P[6,6] = 0.25**2
P[7,7] = 1**2
P[8,8] = 1**2
P[9,9] = 1**2

N = len(velocities)
ukf.Q = Q
ukf.P = P
ukf.x = [0,0,10,0,0,0,1.7,0,0,0]
filter_results = [] 
Phat = []
collision = np.full(N,False)

def calculate_world_position(filter_values, camera_pos):
    relative_pos = np.array([filter_values[0],filter_values[2],filter_values[4]])
    wp = camera_pos + relative_pos
    return wp

for n in range(N):
    # only for visualisation
    camera_pos = np.array(cam_positions[n]) 
    c = np.array(velocities[n])
    ukf.predict(control_input=c)
    ukf_prediction = deepcopy(ukf)
    x = ukf.x.copy()
    P = ukf.P.copy()
    for r in range(N-n):
        ukf_prediction.predict(control_input=c)
        x_pred = ukf_prediction.x.copy()
        P_pred = ukf_prediction.P.copy()
        std_for_x = np.sqrt(P[0,0])
        std_for_y = np.sqrt(P[2,2])
        relative_pos = np.array([x_pred[0],x_pred[2],x_pred[4]])

        if abs(relative_pos[0]) < std_for_x and abs(relative_pos[1]) < std_for_y:
            wp = calculate_world_position(x_pred, camera_pos)
            #print(f"Collision at n: {n}, r: {r} n+r: {n+r} world position: {wp} x: {std_for_x:.2f} y: {std_for_y:.2f}")
            collision[n] = True
            break
    # calculate the world position of the pedestrian 
    wp = calculate_world_position(x, camera_pos)
    # print(f"---- n: {n} ----")
    # print(f"vx: {x[1]:.2f} vy: {x[3]:.2f} vz: {x[5]:.2f}")
    # print(f"px: {x[0]:.2f} py: {x[2]:.2f} pz: {x[4]:.2f}")
    # print(f"Phat: {P}")
    filter_results.append(np.array([wp[0],x[1],wp[1],x[3],wp[2],x[5],x[6]]))
    Phat.append(P)

    if n > 100 and n < 130:
        continue

    # update filter 
    if center_points[n] == None:
        continue
    else:
        x_screen, y_screen = center_points[n]
    h = heights[n]

    m = [x_screen,y_screen,h]
    ukf.update(z=m)

# ignore first collision predictions 
collision[0:5] = False
filter_results_vals = [v.tolist() for v in filter_results]
phat_vals = [v.tolist() for v in Phat]

Phat = np.array(Phat)
Px = np.sqrt(Phat[:, 0, 0]).tolist()
Py = np.sqrt(Phat[:, 2, 2]).tolist()
collision = collision.tolist()
result = {'x': filter_results_vals, 'collision_pred': collision, 'Px': Px, 'Py': Py}    

with open('Filter_Output.pickle', 'wb') as fd:
    pickle.dump(result, fd)
    print("pickled")

# # ####### Plotting #######

def check_for_collision(real_cam_pos, real_ped_pos,threshold=1):
    for i in range(len(real_cam_pos)):
        dx = abs(real_cam_pos[i][0] - real_ped_pos[i][0])
        dy = abs(real_cam_pos[i][1] - real_ped_pos[i][1])
        if dx <= threshold  and dy <= threshold + 1.5:
            return i 
    return None    

filter_results = np.array(filter_results)
phat = np.array(phat_vals)
real_ped_position = np.array(real_ped_position)
real_ped_vel = np.array(real_ped_vel)

collision_idx = check_for_collision(cam_positions, real_ped_position)
showplots = 1
end_point = 250  # Limit number of points shown for clarity
if showplots == 1:
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True)  

    axes_labels = ['x-axis', 'y-axis', 'z-axis', 'height']
    vel_labels = ['vx', 'vy', 'vz']
    
    # Position and height plots
    for i in range(4): 
        if i < 3:
            real_val = real_ped_position[:end_point, i]
        else:
            real_val = pedestrian_height
        residuals = real_val - filter_results[:end_point, 2 * i]
        phat_axis = np.sqrt(phat[:end_point, 2 * i, 2 * i])
        
        ax = axes[i, 0]
        ax.plot(residuals, label='Residuals', color='blue')
        ax.plot(phat_axis, label='Standard Deviation', color='orange')
        ax.plot(-phat_axis, color='orange', linestyle='--')
        
        ax.set_ylim(-6, 6)
        ax.set_title(f'Residuals and Uncertainty ({axes_labels[i]})')
        ax.set_ylabel('Difference')
        ax.grid(True)

    # Velocity plots
    for i in range(3):
        real_vel = real_ped_vel[:end_point, i]
        residuals_vel = real_vel - filter_results[:end_point, 2 * i + 1]
        phat_axis_vel = np.sqrt(phat[:end_point, 2 * i + 1, 2 * i + 1])
        
        ax = axes[i, 1]
        ax.plot(residuals_vel, label='Velocity Residuals', color='blue')
        ax.plot(phat_axis_vel, label='Standard Deviation', color='orange')
        ax.plot(-phat_axis_vel, color='orange', linestyle='--')
        
        ax.set_ylim(-5, 5)
        ax.set_title(f'Velocity Residuals and Uncertainty ({vel_labels[i]})')
        ax.set_ylabel('Difference')
        ax.grid(True)
    
    # Collision prediction
    axes[3, 1].plot(collision[:end_point], label='Collision Prediction', color='blue')
    axes[3, 1].set_title('Collision Prediction')
    if collision_idx is not None:
        axes[3, 1].axvline(collision_idx, color='red', linestyle='--', label='Collision')
    axes[3, 1].legend() 
    axes[3, 1].grid(True)
    
    axes[0, 0].legend()
    
    # Hide unused subplots if any
    # axes[3, 0].axis('off')

    plt.tight_layout()
    plt.show()
