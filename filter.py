import pickle
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
import matplotlib.pyplot as plt
from copy import deepcopy


with open('blender_data.pickle', 'rb') as fd:
    blender_data = pickle.load(fd)

with open('tracking_data.pickle','rb') as fd:
    tracking_data = pickle.load(fd)

camera_data = blender_data['camera_data']
A = camera_data['A']
RT = camera_data['RT']
fps = camera_data['fps']
positions = blender_data['location_data']
real_ped_position = blender_data['pedestrian_pos']
pedestrian_height = blender_data['pedestrian_height']
dt = 1/fps

center_points = tracking_data['center_points']
heights = tracking_data['heights']

assert(len(center_points) == len(heights) and len(positions) == len(center_points))
velocities = [[0,0,0]]
for i in range(1, len(positions)):
    x, y, z = positions[i]
    x_prev, y_prev, z_prev = positions[i-1]
    vx = (x - x_prev) / dt
    vy = (y - y_prev) / dt
    vz = (z - z_prev) / dt
    velocities.append([vx, vy, vz])

def f_cv(x,dt):
    # position and velocity relative to the camera
    px,vx,py,vy,pz,vz,h = x
    px = px + dt * vx
    py = py + dt * vy
    pz = pz + dt * vz  
    h = pedestrian_height
    return np.array([px,vx,py,vy,pz,vz,h])
 
def calcHeightOfPedestrian(pCenter_World, pedestrian_height, A, RT):
    assert(len(pCenter_World) == 4 and pCenter_World[3] == 1)  
    assert(RT.shape == (3, 4))  
    assert(A.shape == (3, 3))  

    pCenter_Camera =  pCenter_World  
    
    top_p = np.array([pCenter_Camera[0], pCenter_Camera[1], pCenter_Camera[2] + pedestrian_height/2,1])  
    bottom_p = np.array([pCenter_Camera[0], pCenter_Camera[1], pCenter_Camera[2] - pedestrian_height/2,1])   
    
    top_p_2D = A @ (RT@top_p)
    bottom_p_2D = A @ (RT@bottom_p)

    top_p_2D /= top_p_2D[2]
    bottom_p_2D /= bottom_p_2D[2]
    height_2D = np.sqrt(np.sum((top_p_2D[:2] - bottom_p_2D[:2]) ** 2)) 
    return height_2D
    
hs = []
def h_cv(x):    
    px,vxp,py,vyp,pz,vzp,h = x
    # world is in reality Camera world coordinates
    p_world =  np.array([px,py,pz,1])
    p0_Camera = RT @ p_world
    p0_2D = A @ p0_Camera
    p0_2D = p0_2D / p0_2D[2]
    cx = p0_2D[0]
    cy = p0_2D[1]

    ph = calcHeightOfPedestrian(p_world,h,A,RT)
    
    return np.array([cx,cy,ph,vxp,vyp,vzp])

sigmas = MerweScaledSigmaPoints(7,alpha=.1, beta=2, kappa=-4)
ukf = UKF(dim_x=7,dim_z=6,fx=f_cv, hx=h_cv, dt=dt, points=sigmas)
ukf.R = np.diag([2**2,2**2,6**2,0.1**2,0.1**2,0.1**2])

std_pos = 1
std_vel = 0.3
std_h = 0.001

Q = np.zeros((7,7))
Q[0,0] = std_pos**20
Q[1,1] = std_vel**2
Q[2,2] = std_pos**2
Q[3,3] = std_vel**2
Q[4,4] = std_pos**2
Q[5,5] = std_vel**2
Q[6,6] = std_h**2

P = np.zeros((7,7))
P[0,0] = 10**2
P[1,1] = 0.5**2
P[2,2] = 100**2
P[3,3] = 0.5**2
P[4,4] = 5**2
P[5,5] = 0.5**2
P[6,6] = 0.25**2

N = len(velocities)
ukf.Q = Q
ukf.x = [0,0,10,0,0,0,0]
ukf.P = P
xhat = [] 
Phat = []
collision = np.full(N,False)

def calculate_world_position(filter_values, camera_pos):
    relative_pos = np.array([filter_values[0],filter_values[2],filter_values[4]])
    wp = camera_pos + relative_pos
    return wp

for n in range(N):
    camera_pos = np.array(positions[n])
    # height of camera set to 0 to better calculate world pos of pedestrian
    camera_pos[2] = 0
    vx,vy,vz = velocities[n]
    
    ukf.predict()
    ukf_prediction = deepcopy(ukf)
    x = ukf.x.copy()
    print(f"vel: {velocities[n]}")
    print(f"{x[1]:.2f},{x[3]:.2f},{x[5]:.2f}")
    P = ukf.P.copy()
    for r in range(N-n):
        ukf_prediction.predict()
        x_pred = ukf_prediction.x.copy()
        P_pred = ukf_prediction.P.copy()
        std_for_x = np.sqrt(P[0,0])
        std_for_y = np.sqrt(P[2,2])
        relative_pos = np.array([x_pred[0],x_pred[2],x_pred[4]])

        if relative_pos[0] < std_for_x and relative_pos[1] < std_for_y:
            wp = calculate_world_position(x_pred, camera_pos)
            #print(f"Collision at n: {n}, r: {r} world position: {wp} x: {std_for_x:.2f} y: {std_for_y:.2f}")
            collision[n] = True
            break

    # calculate the world position of the pedestrian 
    wp = calculate_world_position(x, camera_pos)
    xhat.append(np.array([wp[0],x[1],wp[1],x[3],wp[2],x[5],x[6]]))
    Phat.append(P)

    # update filter 
    if center_points[n] == None:
        #print(n)
        continue
    else:
        x, y = center_points[n]
    h = heights[n]
    # -v because pedestrian is moving towards the camera
    m = [x,y,h,-vx,-vy,-vz]
    
    ukf.update(z=m)

xhat_vals = [v.tolist() for v in xhat]
phat_vals = [v.tolist() for v in Phat]

Phat = np.array(Phat)
Px = np.sqrt(Phat[:, 0, 0]).tolist()
Py = np.sqrt(Phat[:, 2, 2]).tolist()
collision = collision.tolist()
result = {'x': xhat_vals, 'collision_pred': collision, 'Px':Px, 'Py': Py}    

with open('Filter_Output.pickle', 'wb') as fd:
    pickle.dump(result, fd)
    print("pickled")

xhat = np.array(xhat)
phat = np.array(phat_vals)
real_ped_position = np.array(real_ped_position)

showplots = 1
end_point = 100
if showplots == 1:
    fig, axes = plt.subplots(5, 1, figsize=(8, 8), sharex=True)  

    axes_labels = ['x-axis', 'y-axis', 'z-axis', 'height']

    for axis in range(4): 
        if axis < 3:
            real_val = real_ped_position[:end_point,axis]
            # print(real_val)
        else:
            real_val = pedestrian_height
        residuals = real_val - xhat[:end_point, 2 * axis]
        phat_axis = np.sqrt(phat[:end_point, 2 * axis, 2 * axis])
        
        axes[axis].plot(residuals, label='Residuals', color='blue')
        axes[axis].plot(phat_axis, label='Standard Deviation', color='orange')
        axes[axis].plot(-phat_axis, color='orange', linestyle='--')
        
        axes[axis].set_title(f'Residuals and Uncertainty ({axes_labels[axis]})')
        axes[axis].set_ylabel('Difference')
        axes[axis].grid(True)
        axes[axis].legend()

    axes[4].plot(collision[:end_point], label='Collision', color='red')
    axes[4].set_title('Collision Prediction')
    
    plt.tight_layout()
    plt.show()

test = np.array([7,0,90,0,0,0,3.8])

print(h_cv(test))
