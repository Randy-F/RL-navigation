import airsim
import time
import copy
import numpy as np
from PIL import Image
import cv2
import bisect

goal_threshold = 3
np.set_printoptions(precision=3, suppress=True)
IMAGE_VIEW = True

class drone_env:
	def __init__(self,start = [0,0,-5],aim = [32,38,-4]):
		self.start = np.array(start)
		self.aim = np.array(aim)
		self.client = airsim.MultirotorClient()
		self.client.confirmConnection()
		self.client.enableApiControl(True)
		self.client.armDisarm(True)
		self.threshold = goal_threshold
		
	def reset(self):
		self.client.reset()
		self.client.enableApiControl(True)
		self.client.armDisarm(True)
		self.client.moveToPositionAsync(self.start.tolist()[0],self.start.tolist()[1],self.start.tolist()[2],5).join()
		time.sleep(0.5)
		
		
	def isDone(self):
		pos = self.client.getPosition()
		if distance(self.aim,pos) < self.threshold:
			return True
		return False
		
	def moveByDist(self,diff, forward = False):
		temp = airsim.YawMode()
		temp.is_rate = forward
		self.client.moveByVelocityAsync(diff[0], diff[1], diff[2], 1 ,drivetrain = airsim.DrivetrainType.ForwardOnly, yaw_mode = temp).join()
		time.sleep(0.1)
		
		return 0
		
	def render(self,extra1 = "",extra2 = ""):
		pos = v2t(self.client.getPosition())
		goal = distance(self.aim,pos)
		print (extra1,"distance:",int(goal),"position:",pos.astype("int"),extra2)
		
	def help(self):
		print ("drone simulation environment")
		
		
#-------------------------------------------------------
# grid world
		
class drone_env_gridworld(drone_env):
	def __init__(self,start = [0,0,-4],aim = [32,38,-4],scaling_factor = 5):
		drone_env.__init__(self,start,aim)
		self.scaling_factor = scaling_factor
		
	def interpret_action(self,action):
		scaling_factor = self.scaling_factor
		if action == 0:
			quad_offset = (0, 0, 0)
		elif action == 1:
			quad_offset = (scaling_factor, 0, 0)
		elif action == 2:
			quad_offset = (0, scaling_factor, 0)
		elif action == 3:
			quad_offset = (0, 0, scaling_factor)
		elif action == 4:
			quad_offset = (-scaling_factor, 0, 0)	
		elif action == 5:
			quad_offset = (0, -scaling_factor, 0)
		elif action == 6:
			quad_offset = (0, 0, -scaling_factor)
		
		return np.array(quad_offset).astype("float64")
	
	def step(self,action, step_count):
		diff = self.interpret_action(action)
		drone_env.moveByDist(self,diff)
		
		pos_ = v2t(self.client.getPosition())
		vel_ = v2t(self.client.getVelocity())
		state_ = np.append(pos_, vel_)
		pos = self.state[0:3]
		
		info = None
		done = False
		reward = self.rewardf(self.state,state_)
		reawrd = reward / 50
		if action == 0:
			reward -= 10
		if self.isDone():
			done = True
			reward = 100
			info = "success"
		if self.client.getCollisionInfo().has_collided:
			reward = -100
			done = True
			info = "collision"
		if (distance(pos_,self.aim)>150):
			reward = -100
			done = True
			info = "out of range"
			
		self.state = state_
		
		return state_,reward,done,info
	
	def reset(self):
		drone_env.reset(self)
		pos = v2t(self.client.getPosition())
		vel = v2t(self.client.getVelocity())
		state = np.append(pos, vel)
		self.state = state
		return state
		
	def rewardf(self,state,state_):
		
		dis = distance(state[0:3],self.aim)
		dis_ = distance(state_[0:3],self.aim)
		reward = dis - dis_
		reward = reward * 1
		reward -= 1
		return reward
		
#-------------------------------------------------------
# height control
# continuous control
		
class drone_env_heightcontrol(drone_env):
	def __init__(self,start = [0,0,-4],aim = [65, 0, -4],scaling_factor = 5,img_size = [64,64], max_eps_step = 25):
		drone_env.__init__(self,start,aim)
		self.scaling_factor = scaling_factor
		self.aim = np.array(aim)
		self.height_limit = -10
		self.rand = False
		self.aim_height = self.aim[2]
		self.step_count = 0
		self.max_eps_step = max_eps_step
		# self.aim = np.array([65, 0, -5])
		# if aim == None:
		# 	self.rand = True
		# 	self.start = np.array([0,0,-10])
		# else:
		# 	self.aim_height = self.aim[2]
	
	def reset_aim(self):
		# self.aim = (np.random.rand(3)*300).astype("int")-150
		# self.aim[2] = -np.random.randint(10) - 5
		self.aim = np.array([65, 0, -5])
		print ("Our aim is: {}".format(self.aim).ljust(80," "),end = '\r')
		self.aim_height = self.aim[2]
		
	def reset(self):
		if self.rand:
			self.reset_aim()
		drone_env.reset(self)
		self.state, _ = self.getState()
		#  穿越量复位
		self.prePassIndex = 0
		self.step_count = 0
		return self.state
		
	def getState(self):
		_state = self.client.simGetGroundTruthKinematics()
		pos = _state.position
		vel = _state.linear_velocity
		img = self.getImg()
		state = {"pos": pos, "vel": vel, "img": img}
		# training_state = [img, np.array([pos.z_val])]
		training_state = np.transpose(img, [2,0,1])
		
		#一个用于训练，一个用于计算reward
		return training_state, state

	def move(self, action):
		_state = self.client.simGetGroundTruthKinematics()
		pos = _state.position
		
		if abs(action) > 1:
			print ("action value error")
			action = action / abs(action)
		
		x = pos.x_val + 4
		y = pos.y_val + 0
		z = pos.z_val - action * self.scaling_factor
		
		self.client.moveByVelocityZAsync(2.5,0,z,5)
		
		# self.client.moveToPositionAsync(x,y,z,4)
		time.sleep(0.2)

	def isCollided(self, pos, obsArray):
		info = self.client.simGetCollisionInfo()
		return info.has_collided
		# for obs in obsArray:
		# 	if(pos.x_val>obs[0] and pos.x_val<obs[1] \
		# 		and pos.y_val>obs[2] and pos.y_val<obs[3] \
		# 		and pos.z_val>obs[4] and pos.z_val<obs[5]):
		# 		return True
		
		# return False


	def isPassed(self, pos, pass_x = 90):
		return pos.x_val > pass_x

	def cmpReward(self, pos):
		passArray = [27, 44, 70, 90]
		passIndex = bisect.bisect_right(passArray, pos.x_val)
		reward = 0

		if passIndex > self.prePassIndex:
			reward = 50

		self.prePassIndex = passIndex

		return reward

	def getRewardDone(self, state):
		pos = state["pos"]
		info = None
		done = False
		reward = 0
		obsArray = [[23.5, 26.0, -10.0, 10.5, -4.9, 19.0], 
					[40.5, 43.0, -10.0, 10.5, -36.0, -11.0]]

		height_limit = [-32, 16]

		if self.isPassed(pos, 90):
			reward = 200
			done = True
			info = "pass"
		elif self.isCollided(pos, obsArray):
			reward = -200
			done = True
			info = "Collision"
		elif pos.z_val<height_limit[0] or pos.z_val>height_limit[1]:
			reward = -200
			done = True
			info = "over height"
		elif self.step_count > self.max_eps_step:
			reward = 0
			done = True
			info = "over steps"
		else:
			reward = self.cmpReward(pos)
			done = False

		return reward, done, info



	def step(self,action): 
		self.move(action)

		training_state, state = self.getState()
		self.step_count += 1

		reward, done, info = self.getRewardDone(state)

		# self.state = state_
		reward /= 50
		norm_state = copy.deepcopy(training_state)
		# norm_state[1] = norm_state[1]/100
		
		if info == "Collision":
			norm_state = np.zeros((1,64,64))
            # norm_state[0] = np.zeros((64,64,1))
		if IMAGE_VIEW:
			cv2.imshow("view",norm_state[0])
			key = cv2.waitKey(1) & 0xFF;


		return norm_state,reward,done,info
		
	def isDone(self):
		_state = self.client.simGetGroundTruthKinematics()
		pos = _state.position
		pos = [pos.x_val, pos.y_val, pos.z_val]
		pos[2] = self.aim[2]
		if distance(self.aim,pos) < self.threshold:
			return True
		return False
		
	def rewardf(self,state,state_):
		pos = state[1][0]
		pos_ = state_[1][0]
		reward = - abs(pos_)*100 + 5
		
		return reward
		
	def getImg(self):
		try:
			responses = self.client.simGetImages([airsim.ImageRequest(1, airsim.ImageType.DepthPerspective, True, False)])
			img1d = np.array(responses[0].image_data_float, dtype=np.float)
			img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
			image = Image.fromarray(img2d)
			im_final = np.array(image.resize((64, 64)).convert('L'), dtype=np.float)/255
			im_final.resize((64,64,1))
			self.pre_im_final = im_final
		except:
			im_final = self.pre_im_final

		if IMAGE_VIEW:
			cv2.imshow("view",im_final)
			key = cv2.waitKey(1) & 0xFF;
		return im_final
		
def v2t(vect):
	if isinstance(vect,airsim.Vector3r):
		res = np.array([vect.x_val, vect.y_val, vect.z_val])
	else:
		res = np.array(vect)
	return res

def distance(pos1,pos2):
	pos1 = v2t(pos1)
	pos2 = v2t(pos2)
	#dist = np.sqrt(abs(pos1[0]-pos2[0])**2 + abs(pos1[1]-pos2[1])**2 + abs(pos1[2]-pos2[2]) **2)
	dist = np.linalg.norm(pos1-pos2)
		
	return dist