import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit, random
import bernstein_coeff_order10_arbitinterval
import time
import matplotlib.pyplot as plt 
import jax
from jax.ops import index_update, index

def get_weights_biases(weight_biases_mat_file):
    W0, b0, W1, b1, W2, b2, W3, b3 =  weight_biases_mat_file['w0'], weight_biases_mat_file['b0'], \
                                        weight_biases_mat_file['w1'], weight_biases_mat_file['b1'], \
                                    weight_biases_mat_file['w2'], weight_biases_mat_file['b2'], \
                                    weight_biases_mat_file['w3'], weight_biases_mat_file['b3']
    
    return jnp.asarray(W0), jnp.asarray(b0), jnp.asarray(W1), jnp.asarray(b1), \
            jnp.asarray(W2), jnp.asarray(b2), jnp.asarray(W3), jnp.asarray(b3)

class batch_occ_tracking():

	def __init__(self, P, Pdot, Pddot, v_max, a_max, t_fin, num, num_batch_projection,
					 num_batch_cem, tot_time, rho_ineq, maxiter_projection, rho_projection, 
					 rho_target, num_target, a_workspace, b_workspace, num_workspace, rho_workspace,
					  maxiter_cem, d_min_target, d_max_target, P_up_jax, Pdot_up_jax, Pddot_up_jax, occlusion_weight):
		
		self.rho_ineq = rho_ineq
		self.rho_projection = rho_projection
		self.rho_target = rho_target
		self.rho_workspace = rho_workspace
		self.maxiter_projection = maxiter_projection
		self.maxiter_cem = maxiter_cem

		self.t_fin = t_fin
		self.num = num
		self.t = self.t_fin/self.num
		self.num_batch_projection = num_batch_projection		# number of goals
		self.num_batch_cem = num_batch_cem

		self.v_max = v_max
		self.a_max = a_max
	
		self.tot_time = tot_time

		self.P = P
		self.Pdot = Pdot
		self.Pddot = Pddot

		self.P_jax, self.Pdot_jax, self.Pddot_jax = jnp.asarray(self.P), jnp.asarray(self.Pdot), jnp.asarray(self.Pddot)
		self.nvar = jnp.shape(self.P_jax)[1]
	
		self.A_eq = jnp.vstack(( self.P_jax[0], self.Pdot_jax[0], self.Pddot_jax[0], self.Pdot_jax[-1], self.Pddot_jax[-1] ))
			
		self.A_vel = self.Pdot_jax 
		self.A_acc = self.Pddot_jax
		self.A_projection = jnp.identity(self.nvar)		

		self.num_target = num_target
		self.d_min_target = d_min_target
		self.d_max_target = d_max_target

		self.A_target = self.P_jax

		self.num_workspace = num_workspace
		self.a_workspace = a_workspace 
		self.b_workspace = b_workspace

		self.A_workspace = self.P_jax
		self.P_up_jax = P_up_jax
		self.Pdot_up_jax = Pdot_up_jax
		self.Pddot_up_jax = Pddot_up_jax

		self.W0 = None
		self.b0 = None
		self.W1 = None 
		self.b1 = None
		self.W2 = None
		self.b2 = None
		self.W3 = None
		self.b3 = None
		self.x_obs = None
		self.y_obs = None
		self.obstacle_points = None
		self.occlusion_weight = occlusion_weight

	@partial(jit, static_argnums=(0,))	
	def initial_alpha_d_obs(self, x_samples_init, y_samples_init, x_target, y_target):

		wc_alpha_target = (x_samples_init-x_target[:,jnp.newaxis])
		ws_alpha_target = (y_samples_init-y_target[:,jnp.newaxis])

		wc_alpha_target = wc_alpha_target.transpose(1, 0, 2)
		ws_alpha_target = ws_alpha_target.transpose(1, 0, 2)

		wc_alpha_target = wc_alpha_target.reshape(self.num_batch_projection, self.num*self.num_target)
		ws_alpha_target = ws_alpha_target.reshape(self.num_batch_projection, self.num*self.num_target)

		alpha_target = jnp.arctan2( ws_alpha_target, wc_alpha_target)
		c1_d = 1.0*self.rho_target*(jnp.cos(alpha_target)**2 + jnp.sin(alpha_target)**2 )
		c2_d = 1.0*self.rho_target*(wc_alpha_target*jnp.cos(alpha_target) + ws_alpha_target*jnp.sin(alpha_target)  )

		d_target = c2_d/c1_d	
		d_target = jnp.clip( d_target, self.d_min_target*jnp.ones((self.num_batch_projection,  self.num*self.num_target   )), self.d_max_target*jnp.ones((self.num_batch_projection,  self.num*self.num_target   ))  )

		return alpha_target, d_target		


	@partial(jit, static_argnums = (0,))
	def compute_projection(self, x_samples_init, y_samples_init, b_x_eq, b_y_eq, x_target, y_target, alpha_target, d_target, alpha_a, d_a, alpha_v, d_v, lamda_x, lamda_y, x_workspace, y_workspace, alpha_workspace, d_workspace, c_x_samples_init, c_y_samples_init):
		
		temp_x_target = d_target*jnp.cos(alpha_target)
		b_target_x = x_target.reshape(self.num*self.num_target)+temp_x_target
		 
		temp_y_target = d_target*jnp.sin(alpha_target)
		b_target_y = y_target.reshape(self.num*self.num_target)+temp_y_target

		temp_x_workspace = d_workspace*jnp.cos(alpha_workspace)*self.a_workspace
		b_workspace_x = x_workspace.reshape(self.num*self.num_workspace)+temp_x_workspace
		 
		temp_y_workspace = d_workspace*jnp.sin(alpha_workspace)*self.b_workspace
		b_workspace_y = y_workspace.reshape(self.num*self.num_workspace)+temp_y_workspace

		b_ax_ineq = d_a*jnp.cos(alpha_a)
		b_ay_ineq = d_a*jnp.sin(alpha_a)

		b_vx_ineq = d_v*jnp.cos(alpha_v)
		b_vy_ineq = d_v*jnp.sin(alpha_v)

		b_x_projection = x_samples_init
		b_y_projection = y_samples_init
		
		cost = 0.0*jnp.dot(self.Pddot_jax.T, self.Pddot_jax)+self.rho_projection*jnp.dot(self.A_projection.T, self.A_projection)+self.rho_ineq*jnp.dot(self.A_vel.T, self.A_vel)+self.rho_ineq*jnp.dot(self.A_acc.T, self.A_acc)+self.rho_target*jnp.dot(self.A_target.T, self.A_target)+self.rho_workspace*jnp.dot(self.A_workspace.T, self.A_workspace)

		cost_mat = jnp.vstack(( jnp.hstack(( cost, self.A_eq.T )), jnp.hstack((self.A_eq, jnp.zeros(( jnp.shape(self.A_eq)[0], jnp.shape(self.A_eq)[0] )) )) ))

		lincost_x = -lamda_x-self.rho_projection*jnp.dot(self.A_projection.T, c_x_samples_init.T).T-self.rho_ineq*jnp.dot(self.A_acc.T, b_ax_ineq.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, b_vx_ineq.T).T-self.rho_target*jnp.dot(self.A_target.T, b_target_x.T).T-self.rho_workspace*jnp.dot(self.A_workspace.T, b_workspace_x.T).T
		lincost_y = -lamda_y-self.rho_projection*jnp.dot(self.A_projection.T, c_y_samples_init.T).T-self.rho_ineq*jnp.dot(self.A_acc.T, b_ay_ineq.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, b_vy_ineq.T).T-self.rho_target*jnp.dot(self.A_target.T, b_target_y.T).T-self.rho_workspace*jnp.dot(self.A_workspace.T, b_workspace_y.T).T

		sol_x_temp = jnp.linalg.solve(cost_mat, jnp.hstack((-lincost_x, b_x_eq )).T ).T
		sol_y_temp = jnp.linalg.solve(cost_mat, jnp.hstack((-lincost_y, b_y_eq )).T ).T

		c_x_samples = sol_x_temp[:, 0:self.nvar]
		c_y_samples = sol_y_temp[:, 0:self.nvar]

		x_samples = jnp.dot(self.P_jax, c_x_samples.T).T 
		y_samples = jnp.dot(self.P_jax, c_y_samples.T).T

		xdot_samples = jnp.dot(self.Pdot_jax, c_x_samples.T).T 
		ydot_samples = jnp.dot(self.Pdot_jax, c_y_samples.T).T

		xddot_samples = jnp.dot(self.Pddot_jax, c_x_samples.T).T 
		yddot_samples = jnp.dot(self.Pddot_jax, c_y_samples.T).T 

		return c_x_samples, c_y_samples, x_samples, y_samples, xdot_samples, ydot_samples, xddot_samples, yddot_samples


	@partial(jit, static_argnums=(0,))	
	def compute_alph_d(self, x_samples, y_samples, xdot_samples, ydot_samples, xddot_samples, yddot_samples, x_target, y_target, lamda_x, lamda_y, x_workspace, y_workspace):

		#################################### Target

		wc_alpha_target = (x_samples-x_target[:,jnp.newaxis])
		ws_alpha_target = (y_samples-y_target[:,jnp.newaxis])

		wc_alpha_target = wc_alpha_target.transpose(1, 0, 2)
		ws_alpha_target = ws_alpha_target.transpose(1, 0, 2)

		wc_alpha_target = wc_alpha_target.reshape(self.num_batch_projection, self.num*self.num_target)
		ws_alpha_target = ws_alpha_target.reshape(self.num_batch_projection, self.num*self.num_target)

		alpha_target = jnp.arctan2( ws_alpha_target, wc_alpha_target)
		c1_d = 1.0*self.rho_target*(jnp.cos(alpha_target)**2 + jnp.sin(alpha_target)**2 )
		c2_d = 1.0*self.rho_target*(wc_alpha_target*jnp.cos(alpha_target) + ws_alpha_target*jnp.sin(alpha_target)  )

		d_target = c2_d/c1_d
		d_target = jnp.clip( d_target, self.d_min_target*jnp.ones((self.num_batch_projection,  self.num*self.num_target   )), self.d_max_target*jnp.ones((self.num_batch_projection,  self.num*self.num_target   ))  )

		#################################### Workspace

		wc_alpha_workspace = (x_samples-x_workspace[:,jnp.newaxis])
		ws_alpha_workspace = (y_samples-y_workspace[:,jnp.newaxis])

		wc_alpha_workspace = wc_alpha_workspace.transpose(1, 0, 2)
		ws_alpha_workspace = ws_alpha_workspace.transpose(1, 0, 2)

		wc_alpha_workspace = wc_alpha_workspace.reshape(self.num_batch_projection, self.num*self.num_workspace)
		ws_alpha_workspace = ws_alpha_workspace.reshape(self.num_batch_projection, self.num*self.num_workspace)

		alpha_workspace = jnp.arctan2( ws_alpha_workspace*self.a_workspace, wc_alpha_workspace*self.b_workspace)
		c1_d_workspace = 1.0*self.rho_workspace*(self.a_workspace**2*jnp.cos(alpha_workspace)**2 + self.b_workspace**2*jnp.sin(alpha_workspace)**2 )
		c2_d_workspace = 1.0*self.rho_workspace*(self.a_workspace*wc_alpha_workspace*jnp.cos(alpha_workspace) + self.b_workspace*ws_alpha_workspace*jnp.sin(alpha_workspace)  )

		d_workspace = c2_d_workspace/c1_d_workspace
		d_workspace = jnp.minimum(jnp.ones((self.num_batch_projection,  self.num*self.num_workspace   )), d_workspace   )

		####################### velocity terms

		wc_alpha_vx = xdot_samples
		ws_alpha_vy = ydot_samples
		alpha_v = jnp.arctan2( ws_alpha_vy, wc_alpha_vx)		

		c1_d_v = 1.0*self.rho_ineq*(jnp.cos(alpha_v)**2 + jnp.sin(alpha_v)**2 )
		c2_d_v = 1.0*self.rho_ineq*(wc_alpha_vx*jnp.cos(alpha_v) + ws_alpha_vy*jnp.sin(alpha_v)  )

		d_temp_v = c2_d_v/c1_d_v

		d_v = jnp.minimum(self.v_max*jnp.ones((self.num_batch_projection, self.num)), d_temp_v   )
		
		################# acceleration terms

		wc_alpha_ax = xddot_samples
		ws_alpha_ay = yddot_samples
		alpha_a = jnp.arctan2( ws_alpha_ay, wc_alpha_ax)		
		c1_d_a = 1.0*self.rho_ineq*(jnp.cos(alpha_a)**2 + jnp.sin(alpha_a)**2 )
		c2_d_a = 1.0*self.rho_ineq*(wc_alpha_ax*jnp.cos(alpha_a) + ws_alpha_ay*jnp.sin(alpha_a)  )

		d_temp_a = c2_d_a/c1_d_a

		d_a = jnp.minimum(self.a_max*jnp.ones((self.num_batch_projection, self.num)), d_temp_a   )

		#########################################33
		res_ax_vec = xddot_samples-d_a*jnp.cos(alpha_a)
		res_ay_vec = yddot_samples-d_a*jnp.sin(alpha_a)

		res_vx_vec = xdot_samples-d_v*jnp.cos(alpha_v)
		res_vy_vec = ydot_samples-d_v*jnp.sin(alpha_v)

		res_x_target_vec = wc_alpha_target-d_target*jnp.cos(alpha_target)
		res_y_target_vec = ws_alpha_target-d_target*jnp.sin(alpha_target)

		res_x_workspace_vec = wc_alpha_workspace-self.a_workspace*d_workspace*jnp.cos(alpha_workspace)
		res_y_workspace_vec = ws_alpha_workspace-self.b_workspace*d_workspace*jnp.sin(alpha_workspace)

		lamda_x = lamda_x-self.rho_ineq*jnp.dot(self.A_acc.T, res_ax_vec.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, res_vx_vec.T).T-self.rho_target*jnp.dot(self.A_target.T, res_x_target_vec.T).T-self.rho_workspace*jnp.dot(self.A_workspace.T, res_x_workspace_vec.T).T
		lamda_y = lamda_y-self.rho_ineq*jnp.dot(self.A_acc.T, res_ay_vec.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, res_vy_vec.T).T-self.rho_target*jnp.dot(self.A_target.T, res_y_target_vec.T).T-self.rho_workspace*jnp.dot(self.A_workspace.T, res_y_workspace_vec.T).T

		res_target_vec = jnp.hstack(( res_x_target_vec, res_y_target_vec  ))
		res_workspace_vec = jnp.hstack(( res_x_workspace_vec, res_y_workspace_vec  ))
		
		res_acc_vec = jnp.hstack(( res_ax_vec,  res_ay_vec  ))
		res_vel_vec = jnp.hstack(( res_vx_vec,  res_vy_vec  ))

		res_norm_batch = jnp.linalg.norm(res_acc_vec, axis =1)+jnp.linalg.norm(res_vel_vec, axis =1)+jnp.linalg.norm(res_target_vec, axis =1)+jnp.linalg.norm(res_workspace_vec, axis =1)

		res_target_norm = jnp.linalg.norm(res_target_vec)
		res_workspace_norm = jnp.linalg.norm(res_workspace_vec)
		
		res_acc_norm = jnp.linalg.norm(res_acc_vec)
		res_vel_norm = jnp.linalg.norm(res_vel_vec)

		return alpha_workspace, d_workspace, alpha_target, d_target, alpha_a, d_a, alpha_v, d_v, lamda_x, lamda_y, res_target_vec, res_acc_vec, res_vel_vec, res_target_norm, res_acc_norm, res_vel_norm, res_norm_batch	
	
	@partial(jit, static_argnums=(0,))
	def jax_where(self, p):

		return jnp.where(p<0.01, size = 1000)	
	
	@partial(jit, static_argnums=(0,))	
	def compute_inital_guess( self, x_samples_init, y_samples_init):

		cost_regression = jnp.dot(self.P_jax.T, self.P_jax)+0.0001*jnp.identity(self.nvar)
		lincost_regression_x = -jnp.dot(self.P_jax.T, x_samples_init.T).T 
		lincost_regression_y = -jnp.dot(self.P_jax.T, y_samples_init.T).T 

		cost_mat_inv = jnp.linalg.inv(cost_regression)

		c_x_samples_init = jnp.dot(cost_mat_inv, -lincost_regression_x.T).T 
		c_y_samples_init = jnp.dot(cost_mat_inv, -lincost_regression_y.T).T

		x_guess = jnp.dot(self.P_jax, c_x_samples_init.T).T
		y_guess = jnp.dot(self.P_jax, c_y_samples_init.T).T

		return c_x_samples_init, c_y_samples_init, x_guess, y_guess

	@partial(jit, static_argnums=(0,))	
	def compute_boundary_vec(self, x_init, vx_init, ax_init, y_init, vy_init, ay_init, vx_target, vy_target):

		x_init_vec = x_init*jnp.ones((self.num_batch_projection, 1))
		y_init_vec = y_init*jnp.ones((self.num_batch_projection, 1)) 

		vx_init_vec = vx_init*jnp.ones((self.num_batch_projection, 1))
		vy_init_vec = vy_init*jnp.ones((self.num_batch_projection, 1))

		ax_init_vec = ax_init*jnp.ones((self.num_batch_projection, 1))
		ay_init_vec = ay_init*jnp.ones((self.num_batch_projection, 1))

		vx_fin_vec = vx_target*jnp.ones((self.num_batch_projection, 1))
		vy_fin_vec = vy_target*jnp.ones((self.num_batch_projection, 1))

		ax_fin_vec = 0.0*jnp.ones((self.num_batch_projection, 1))
		ay_fin_vec = 0*jnp.ones((self.num_batch_projection, 1))
		
		b_eq_x = jnp.hstack(( x_init_vec, vx_init_vec, ax_init_vec, vx_fin_vec, ax_fin_vec ))
		b_eq_y = jnp.hstack(( y_init_vec, vy_init_vec, ay_init_vec, vy_fin_vec, ay_fin_vec ))
		
		return b_eq_x, b_eq_y
	
	@partial(jit, static_argnums=(0, 8))	

	def compute_initial_samples(self, eps_k, x_target_init, y_target_init, x_target_fin, y_target_fin, x_samples_shift, y_samples_shift, ellite_num_shift, x_init, y_init):

		num_shift = jnp.shape(x_samples_shift)[0]	

		x_target_init = x_init  
		y_target_init = y_init	

		goal_rot = -jnp.arctan2(y_target_fin-y_target_init, x_target_fin-x_target_init)

		x_init_temp = x_target_init*jnp.cos(goal_rot)-y_target_init*jnp.sin(goal_rot)
		y_init_temp = x_target_init*jnp.sin(goal_rot)+y_target_init*jnp.cos(goal_rot)

		x_fin_temp = x_target_fin*jnp.cos(goal_rot)-y_target_fin*jnp.sin(goal_rot)
		y_fin_temp = x_target_fin*jnp.sin(goal_rot)+y_target_fin*jnp.cos(goal_rot)

		x_interp = jnp.linspace(x_init_temp, x_fin_temp, self.num)
		y_interp = jnp.linspace(y_init_temp, y_fin_temp, self.num)

		x_guess_temp = x_interp+0.0*eps_k 
		y_guess_temp = y_interp+eps_k

		x_samples_init = x_guess_temp*jnp.cos(goal_rot)+y_guess_temp*jnp.sin(goal_rot)
		y_samples_init = -x_guess_temp*jnp.sin(goal_rot)+y_guess_temp*jnp.cos(goal_rot)

		x_samples_init = index_update(x_samples_init, index[0:num_shift, :], x_samples_shift[0:num_shift, :])
		y_samples_init = index_update(y_samples_init, index[0:num_shift, :], y_samples_shift[0:num_shift, :])

		return x_samples_init, y_samples_init

	@partial(jit, static_argnums = (0,) )
	def compute_projection_samples(self, x_init, vx_init, ax_init, y_init, vy_init, ay_init, alpha_a,
									 d_a, alpha_v, d_v, x_target, y_target, lamda_x, lamda_y, x_samples_init, y_samples_init, 
									 x_workspace, y_workspace, alpha_workspace, d_workspace, c_x_samples_init, 
									 c_y_samples_init, vx_target, vy_target):

		b_x_eq, b_y_eq = self.compute_boundary_vec(x_init, vx_init, ax_init, y_init, vy_init, ay_init, vx_target, vy_target)

		alpha_target, d_target = self.initial_alpha_d_obs(x_samples_init, y_samples_init, x_target, y_target)

		for i in range(0, self.maxiter_projection):

			c_x_samples, c_y_samples, x_samples, y_samples, xdot_samples, \
				ydot_samples, xddot_samples, yddot_samples = self.compute_projection(x_samples_init, y_samples_init, b_x_eq, b_y_eq, x_target, 
				y_target, alpha_target, d_target, alpha_a, d_a, alpha_v, d_v, lamda_x, 
				lamda_y, x_workspace, y_workspace, alpha_workspace, d_workspace, c_x_samples_init, c_y_samples_init)
		
			alpha_workspace, d_workspace, alpha_target, d_target, alpha_a, d_a, alpha_v, d_v, lamda_x, lamda_y, \
				res_target_vec, res_acc_vec, res_vel_vec, res_target_norm, \
					 res_acc_norm, res_vel_norm, res_norm_batch = self.compute_alph_d(x_samples, y_samples, xdot_samples, ydot_samples,
					  xddot_samples, yddot_samples, x_target, y_target, lamda_x, lamda_y, x_workspace, y_workspace)

		
		return x_samples, y_samples, xdot_samples, ydot_samples, xddot_samples, \
			yddot_samples, c_x_samples, c_y_samples, alpha_v, d_v, alpha_a, d_a, alpha_target, \
				d_target, lamda_x, lamda_y, res_norm_batch, alpha_workspace, d_workspace
 	   
	@partial(jit, static_argnums=(0,))	

	def compute_cost_batch(self, xddot_samples, yddot_samples, x_samples, y_samples, d_avg_target, x_target, y_target,obstacle_points): 

		mu =  2.3333
		std = 6.0117

		tiled_obstacle_points = jnp.tile(obstacle_points, (self.num * self.num_batch_cem,1))
		tiled_target_trajectory_x = jnp.tile(x_target, (self.num_batch_cem))
		tiled_target_trajectory_y = jnp.tile(y_target, (self.num_batch_cem))

		target_robot_matrix  = jnp.hstack((x_samples.reshape(self.num_batch_cem * self.num, 1), y_samples.reshape(self.num_batch_cem * self.num, 1), 
											tiled_target_trajectory_x.reshape(self.num_batch_cem * self.num, 1),
												tiled_target_trajectory_y.reshape(self.num_batch_cem * self.num, 1)))
		tiled_target_robot_matrix = jnp.repeat(target_robot_matrix, (obstacle_points.shape)[0], axis=0)
		input_matrix = jnp.hstack((tiled_target_robot_matrix, tiled_obstacle_points))

		input_matrix = (input_matrix - mu) / std

		A0 = jnp.maximum(0, self.W0 @ input_matrix.T + self.b0.T)
		A1 = jnp.maximum(0, self.W1 @ A0 + self.b1.T)  
		A2 = jnp.maximum(0, self.W2 @ A1 + self.b2.T)  
		occlusion_cost = (self.W3 @ A2 + self.b3.T)
		occlusion_cost = occlusion_cost

		occlusion_cost = occlusion_cost.reshape(self.num_batch_cem, self.num, (obstacle_points.shape)[0])
		occlusion_cost = jnp.sum(occlusion_cost, axis=2)
		occlusion_cost = jnp.maximum(occlusion_cost, 0)
		occlusion_cost = jnp.sum(occlusion_cost, axis=1)
		target_dist = ((x_samples-x_target)**2+(y_samples-y_target)**2-d_avg_target**2)
		cost_smoothness = ( jnp.linalg.norm(xddot_samples, axis = 1  )**2 +jnp.linalg.norm(yddot_samples, axis = 1  )**2 ) * 10
		total_cost = cost_smoothness + 0.8 * jnp.linalg.norm(target_dist, axis =1)**2 + occlusion_cost * self.occlusion_weight

		return total_cost
	
	@partial(jit, static_argnums=(0,))
	def compute_mean_covariance(self, c_x_ellite, c_y_ellite):

		c_x_mean = jnp.mean(c_x_ellite, axis = 0)
		c_y_mean = jnp.mean(c_y_ellite, axis = 0)

		cov_x = jnp.cov(c_x_ellite.T)
		cov_y = jnp.cov(c_y_ellite.T)

		return c_x_mean, c_y_mean, cov_x, cov_y

	@partial(jit,static_argnums=(0, ) )
	def compute_ellite_samples(self, key, c_x_mean, c_y_mean, cov_x, cov_y ):

		c_x_samples = jax.random.multivariate_normal(key, c_x_mean, cov_x + 0.005 * jnp.identity(self.nvar), (self.num_batch_projection, ))
		c_y_samples = jax.random.multivariate_normal(key, c_y_mean, cov_y + 0.005 * jnp.identity(self.nvar), (self.num_batch_projection, ))

		x_samples_init = jnp.dot(self.P_jax, c_x_samples.T).T 
		y_samples_init = jnp.dot(self.P_jax, c_y_samples.T).T 

		return c_x_samples, c_y_samples, x_samples_init, y_samples_init

	@partial(jit, static_argnums = (0,27) )
	def compute_cem(self, key, x_init, vx_init, ax_init, y_init, vy_init, ay_init, alpha_a, d_a, alpha_v, d_v,
					 x_target, y_target, lamda_x, lamda_y, x_samples_init, y_samples_init, x_workspace, y_workspace,
					  alpha_workspace, d_workspace, c_x_samples_init, c_y_samples_init, vx_target, vy_target,
					   d_avg_target, ellite_num_shift, obstacle_points):

		ellite_num_projection = self.num_batch_cem

		for i in range(0, self.maxiter_cem):

			key, subkey = random.split(key)

			x_samples, y_samples, xdot_samples, ydot_samples, xddot_samples, yddot_samples, c_x_samples, c_y_samples, \
			alpha_v, d_v, alpha_a, d_a, alpha_target, d_target, lamda_x, lamda_y, \
				 res_norm_batch, alpha_workspace, d_workspace = self.compute_projection_samples(x_init, vx_init, ax_init, y_init, vy_init, ay_init
																								, alpha_a, d_a, alpha_v, d_v, x_target, y_target, lamda_x, 
																								 lamda_y, x_samples_init, y_samples_init, x_workspace, y_workspace, 
																								 alpha_workspace, d_workspace, c_x_samples_init, c_y_samples_init, vx_target, vy_target)

			idx_ellite_projection = jnp.argsort(res_norm_batch)

			
			c_x_ellite_projection = c_x_samples[idx_ellite_projection[0:ellite_num_projection]]
			c_y_ellite_projection = c_y_samples[idx_ellite_projection[0:ellite_num_projection]]

			x_ellite_projection = x_samples[idx_ellite_projection[0:ellite_num_projection]]
			y_ellite_projection = y_samples[idx_ellite_projection[0:ellite_num_projection]]

			xddot_ellite_projection = xddot_samples[idx_ellite_projection[0:ellite_num_projection]]
			yddot_ellite_projection = yddot_samples[idx_ellite_projection[0:ellite_num_projection]]

			total_cost = self.compute_cost_batch(xddot_ellite_projection, yddot_ellite_projection, x_ellite_projection, y_ellite_projection,
													 d_avg_target, x_target, y_target, obstacle_points)

			idx = jnp.argsort(total_cost)
			idx_min = jnp.argmin(total_cost)

			ellite_num = int(ellite_num_projection * 0.3)

			c_x_ellite = c_x_ellite_projection[idx[0:ellite_num]]
			c_y_ellite = c_y_ellite_projection[idx[0:ellite_num]]
			
			x_ellite = x_ellite_projection[idx[0:ellite_num]]
			y_ellite = y_ellite_projection[idx[0:ellite_num]]

			c_x_mean, c_y_mean, cov_x, cov_y = self.compute_mean_covariance(c_x_ellite, c_y_ellite)
			c_x_samples, c_y_samples, x_samples_shift, y_samples_shift = self.compute_ellite_samples(key, c_x_mean, c_y_mean, cov_x, cov_y )

			c_x_best = c_x_ellite[0]
			c_y_best = c_y_ellite[0]

			x_samples_shift = jnp.vstack(( x_samples[idx_ellite_projection[0:ellite_num_shift]], x_ellite))
			y_samples_shift = jnp.vstack(( y_samples[idx_ellite_projection[0:ellite_num_shift]], y_ellite))

		
		return 	c_x_best, c_y_best, total_cost[idx_min], x_ellite_projection[idx_min], y_ellite_projection[idx_min], alpha_v, d_v, alpha_a, d_a, alpha_target, d_target, lamda_x, lamda_y, alpha_workspace, d_workspace, key, x_samples_shift, y_samples_shift

	@partial(jit, static_argnums = (0,) )
	def compute_controls(self, c_x_best, c_y_best, dt_up, vx_target, vy_target, 
							t_update, tot_time_copy_up, x_init, y_init, alpha_init,
                            x_target_init, y_target_init):
		
		num_average_samples = 10
		x_up = jnp.dot(self.P_up_jax, c_x_best)
		y_up = jnp.dot(self.P_up_jax, c_y_best)
		
		xddot_up = jnp.dot(self.Pddot_up_jax, c_x_best)
		yddot_up = jnp.dot(self.Pddot_up_jax, c_y_best)

		xdot_up = jnp.dot(self.Pdot_up_jax, c_x_best)
		ydot_up = jnp.dot(self.Pdot_up_jax, c_y_best)
		
		vx_control = jnp.mean(xdot_up[0:num_average_samples])
		vy_control = jnp.mean(ydot_up[0:num_average_samples])

		ax_control = jnp.mean(xddot_up[0:num_average_samples])
		ay_control = jnp.mean(yddot_up[0:num_average_samples])

		x_target_up = x_target_init + vx_target * tot_time_copy_up.flatten()
		y_target_up = y_target_init + vy_target * tot_time_copy_up.flatten()

		alpha_drone_temp = jnp.arctan2(y_target_up - y_up, x_target_up- x_up )
		alpha_drone = jnp.unwrap(jnp.hstack((alpha_init, alpha_drone_temp)))
		alpha_drone = alpha_drone[1:]

		vx_local = xdot_up * jnp.cos(alpha_drone) + ydot_up * jnp.sin(alpha_drone)
		vy_local = -xdot_up * jnp.sin(alpha_drone) + ydot_up * jnp.cos(alpha_drone)

		vx_control_local = jnp.mean(vx_local[0:num_average_samples])
		vy_control_local = jnp.mean(vy_local[0:num_average_samples])

		alphadot = jnp.diff(jnp.hstack((alpha_init,  alpha_drone)) )/dt_up
		alphadot_drone = jnp.mean(alphadot[0:num_average_samples])

		return vx_control_local, vy_control_local, ax_control, ay_control, \
						 alphadot_drone, jnp.mean(x_up[0:num_average_samples]), jnp.mean(y_up[0:num_average_samples]), vx_control, vy_control
		






