# -*- coding: utf-8 -*-
"""
@author: James Quigley
"""

import torch
import torch.nn as nn
import numpy as np
import time

# ScaleAndShiftLayer and TrainableFourierFeatureMap classes are built into the
# Fourier Model to avoid the need to manually compute an inverse fourier 
# transformation. By building these classes into the sequential network, the 
# automatic differenciator in PyTorch can compute the derivatives interms of the
# unscaled coordinates.

class ScaleAndShiftLayer(nn.Module):
    def __init__(self, scale_factors, means):
        super(ScaleAndShiftLayer, self).__init__()
        self.scale_factors = torch.tensor(scale_factors, dtype=torch.float32)
        self.means = torch.tensor(means, dtype=torch.float32)

    def forward(self, x):
        return x * self.scale_factors + self.means

class TrainableFourierFeatureMap(nn.Module):
    def __init__(self, num_bins, b_scales = [0.0015, 0.003, 0.15, 1/80,1]):
        super(TrainableFourierFeatureMap, self).__init__()
        self.B = torch.randn(5, num_bins)*2*np.pi*torch.tensor(b_scales, dtype=torch.float32).unsqueeze(-1)

    def forward(self, x):
        x_proj = x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

shape=(400, 700)
size = 60
nu=0.
gravity = 9.81

class FCNN():
    
    #This class is for the Fully Connected Neural Network
    
    def __init__(self, loss="MSE"):
        print("\nNew object initializing")
        self.network()

        if loss=="MSE":
          self.loss_func = nn.MSELoss()
          self.loss_pow=2
        else:
          self.loss_func = nn.L1Loss()
          self.loss_pow=1

        self.val_loss_func = nn.L1Loss()
        self.loss = 0
        self.loss_hist = []
        self.val_hist = []
        self.uvw_loss = 0
        self.fg_loss = 0
        self.epoch=0
        self.optimizer = torch.optim.LBFGS(
            self.net.parameters(), lr=1.0, max_iter=100, max_eval=None, tolerance_grad=1e-11,
            tolerance_change=1e-9, history_size=100, line_search_fn="strong_wolfe") #history_size=200
        print("\tNew Optimizer Created.")



        self.means =  torch.tensor([0, 0, -43.78, 0, 0, 12.3, -2.58, 0.29, 3.6], dtype=torch.float32)
        self.scaling_factors =torch.tensor(np.array((32,16,1,16,1,32,50,16,200)), dtype=torch.float32)

    def network(self):
        print("New network created")
        #Creating neural network with 5 inputs and 4 outputs which is 9 layers deep
        self.net = nn.Sequential(
            nn.Linear(5,size), nn.Tanh(),
            nn.Linear(size, size), nn.Tanh(),
            nn.Linear(size, size), nn.Tanh(),
            nn.Linear(size, size), nn.Tanh(),
            nn.Linear(size, size), nn.Tanh(),
            nn.Linear(size, size), nn.Tanh(),
            nn.Linear(size, size), nn.Tanh(),
            nn.Linear(size, size), nn.Tanh(),
            nn.Linear(size, size), nn.Tanh(),
            nn.Linear(size, size), nn.Tanh(),
            nn.Linear(size, 4)
        )

    def function(self, x, y, z, angle, angle_diff):

        res = self.net(torch.hstack((x, y, z, angle, angle_diff)))
        u, v, w, p_star = res[:, 0:1], res[:, 1:2], res[:, 2:3] , res[:, 3:4]

        if self.use_c or self.use_m:
            u_x  = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            v_y  = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            w_z  = torch.autograd.grad(w, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        if self.use_c:
            # add considrartion of continuty (mass conservation equation)
            c = u_x*self.scaling_factors[5]/self.scaling_factors[0] +\
                v_y*self.scaling_factors[6]/self.scaling_factors[1] +\
                w_z*self.scaling_factors[7]/self.scaling_factors[2]
            self.c =self.lambda_c*c
        else:
           self.c = self.null

        if self.use_m:
            # compute the u gradients
            u_y  = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            u_z  = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]

            # compute the v gradients
            v_x  = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            v_z  = torch.autograd.grad(v, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]

            # compute w gradients
            w_x  = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            w_y  = torch.autograd.grad(w, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]

            #compute second partial derivatives
            u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
            u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            v_zz = torch.autograd.grad(v_z, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            w_zz = torch.autograd.grad(w_z, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]


            # compute the modified pressure gradients
            p_x  = torch.autograd.grad(p_star, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]*self.scaling_factors[8]/self.scaling_factors[0]
            p_y  = torch.autograd.grad(p_star, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]*self.scaling_factors[8]/self.scaling_factors[1]
            p_z  = torch.autograd.grad(p_star, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]*self.scaling_factors[8]/self.scaling_factors[2]


            #momentume in x, y, z directions. I set u_t =0 since our model is assumed steady-state
            m_x = (u*self.scaling_factors[5]+self.means[5]) * u_x*self.scaling_factors[5]/self.scaling_factors[0] +\
                  (v*self.scaling_factors[6]+self.means[6]) * u_y*self.scaling_factors[5]/self.scaling_factors[1] +\
                  (w*self.scaling_factors[7]+self.means[7]) * u_z*self.scaling_factors[5]/self.scaling_factors[2] +\
                  p_x #-\
                  #nu * (u_xx*self.scaling_factors[5]/self.scaling_factors[0]**2 +\
                  #      u_yy*self.scaling_factors[5]/self.scaling_factors[1]**2 +\
                  #      u_zz*self.scaling_factors[5]/self.scaling_factors[2]**2)

            m_y = (u*self.scaling_factors[5]+self.means[5]) * v_x*self.scaling_factors[6]/self.scaling_factors[0] +\
                  (v*self.scaling_factors[6]+self.means[6]) * v_y*self.scaling_factors[6]/self.scaling_factors[1] +\
                  (w*self.scaling_factors[7]+self.means[7]) * v_z*self.scaling_factors[6]/self.scaling_factors[2] +\
                  p_y #-\
                 #nu * (v_xx*self.scaling_factors[6]/self.scaling_factors[0] +\
                 #      v_yy*self.scaling_factors[6]/self.scaling_factors[1] +\
                 #      v_zz*self.scaling_factors[6]/self.scaling_factors[2])

            m_z =(u*self.scaling_factors[5]+self.means[5]) * w_x*self.scaling_factors[7]/self.scaling_factors[0] +\
                 (v*self.scaling_factors[6]+self.means[6]) * w_y*self.scaling_factors[7]/self.scaling_factors[1] +\
                 (w*self.scaling_factors[7]+self.means[7]) * w_z*self.scaling_factors[7]/self.scaling_factors[2] +\
                  p_z - gravity# - \
                 #nu * (w_xx*self.scaling_factors[7]/self.scaling_factors[0] +\
                 #      w_yy*self.scaling_factors[7]/self.scaling_factors[1] +\
                 #      w_zz*self.scaling_factors[7]/self.scaling_factors[2])


            self.mx=m_x
            self.my=m_y
            self.mz=self.null
        else:
            self.mx = self.my = self.mz = self.null
        return u, v, w, p_star, self.c, self.mx, self.my, self.mz

    def closure(self):
        self.optimizer.zero_grad()

        u_pred, v_pred, w_pred, p_pred, c_pred, m_x_pred, m_y_pred, m_z_pred = self.function(self.x, self.y, self.z, self.angle, self.angle_diff)
        self.pi_null = torch.zeros((m_x_pred.shape[0], 1))

        if self.use_data:
            u_loss = self.loss_func(u_pred[:self.len], self.u)*self.scaling_factors[5]
            v_loss = self.loss_func(v_pred[:self.len], self.v)*self.scaling_factors[6]
            w_loss = self.loss_func(w_pred[:self.len], self.w)*self.scaling_factors[7]
            self.uvw_loss = u_loss + v_loss + w_loss
            self.p_loss = self.lambda_p*self.loss_func(p_pred[:self.len], self.p_star)*self.scaling_factors[7]
        else:
            self.uvw_loss = 0
            self.p_loss = 0

        if self.use_m:
            self.pi_null = torch.zeros((m_x_pred.shape[0], 1))
            m_x_loss = self.loss_func(m_x_pred, self.pi_null) #do scaling factors need to be included here?
            m_y_loss = self.loss_func(m_y_pred, self.pi_null)
            #m_z_loss = self.loss_func(m_z_pred, self.pi_null)
            self.m_loss = self.lambda_m*(2*m_x_loss + m_y_loss)
        else:
            self.m_loss = 0

        if self.use_c:
            self.pi_null = torch.zeros((c_pred.shape[0], 1))
            self.c_loss = self.loss_func(c_pred, self.pi_null)
        else:
            self.c_loss = 0


        self.loss = self.uvw_loss + self.p_loss + self.c_loss + self.m_loss #  ####change this back!

        self.validate()

        self.loss.backward()
        self.epoch += 1
        if not self.epoch%100:
            print('Epoch: {:}, Loss: {:0.6f},\tValidation Loss: {:0.6f}, \tUVW Loss: {:0.6f}, \tP Loss: {:0.6f}, \tC Loss: {:0.6f}, M Loss: {:0.6f}'.format(self.epoch, self.loss, self.val_loss, self.uvw_loss, self.p_loss, self.c_loss,  self.m_loss))

        self.loss_hist.append(self.loss.item())

        return self.loss

    def validate(self):

        res = self.net(torch.hstack((self.val_x, self.val_y, self.val_z, self.val_angle, self.val_angle_diff)))
        u_pred, v_pred, w_pred, p_pred = res[:, 0:1], res[:, 1:2], res[:, 2:3] , res[:, 3:4]

        u_loss = self.val_loss_func(u_pred, self.val_u)*self.scaling_factors[5]
        v_loss = self.val_loss_func(v_pred, self.val_v)*self.scaling_factors[6]
        w_loss = self.val_loss_func(w_pred, self.val_w)*self.scaling_factors[7]
        #p_loss = self.val_loss_func(p_pred, self.val_p_star)

        self.val_loss = (u_loss + v_loss + w_loss) # + p_loss
        self.val_hist.append(self.val_loss.item())


    def load_training_data(self, X, Y, Z, angle, angle_diff, U, V, W, P_star, angles_used):

        #Scaling the data
        data = torch.tensor(np.hstack((X, Y, Z, angle, angle_diff, U, V, W, P_star)), dtype=torch.float32)
        data = (data - self.means)/self.scaling_factors

        #Saving the scaled data
        self.x, self.y, self.z, self.angle, self.angle_diff, self.u, self.v, self.w, self.p_star = torch.split(data, 1, dim=1)
        self.x.requires_grad_()
        self.y.requires_grad_()
        self.z.requires_grad_()
        self.deg = angles_used
        self.null = torch.zeros((self.x.shape[0], 1))
        self.len = self.x.shape[0]

    def load_pinn_data(self, X, Y, Z, angle, angle_diff):

        #Takes inputs (training data without labels) as 1D numpy arrays and saves them.
        #This data is used for physics-only training.

        data = torch.tensor(np.hstack((X, Y, Z, angle, angle_diff)), dtype=torch.float32)
        data = (data - self.means[:5])/self.scaling_factors[:5]
        self.p_x, self.p_y, self.p_z, self.p_angle, self.p_angle_diff = torch.split(data, 1, dim=1)
        self.x          = torch.cat((self.x, self.p_x), dim=0)
        self.y          = torch.cat((self.y, self.p_y), dim=0)
        self.z          = torch.cat((self.z, self.p_x), dim=0)
        self.angle      = torch.cat((self.angle, self.p_angle), dim=0)
        self.angle_diff = torch.cat((self.angle_diff, self.p_angle_diff), dim=0)


    def load_validaiton_data(self, X, Y, Z, angle, angle_diff, U, V, W, P_star, angles_used):

        #takes validation inputs and outputs as 1D numpy arrays and saves them as tensors

        print("Loading new validation tensors...")
        data = torch.tensor(np.hstack((X, Y, Z, angle, angle_diff, U, V, W, P_star)), dtype=torch.float32)
        data = (data - self.means)/self.scaling_factors

        #Saving the scaled data
        self.val_x, self.val_y, self.val_z, self.val_angle, self.val_angle_diff, self.val_u, self.val_v, self.val_w, self.val_p_star = torch.split(data, 1, dim=1)
        self.val_deg = angles_used
        self.val_null = torch.zeros((self.val_x.shape[0], 1))
        self.val_loss =0
        print("New validation tensors Loaded!\n")

    def predict(self, x, y, z, angle, angle_diff, shape=shape):

        #Takes NN inputs as 1D numpy arrays and resurns NN output predictions in the correct shape

        data = torch.tensor(np.hstack((x, y, z, angle, angle_diff)), dtype=torch.float32)
        data = (data - self.means[:5])/self.scaling_factors[:5]
        res = self.net(data)*self.scaling_factors[-4:] +self.means[-4:]

        return tuple([i.data.cpu().numpy().reshape(shape) for i in res.split(1, dim=1)])



    def train(self, epochs=25, use_c=True,use_m=True, use_data=True, lr=0.2, skip_n=1,  p=0.015, c=30, m=0.03, skip_fraction=0/9):
        print("Beginning Training..")
        start_time = time.time()
        self.use_c = use_c
        self.use_m = use_m
        self.use_data = use_data
        self.skip_n = skip_n
        self.lambda_p = p
        self.lambda_c = c
        self.lambda_m = m
        self.skip_fraction = skip_fraction
        for group in self.optimizer.param_groups:
          group['max_iter']=epochs
          group['max_eval']=epochs
          group['lr']=lr

        if self.skip_n!=1:
          print("Skipping every", skip_n, "data points.")

        print("Traing:\tUse Data=", self.use_data,
              "Traing:\tUse Momentum Eqs=", self.use_m,
              "\tUse Mass Conservation=", self.use_c,
              "\tEpochs = ", self.optimizer.param_groups[0]['max_iter'],
              "\tLearning Rate = ", self.optimizer.param_groups[0]['lr'],)
        print("\tTraining angles = ",self.deg, "\tValidation angles =", self.val_deg)
        print("nu = ", nu)


        self.net.train()
        self.optimizer.step(self.closure)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nTraining complete.\tElapsed time: {int(elapsed_time)} seconds")


    def get_loss(self):
        return self.loss_hist, self.val_hist



class FFNN():
    #This class has the Fourier Feature Layer built into the NN
    def __init__(self, loss="MSE", bins = 50, b_scales = [0.0012, 0.003, 0.15, 1/80,1]):
        print("\nNew object initializing")

        size=bins*2
        self.network(size, b_scales, [32.,  50.,  30., 220.],[0, -2.97, 0.25, 4.5623])

        if loss=="MSE":
          self.loss_func = nn.MSELoss()
          self.loss_pow=2
        else:
          self.loss_func = nn.L1Loss()
          self.loss_pow=1

        self.val_loss_func = nn.L1Loss()
        self.loss = 0
        self.loss_hist = []
        self.val_hist = []
        self.uvw_loss = 0
        self.fg_loss = 0
        self.epoch=0
        self.optimizer = torch.optim.LBFGS(
            self.net.parameters(), lr=1.0, max_iter=100, max_eval=None, tolerance_grad=1e-12,
            tolerance_change=1e-12, history_size=100, line_search_fn="strong_wolfe")
        print("\tNew Optimizer Created.")


    def network(self, bins, b_scale, scale_factors, constants):
        print("New network created")
        size = bins*2

        #The model is composed of a Fourier Transformation layer, 9 hidden layers and an output scalling layer.
        self.net = nn.Sequential(
            TrainableFourierFeatureMap(bins, b_scale),
            nn.Linear(size,size), nn.Tanh(),
            nn.Linear(size,size), nn.Tanh(),
            nn.Linear(size,size), nn.Tanh(),
            nn.Linear(size,size), nn.Tanh(),
            nn.Linear(size,size), nn.Tanh(),
            nn.Linear(size,size), nn.Tanh(),
            nn.Linear(size,size), nn.Tanh(),
            nn.Linear(size,size), nn.Tanh(),
            nn.Linear(size,size), nn.Tanh(),
            nn.Linear(size, 4),
            ScaleAndShiftLayer(scale_factors, constants)
        )

    def function(self, x, y, z, angle, angle_diff):

        #self.net is the forward pass of the model.
        #Importantly if x, y, z have requires_grad_(),
        #gradient data with respect to these inputs will be calculated and stored in the resulting tensors.

        res = self.net(torch.hstack((x, y, z, angle, angle_diff)))
        u, v, w, p_star = res[:, 0:1], res[:, 1:2], res[:, 2:3] , res[:, 3:4]

        if self.use_c or self.use_m:
            u_x  = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            v_y  = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            w_z  = torch.autograd.grad(w, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        if self.use_c:
            # add considrartion of continuty (mass conservation equation)
            self.c =self.lambda_c*(u_x + v_y + w_z)
        else:
           self.c = self.null

        if self.use_m:
            # compute the u gradients
            u_y  = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            u_z  = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]

            # compute the v gradients
            v_x  = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            v_z  = torch.autograd.grad(v, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]

            # compute w gradients
            w_x  = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            w_y  = torch.autograd.grad(w, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]

            #compute second partial derivatives
            u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
            u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            v_zz = torch.autograd.grad(v_z, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            #w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            #w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            #w_zz = torch.autograd.grad(w_z, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]


            # compute the modified pressure gradients
            p_x  = torch.autograd.grad(p_star, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            p_y  = torch.autograd.grad(p_star, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            p_z  = torch.autograd.grad(p_star, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]


            #momentume in x, y, z directions. I set u_t =0 since our model is assumed steady-state
            m_x = u*u_x + v*u_y + w*u_z + p_x - nu * (u_xx + u_yy + u_zz)
            m_y = u*v_x + v*v_y + w*v_z + p_y - nu * (v_xx + v_yy + v_zz)
            #m_z = u*w_x + v*w_y + w*w_z + p_z - nu * (w_xx + w_yy + w_zz) - g
            self.mx=m_x
            self.my=m_y
            self.mz=self.null
        else:
            self.mx = self.my = self.mz = self.null
        return u, v, w, p_star, self.c, self.mx, self.my, self.mz

    def closure(self):
        self.optimizer.zero_grad()
        u_pred, v_pred, w_pred, p_pred, c_pred, m_x_pred, m_y_pred, m_z_pred = self.function(self.x, self.y, self.z, self.angle, self.angle_diff)

        if self.use_data:
            u_loss = self.loss_func(u_pred[:self.len], self.u)
            v_loss = self.loss_func(v_pred[:self.len], self.v)
            w_loss = self.loss_func(w_pred[:self.len], self.w)
            self.uvw_loss = u_loss + v_loss + w_loss
            self.p_loss = self.lambda_p*self.loss_func(p_pred[:self.len], self.p_star)
        else:
            self.uvw_loss = 0
            self.p_loss = 0

        if self.use_m:
            self.pi_null = torch.zeros((m_x_pred.shape[0], 1))
            m_x_loss = self.loss_func(m_x_pred, self.pi_null) #do scaling factors need to be included here?
            m_y_loss = self.loss_func(m_y_pred, self.pi_null)
            #m_z_loss = self.loss_func(m_z_pred, self.pi_null)
            self.m_loss = self.lambda_m*(2*m_x_loss + m_y_loss)
        else:
            self.m_loss = 0

        if self.use_c:
            self.pi_null = torch.zeros((c_pred.shape[0], 1))
            self.c_loss = self.loss_func(c_pred, self.pi_null)
        else:
            self.c_loss = 0

        self.loss = self.uvw_loss + self.p_loss + self.c_loss + self.m_loss

        self.validate()

        self.loss.backward()
        self.epoch += 1
        if not self.epoch%100:
            print('Epoch: {:}, Loss: {:0.3f},\tValidation Loss: {:0.3f}, \tUVW Loss: {:0.3f}, \tP Loss: {:0.3f}, \tC Loss: {:0.3f}, M Loss: {:0.3f}'.format(self.epoch, self.loss, self.val_loss, self.uvw_loss, self.p_loss, self.c_loss,  self.m_loss))

        self.loss_hist.append(self.loss.item())

        return self.loss

    def validate(self):

        #tests model against validation data and appends the result to val_hist

        val_data = torch.hstack((self.val_x, self.val_y, self.val_z, self.val_angle, self.val_angle_diff))
        res = self.net(val_data)
        u_pred, v_pred, w_pred, p_pred = res[:, 0:1], res[:, 1:2], res[:, 2:3] , res[:, 3:4]

        u_loss = self.val_loss_func(u_pred, self.val_u)
        v_loss = self.val_loss_func(v_pred, self.val_v)
        w_loss = self.val_loss_func(w_pred, self.val_w)
        #p_loss = self.val_loss_func(p_pred, self.val_p_star)

        self.val_loss = (u_loss + v_loss + w_loss) # + p_loss
        self.val_hist.append(self.val_loss.item())


    def load_training_data(self, X, Y, Z, angle, angle_diff, U, V, W, P_star, angles_used):

        #loads labelled training data as 1D numpy arrays and saves them as tensors

        print("Loading new training tensors...")
        data = torch.tensor(np.hstack((X, Y, Z, angle, angle_diff, U, V, W, P_star)), dtype=torch.float32)

        #Saving the tensor data
        self.x, self.y, self.z, self.angle, self.angle_diff, self.u, self.v, self.w, self.p_star = data.split(1, dim=1)
        self.x.requires_grad_() #gradients are required on x, y, & z as we take derivatives w.r.t these coordinates.
        self.y.requires_grad_()
        self.z.requires_grad_()
        self.deg = angles_used
        self.null = torch.zeros((self.x.shape[0], 1))
        self.len = self.x.shape[0]

    def load_pinn_data(self, X, Y, Z, angle, angle_diff):

        #takes NN inputs (training data without labels) as 1D numpy arrays and saves them

        data = torch.tensor(np.hstack((X, Y, Z, angle, angle_diff)), dtype=torch.float32)
        self.p_x, self.p_y, self.p_z, self.p_angle, self.p_angle_diff = torch.split(data, 1, dim=1)
        self.x = torch.cat((self.x, self.p_x), dim=0)
        self.y = torch.cat((self.y, self.p_y), dim=0)
        self.z = torch.cat((self.z, self.p_x), dim=0)
        self.angle = torch.cat((self.angle, self.p_angle), dim=0)
        self.angle_diff = torch.cat((self.angle_diff, self.p_angle_diff), dim=0)

    def load_validaiton_data(self, X, Y, Z, angle, angle_diff, U, V, W, P_star, angles_used):

        #takes validation inputs and outputs as 1D numpy arrays and saves them as tensors

        print("Loading new validation tensors...")
        data = torch.tensor(np.hstack((X, Y, Z, angle, angle_diff, U, V, W, P_star)), dtype=torch.float32)

        #Saving the scaled data
        self.val_x, self.val_y, self.val_z, self.val_angle, self.val_angle_diff, self.val_u, self.val_v, self.val_w, self.val_p_star = torch.split(data, 1, dim=1)
        self.val_deg = angles_used
        self.val_null = torch.zeros((self.val_x.shape[0], 1))
        self.val_loss =0
        print("New validation tensors Loaded!\n")

    def predict(self, x, y, z, angle, angle_diff, shape=shape):

        #function takes NN inputs as 1D numpy arrays and resurns NN outputs in the correct shape

        data = torch.tensor(np.hstack((x, y, z, angle, angle_diff)), dtype=torch.float32)
        res = self.net(data)
        u, v, w, p = res[:, 0:1], res[:, 1:2], res[:, 2:3] , res[:, 3:4]
        return tuple([i.data.cpu().numpy().reshape(shape) for i in [u, v, w, p]])



    def train(self, epochs=25, use_c=True,use_m=True, use_data=True, lr=1.0,  p=0.015, c=30, m=0.03):
        print("Beginning Training..")
        start_time = time.time()
        self.use_c = use_c
        self.use_m = use_m
        self.use_data = use_data
        self.lambda_p = p
        self.lambda_c = c
        self.lambda_m = m
        for group in self.optimizer.param_groups:
          group['max_iter']=epochs
          group['max_eval']=epochs
          group['lr']=lr

        print("Traing:\tUse Data=", self.use_data,
              "Traing:\tUse Momentum Eqs=", self.use_m,
              "\tUse Mass Conservation=", self.use_c,
              "\tEpochs = ", self.optimizer.param_groups[0]['max_iter'],
              "\tLearning Rate = ", self.optimizer.param_groups[0]['lr'],)
        print("\tTraining angles = ",self.deg, "\tValidation angles =", self.val_deg)

        self.net.train()
        self.optimizer.step(self.closure)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nLowested Validation {np.min(self.val_hist)}.\tElapsed time: {int(elapsed_time)} seconds")
        #print(f"\nLowested Validation {np.min(sel.val_hist)}")

    def get_loss(self):
        return self.loss_hist, self.val_hist