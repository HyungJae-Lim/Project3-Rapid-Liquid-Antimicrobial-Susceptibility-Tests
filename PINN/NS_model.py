# PINN for incompressible 2-D NS equation (steady version)
# coded by St.Watermelon

from flow_boundary_condition import flow_data
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from time import time

class IncompressibleNet(Model):

    def __init__(self):
        super(IncompressibleNet, self).__init__()

        initializer = tf.keras.initializers.GlorotUniform 
        self.h1 = Dense(50, activation='tanh', kernel_initializer=initializer)
        self.h2 = Dense(50, activation='tanh', kernel_initializer=initializer)
        self.h3 = Dense(50, activation='tanh', kernel_initializer=initializer)
        self.h4 = Dense(50, activation='tanh', kernel_initializer=initializer)
        self.h5 = Dense(50, activation='tanh', kernel_initializer=initializer)
        self.u = Dense(6, activation='linear', kernel_initializer=initializer)


    def call(self, pos):
        x = self.h1(pos)  # pos = (x, y)
        x = self.h2(x)
        x = self.h3(x)
        x = self.h4(x)
        x = self.h5(x)
        out = self.u(x)  # u,v,p,sig_xx,sig_xy,sig_yy

        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]
        sig_xx = out[:, 3:4]
        sig_xy = out[:, 4:5]
        sig_yy = out[:, 5:6]
        return u, v, p, sig_xx, sig_xy, sig_yy


class NSpinn(object):

    def __init__(self):

        self.lr = 0.0005
        self.opt = Adam(self.lr)

        # density and viscosity
        self.rho = 1.0
        self.mu = 0.02

        self.flow = IncompressibleNet()
        self.flow.build(input_shape=(None, 2))

        self.train_loss_history = []
        self.iter_count = 0
        self.instant_loss = 0

    def ns_net(self, xy):
        x = xy[:, 0:1]
        y = xy[:, 1:2]

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            xy_c = tf.concat([x,y], axis=1)
            u, v, p, sig_xx, sig_xy, sig_yy = self.flow(xy_c)
            u_x = tape.gradient(u, x)
        u_y = tape.gradient(u, y)
        v_x = tape.gradient(v, x)
        v_y = tape.gradient(v, y)
        sig_xx_x = tape.gradient(sig_xx, x)
        sig_yy_y = tape.gradient(sig_yy, y)
        sig_xy_x = tape.gradient(sig_xy, x)
        sig_xy_y = tape.gradient(sig_xy, y)
        del tape

        r_1 = self.rho * (u*u_x+v*u_y) - sig_xx_x - sig_xy_y
        r_2 = self.rho * (u*v_x+v*v_y) - sig_xy_x - sig_yy_y
        r_3 = -p + 2*self.mu*u_x - sig_xx
        r_4 = -p + 2*self.mu*v_y - sig_yy
        r_5 = self.mu*(u_y+v_x) - sig_xy
        r_6 = u_x+v_y

        return r_1, r_2, r_3, r_4, r_5, r_6


    def compute_loss(self, r_1, r_2, r_3, r_4, r_5, r_6,  \
                    u_hat, v_hat, uv_bnd_sol, p_hat, outlet_p):
        u_sol = uv_bnd_sol[:, 0:1]
        v_sol = uv_bnd_sol[:, 1:2]
        loss_bnd = tf.reduce_mean(tf.square(u_hat-u_sol)) \
                 + tf.reduce_mean(tf.square(v_hat-v_sol))
        loss_outlet = tf.reduce_mean(tf.square(p_hat-outlet_p))
        loss_col = tf.reduce_mean(tf.square(r_1)) \
                 + tf.reduce_mean(tf.square(r_2)) \
                 + tf.reduce_mean(tf.square(r_3)) \
                 + tf.reduce_mean(tf.square(r_4)) \
                 + tf.reduce_mean(tf.square(r_5)) \
                 + tf.reduce_mean(tf.square(r_6))

        return loss_bnd+loss_outlet+loss_col


    def save_weights(self, path):
        self.flow.save_weights(path + 'flow.h5')


    def load_weights(self, path):
        self.flow.load_weights(path + 'flow.h5')


    def compute_grad(self, xy_col, xy_bnd, uv_bnd_sol, outlet_xy, outlet_p):
        with tf.GradientTape() as tape:
            r_1, r_2, r_3, r_4, r_5, r_6 = self.ns_net(xy_col)
            u_hat, v_hat, _, _, _, _ = self.flow(xy_bnd)
            _, _, p_hat, _, _, _ = self.flow(outlet_xy)
            loss = self.compute_loss(r_1, r_2, r_3, r_4, r_5, r_6, \
                                     u_hat, v_hat, uv_bnd_sol, \
                                     p_hat, outlet_p)

        grads = tape.gradient(loss, self.flow.trainable_variables)

        return loss, grads


    def callback(self, arg=None):
        if self.iter_count % 10 == 0:
            print('iter=', self.iter_count, ', loss=', self.instant_loss)
            self.train_loss_history.append([self.iter_count, self.instant_loss])
        self.iter_count += 1



    def train_with_adam(self, xy_col, xy_bnd, uv_bnd_sol, outlet_xy, outlet_p, adam_num):

        @tf.function
        def learn():
            loss, grads = self.compute_grad(xy_col, xy_bnd, \
                                            uv_bnd_sol, outlet_xy, outlet_p)
            self.opt.apply_gradients(zip(grads, self.flow.trainable_variables))

            return loss

        for iter in range(int(adam_num)):

            loss = learn()

            self.instant_loss = loss.numpy()
            self.callback()


    def train_with_lbfgs(self, xy_col, xy_bnd, uv_bnd_sol, outlet_xy, outlet_p, lbfgs_num):

        def vec_weight():
            # vectorize weights
            weight_vec = []

            # Loop over all weights
            for v in self.flow.trainable_variables:
                weight_vec.extend(v.numpy().flatten())

            weight_vec = tf.convert_to_tensor(weight_vec)
            return weight_vec
        w0 = vec_weight().numpy()

        def restore_weight(weight_vec):
            # restore weight vector to model weights
            idx = 0
            for v in self.flow.trainable_variables:
                vs = v.shape

                # weight matrices
                if len(vs) == 2:
                    sw = vs[0] * vs[1]
                    updated_val = tf.reshape(weight_vec[idx:idx + sw], (vs[0], vs[1]))
                    idx += sw

                # bias vectors
                elif len(vs) == 1:
                    updated_val = weight_vec[idx:idx + vs[0]]
                    idx += vs[0]

                # assign variables (Casting necessary since scipy requires float64 type)
                v.assign(tf.cast(updated_val, dtype=tf.float32))


        def loss_grad(w):
            # update weights in model
            restore_weight(w)

            loss, grads = self.compute_grad(xy_col, xy_bnd, \
                                            uv_bnd_sol, outlet_xy, outlet_p)
            # vectorize gradients
            grad_vec = []
            for g in grads:
                grad_vec.extend(g.numpy().flatten())

            # gradient list to array
            # scipy-routines requires 64-bit floats
            loss = loss.numpy().astype(np.float64)
            self.instant_loss = loss
            grad_vec = np.array(grad_vec, dtype=np.float64)

            return loss, grad_vec

        return scipy.optimize.minimize(fun=loss_grad,
                                       x0=w0,
                                       jac=True,
                                       method='L-BFGS-B',
                                       callback=self.callback,
                                       options={'maxiter': lbfgs_num,
                                                'maxfun': 50000,
                                                'maxcor': 50,
                                                'maxls': 50,
                                                'ftol': 1.0 * np.finfo(float).eps})



    def predict(self, xy):
        u, v, p, _, _, _ = self.flow(xy)
        return u, v, p


    def train(self, adam_num, lbfgs_num):

        xy_col, xy_bnd, uv_bnd_sol, outlet_xy, outlet_p = flow_data()

        # Start timer
        t0 = time()
        self.train_with_adam(xy_col, xy_bnd, uv_bnd_sol, outlet_xy, outlet_p, adam_num)
        # Print computation time
        print('\nComputation time of adam: {} seconds'.format(time() - t0))
        t1 = time()
        self.train_with_lbfgs(xy_col, xy_bnd, uv_bnd_sol, outlet_xy, outlet_p, lbfgs_num)
        # Print computation time
        print('\nComputation time of L-BFGS-B: {} seconds'.format(time() - t1))

        self.save_weights("./save_weights/")

        np.savetxt('./save_weights/loss.txt', self.train_loss_history)
        train_loss_history = np.array(self.train_loss_history)

        plt.plot(train_loss_history[:, 0], train_loss_history[:, 1])
        plt.yscale("log")
        plt.show()


# main
def main():

    adam_num = 10001
    lbfgs_num = 20000
    agent = NSpinn()

    agent.train(adam_num, lbfgs_num)


if __name__=="__main__":
    main()