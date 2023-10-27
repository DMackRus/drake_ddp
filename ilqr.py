##
#
# A simple implementation of iterative LQR (iLQR) for discrete-time systems in Drake.
#
##

from pydrake.all import *
import time
import numpy as np
import utils_derivs_interpolation
import csv
import os
import matplotlib.pyplot as plt

class IterativeLinearQuadraticRegulator():
    """
    Set up and solve a trajectory optimization problem of the form

        min_{u} sum{ (x-x_nom)'Q(x-x_nom) + u'Ru } + (x-x_nom)'Qf(x-x_nom)
        s.t.    x_{t+1} = f(x_t, u_t)

    using iLQR.
    """
    def __init__(self, system, num_timesteps, 
            input_port_index=0, delta=1e-2, beta=0.95, gamma=0.0, derivs_keypoint_methods = None):
        """
        Args:
            system:             Drake System describing the discrete-time dynamics
                                 x_{t+1} = f(x_t,u_t). Must be discrete-time.
            num_timesteps:      Number of timesteps to consider in the optimization.
            input_port_index:   InputPortIndex for the control input u_t. Default is to
                                 use the first port. 
            delta:              Termination criterion - the algorithm ends when the improvement
                                 in the total cost is less than delta. 
            beta:               Linesearch parameter in (0,1). Higher values lead to smaller
                                 linesearch steps. 
            gamma:              Linesearch parameter in [0,1). Higher values mean linesearch
                                 is performed more often in hopes of larger cost reductions.
        """
        assert system.IsDifferenceEquationSystem()[0],  "must be a discrete-time system"

        # float-type copy of the system and context for linesearch.
        # Computations using this system are fast but don't support gradients
        self.system = system
        self.context = self.system.CreateDefaultContext()
        self.input_port = self.system.get_input_port(input_port_index)

        # Autodiff copy of the system for computing dynamics gradients
        self.system_ad = system.ToAutoDiffXd()
        self.context_ad = self.system_ad.CreateDefaultContext()
        self.input_port_ad = self.system_ad.get_input_port(input_port_index)
       
        # Set some parameters
        self.N = num_timesteps
        self.delta = delta
        self.beta = beta
        self.gamma = gamma

        # Define state and input sizes
        self.n = self.context.get_discrete_state_vector().size()
        self.m = self.input_port.size()

        # Initial and target states
        self.x0 = np.zeros(self.n)
        self.x_xom = np.zeros(self.n)

        # Quadratic cost terms
        self.Q = np.eye(self.n)
        self.R = np.eye(self.m)
        self.Qf = np.eye(self.n)

        # Arrays to store best guess of control and state trajectory
        self.x_bar = np.zeros((self.n,self.N))
        self.u_bar = np.zeros((self.m,self.N-1))

        # Arrays to store dynamics gradients
        self.fx = np.zeros((self.n,self.n,self.N-1))
        self.fu = np.zeros((self.n,self.m,self.N-1))

        # Local feedback gains u = u_bar - eps*kappa_t - K_t*(x-x_bar)
        self.kappa = np.zeros((self.m,self.N-1))
        self.K = np.zeros((self.m,self.n,self.N-1))

        # Coefficents Qu'*Quu^{-1}*Qu for computing the expected 
        # reduction in cost dV = sum_t eps*(1-eps/2)*Qu'*Quu^{-1}*Qu
        self.dV_coeff = np.zeros(self.N-1)

        # -------------------------------- Derivatives interpolation  --------------------------------------

        self.saveFileStartIndex = 0

        self.fx_baseline = np.zeros((self.n,self.n,self.N-1))
        self.fu_baseline = np.zeros((self.n,self.m,self.N-1))

        self.deriv_calculated_at_index = [False] * self.N
        self.time_getDerivs = 0
        self.percentage_derivs = 0
        self.time_backwardsPass = 0
        self.time_fp = 0

        # Total number of columns dynamics gradients over the trajectory (trajec length * dof) (we group dof columns into triplets)
        self.total_num_columns_derivs = self.N * self.n

        # If no derivs_interpolation specified - use the baseline case (setInterval1 - computing derivatives at every time-step)
        if derivs_keypoint_methods is None:
            self.derivs_interpolation = [utils_derivs_interpolation.derivs_interpolation('setInterval', 1, 0, 0, 0)]
        else:
            self.derivs_interpolation = derivs_keypoint_methods

        self.initialCost = None
        self.save_trajecInfo = False
        self.taskName = "blankTask"
        self.saveIndex = 0

    def SetTaskName(self, taskName):
        self.taskName = taskName

    def SetInitialState(self, x0):
        """
        Fix the initial condition for the optimization.

        Args:
            x0: Vector containing the initial state of the system
        """
        self.x0 = x0

    def SetTargetState(self, x_nom):
        """
        Fix the target state that we're trying to drive the system to.

        Args:
            x_nom:  Vector containing the target system state
        """
        self.x_nom = np.asarray(x_nom).reshape((self.n,))

    def SetRunningCost(self, Q, R):
        """
        Set the quadratic running cost

            (x-x_nom)'Q(x-x_nom) + u'Ru

        Args:
            Q:  The (n,n) state penalty matrix
            R:  The (m,m) control penalty matrix
        """
        assert Q.shape == (self.n,self.n)
        assert R.shape == (self.m,self.m)

        self.Q = Q
        self.R = R

    def SetTerminalCost(self, Qf):
        """
        Set the terminal cost

            (x-x_nom)'Qf(x-x_nom)

        Args:
            Qf: The (n,n) final state penalty matrix
        """
        assert Qf.shape == (self.n, self.n)
        self.Qf = Qf
    
    def SetInitialGuess(self, u_guess):
        """
        Set the initial guess of control tape.

        Args:
            u_guess:    (m,N-1) numpy array containing u at each timestep
        """
        assert u_guess.shape == (self.m, self.N-1)
        self.u_bar = u_guess

    def SetControlLimits(self, u_min, u_max):
        pass

    def SetDerivativeInterpolator(self, derivs_interpolation):
        self.derivs_interpolation = derivs_interpolation

    def _running_cost_partials(self, x, u):
        """
        Return the partial derivatives of the (quadratic) running cost

            l = x'Qx + u'Ru

        for the given state and input values.

        Args:
            x:  numpy array representing state
            u:  numpy array representing control

        Returns:
            lx:     1st order partial w.r.t. x
            lu:     1st order partial w.r.t. u
            lxx:    2nd order partial w.r.t. x
            luu:    2nd order partial w.r.t. u
            lux:    2nd order partial w.r.t. u and x
        """
        lx = 2*self.Q@x - 2*self.x_nom.T@self.Q
        lu = 2*self.R@u
        lxx = 2*self.Q
        luu = 2*self.R
        lux = np.zeros((self.m,self.n))

        return (lx, lu, lxx, luu, lux)

    def _terminal_cost_partials(self, x):
        """
        Return the partial derivatives of the (quadratic) terminal cost

            lf = x'Qfx

        for the given state values. 

        Args:
            x: numpy array representing state

        Returns:
            lf_x:   gradient of terminal cost
            lf_xx:  hessian of terminal cost
        """
        lf_x = 2*self.Qf@x - 2*self.x_nom.T@self.Qf
        lf_xx = 2*self.Qf

        return (lf_x, lf_xx)
    
    def _calc_dynamics(self, x, u):
        """
        Given a system state (x) and a control input (u),
        compute the next state 

            x_next = f(x,u)

        Args:   
            x:  An (n,) numpy array representing the state
            u:  An (m,) numpy array representing the control input

        Returns:
            x_next: An (n,) numpy array representing the next state
        """
        # Set input and state variables in our stored model accordingly
        self.context.SetDiscreteState(x)
        self.input_port.FixValue(self.context, u)

        # Compute the forward dynamics x_next = f(x,u)
        state = self.context.get_discrete_state()
        self.system.CalcForcedDiscreteVariableUpdate(self.context, state)
        x_next = state.get_vector().value().flatten()

        return x_next

    def _calc_dynamics_partials(self, x, u):
        """
        Given a system state (x) and a control input (u),
        compute the first-order partial derivitives of the dynamics

            x_next = f(x,u)
            fx = partial f(x,u) / partial x
            fu = partial f(x,u) / partial u
        
        Args:   
            x:  An (n,) numpy array representing the state
            u:  An (m,) numpy array representing the control input

        Returns:
            fx:     A (n,n) numpy array representing the partial derivative 
                    of f with respect to x.
            fu:     A (n,m) numpy array representing the partial derivative 
                    of f with respect to u.
        """
        # Create autodiff versions of x and u
        xu = np.hstack([x,u])

        # If no specified columns, compute all of the colums of fx and fu
        xu_ad = InitializeAutoDiff(xu)
        
        x_ad = xu_ad[:self.n]
        u_ad = xu_ad[self.n:]

        # Set input and state variables in our stored model accordingly
        self.context_ad.SetDiscreteState(x_ad)
        self.input_port_ad.FixValue(self.context_ad, u_ad)

        # Compute the forward dynamics x_next = f(x,u)
        state = self.context_ad.get_discrete_state()
        self.system_ad.CalcForcedDiscreteVariableUpdate(self.context_ad, state)
        x_next = state.get_vector().CopyToVector()
       
        # Compute partial derivatives
        G = ExtractGradient(x_next)
        fx = G[:,:self.n]
        fu = G[:,self.n:]

        return (fx, fu)

    def _linesearch(self, L_last):
        """
        Determine a value of eps in (0,1] that results in a suitably
        reduced cost, based on forward simulations of the system. 

        This involves simulating the system according to the control law

            u = u_bar - eps*kappa - K*(x-x_bar).

        and reducing eps by a factor of beta until the improvement in
        total cost is greater than gamma*(expected cost reduction)

        Args:
            L_last: Total cost from the last iteration.

        Returns:
            eps:        Linesearch parameter
            x:          (n,N) numpy array of new states
            u:          (m,N-1) numpy array of new control inputs
            L:          Total cost/loss associated with the new trajectory
            n_iters:    Number of linesearch iterations taken

        Raises:
            RuntimeError: if eps has been reduced to <1e-8 and we still
                           haven't found a suitable parameter.
        """
        eps = 1.0
        n_iters = 0
        while eps >= 1e-8:
            n_iters += 1

            # Simulate system forward using the given eps value
            L = 0
            expected_improvement = 0
            x = np.zeros((self.n,self.N))
            u = np.zeros((self.m,self.N-1))

            x[:,0] = self.x0
            for t in range(0,self.N-1):
                u[:,t] = self.u_bar[:,t] - eps*self.kappa[:,t] - self.K[:,:,t]@(x[:,t] - self.x_bar[:,t])
                   
                try:
                    x[:,t+1] = self._calc_dynamics(x[:,t], u[:,t])
                except RuntimeError as e:
                    # If dynamics are infeasible, consider the loss to be infinite 
                    # and stop simulating. This will lead to a reduction in eps
                    print("Warning: encountered infeasible simulation in linesearch")
                    #print(e)
                    L = np.inf
                    break
                except:
                    print("delta > 0 error - probably")
                    L = np.inf
                    break

                L += (x[:,t]-self.x_nom).T@self.Q@(x[:,t]-self.x_nom) + u[:,t].T@self.R@u[:,t]
                expected_improvement += -eps*(1-eps/2)*self.dV_coeff[t]
            L += (x[:,-1]-self.x_nom).T@self.Qf@(x[:,-1]-self.x_nom)

            # Chech whether the improvement is sufficient
            improvement = L_last - L
            if improvement > self.gamma*expected_improvement:
                return eps, x, u, L, n_iters

            # Otherwise reduce eps by a factor of beta
            eps *= self.beta

        return 0, x, u, L, n_iters

        # raise RuntimeError("linesearch failed after %s iterations"%n_iters)
    
    def _forward_pass(self, L_last, interpolation_method):
        """
        Simulate the system forward in time using the local feedback
        control law

            u = u_bar - eps*kappa - K*(x-x_bar).

        Performs a linesearch on eps to (approximately) determine the 
        largest value in (0,1) that results in a reduced cost. 

        Args:
            L_last: Total loss from last iteration, used for linesearch

        Updates:
            u_bar:  The current best-guess of optimal u
            x_bar:  The current best-guess of optimal x
            fx:     Dynamics gradient w.r.t. x
            fu:     Dynamics gradient w.r.t. u

        Returns:
            L:          Total cost associated with this iteration
            eps:        Linesearch parameter used
            ls_iters:   Number of linesearch iterations
        """
        # Do linesearch to determine eps
        timeStart = time.time()
        eps, x, u, L, ls_iters = self._linesearch(L_last)
        timeEnd = time.time()
        self.time_fp = timeEnd - timeStart

        timeStart = time.time()
        self._get_derivatives(x, u, interpolation_method)
        timeEnd = time.time()
        self.time_getDerivs = timeEnd - timeStart

        # Update stored values
        self.u_bar = u
        self.x_bar = x

        return L, eps, ls_iters

    def _get_derivatives(self, x, u, interpolation_method):
        """
        Calculates the derivatives fx and fu over the entire trajectory. Depending on 
        the keypoint method specified, this function will calculate a set of keypoints.
        At these keypoints the derivatives will be calculated exactly using autodiff.
        In-between these keypoints this function will linearly interpolate approximations
        to the derivatives between the computed values. This is done to reduce the amount
        of time spent per iteration computing derivatives.

        Updates:
            fx:      dynamcis partial wrt state at each timestep
            fu:      dynamcis partial wrt control at each timestep

        """

        DEBUG = False

        # Calculate keypoints over the trajectory
        keyPoints = []
        if(interpolation_method.keypoint_method == 'setInterval'):
            keyPoints = self.get_keypoints_set_interval(interpolation_method)
        elif(interpolation_method.keypoint_method == 'adaptiveJerk'):
            keyPoints = self.get_keypoints_adaptive_jerk(x, u, interpolation_method)
        elif(interpolation_method.keypoint_method == 'iterativeError'):
            keyPoints = self.get_keypoints_iterative_error(x, u, interpolation_method)
            self.deriv_calculated_at_index = [False] * self.N
        elif(interpolation_method.keypoint_method == 'magvelChange'):
            keyPoints = self.get_keypoints_magvel_change(x, u, interpolation_method)
        else:
            raise Exception('unknown interpolation method')

        self.percentage_derivs = (len(keyPoints) / (self.N - 1)) * 100

        # Calculate derivatives at keypoints (iterative error method will have already done this)
        if interpolation_method.keypoint_method != 'iterativeError':
            for t in range(len(keyPoints)):
                self.fx[:,:,keyPoints[t]], self.fu[:,:,keyPoints[t]] = self._calc_dynamics_partials(x[:,keyPoints[t]], u[:,keyPoints[t]])


        # Interpolate derivatives if required (Interpolation not needed in baseline case (setInterval1))
        if not (interpolation_method.keypoint_method == 'setInterval' and interpolation_method.minN == 1):
            self.interpolate_derivatives(keyPoints)


        if(DEBUG):
            for t in range(self.N-1):
                self.fx_baseline[:,:,t], self.fu_baseline[:,:,t] = self._calc_dynamics_partials(x[:,t], u[:,t])

            indexX = self.n - 1
            indexY = self.n - 1
            
            plt.plot(self.fx_baseline[indexX,indexY,:], label='fx_baseline')
            plt.plot(self.fx[indexX,indexY,:], label='fx')
            plt.show()

            error = self._error_in_trajectory()
            print("Error in trajectory: ", error)
                


    def get_keypoints_set_interval(self, interpolation_method):
        """
        Computes keypoints over the trajectory at set intervals as specified by the
        interpolation method

        Returns:
            keypoints:  list of keypoints to compute dynamics gradients at via autodiff
        """

        keypoints = []

        keypoints = np.arange(0,self.N-1, interpolation_method.minN).astype(int)
        if keypoints[-1] != self.N-2:
            keypoints[-1] = self.N-2

        return keypoints

    def get_keypoints_adaptive_jerk(self, x, u, interpolation_method):
        """
        Computes keypoints over the trajectory adaptively by looking at the jerk
        profile over the trajectory and changing the sample rate based on the jerk

        Returns:
            keypoints:  list of keypoints to compute dynamics gradients at via autodiff
        """
        keypoints = []

        dof = int(self.n/2)

        jerk_profile = self.calc_jerk_profile(x)
        counter = 0
        keypoints.append(0)

        for t in range(len(jerk_profile)):
            counter += 1

            if counter >= interpolation_method.minN:
                for i in range(dof):
                    if jerk_profile[t, i] > interpolation_method.jerk_threshold:
                        keypoints.append(t)
                        counter = 0
                        break
            
            if counter >= interpolation_method.maxN:
                keypoints.append(t)
                counter = 0

            
        if keypoints[-1] != self.N-2:
            keypoints[-1] = self.N-2
                        
        return keypoints

    def calc_jerk_profile(self, x):
        """
        Calculates the jerk profile (derivative of acceleration) for each
        degree of freedom over the trajectory

        Returns:    jerk_profile:   jerk profile over the trajectory for each dof
        """
        dof = int(self.n/2)
        jerk = np.zeros((self.N-3, dof))
        for i in range(dof):
            for t in range(self.N-3):
                acell1 = x[i + dof,t+2] - x[i + dof, t+1]
                acell2 = x[i + dof,t+1] - x[i + dof, t]

                jerk[t, i] = acell1 - acell2
                
        return jerk

    def calc_vel_profile(self, x):
        """
        Calculates the velocity profile for each
        degree of freedom over the trajectory

        Returns:    vel_profile:   velocity profile over the trajectory for each dof
        """
        dof = int(self.n/2)
        vel_profile = np.zeros((self.N-1, dof))

        for i in range(dof):
            for t in range(self.N-1):
                vel_profile[t, i] = x[i + dof, t]
                
        return vel_profile

    
    def get_keypoints_magvel_change(self, x, u, interpolation_method):
        """
        Calculates keypoints at which to calcualte derivatives by tracking the change
        in velocities for different dofs in the system. If the change goes above some threshold
        then a keypoint is added.

        Updates:
            fx:      at certain timesteps
            fu:      at certain timesteps
        Returns:
            keypoints:  list of keypoints to compute dynamics gradients at via autodiff
        """
        keypoints = []

        dof = int(self.n/2)
        velProfile = self.calc_vel_profile(x)

        lastVelCounter = np.zeros((dof))
        lastVelDirection = np.zeros((dof))
        counter = 0

        for i in range(dof):
            lastVelCounter[i] = velProfile[0, i]

        keypoints.append(0)

        

        for t in range(len(velProfile)):
            counter += 1
            if counter >= interpolation_method.minN:
                for i in range(dof):
                    currentVelDirection = velProfile[t, i] - velProfile[t-1, i]
                    currentVelChange = velProfile[t, i] - lastVelCounter[i]

                    # if(currentVelDirection * lastVelDirection[i] < 0):
                    #     counter = 0
                    #     keypoints.append(t)
                    #     for i in range(dof):
                    #         lastVelCounter[i] = velProfile[t, i]
                    #         lastVelDirection[i] = velProfile[t, i] - velProfile[t-1, i]

                    #     break

                    if(currentVelChange > interpolation_method.velChange_threshold):
                        counter = 0
                        keypoints.append(t)
                        for i in range(dof):
                            lastVelCounter[i] = velProfile[t, i]
                            lastVelDirection[i] = velProfile[t, i] - velProfile[t-1, i]

                        break
            
            if counter >= interpolation_method.maxN:
                keypoints.append(t)
                counter = 0

                for i in range(dof):
                    lastVelCounter[i] = velProfile[t, i]
                    lastVelDirection[i] = velProfile[t, i] - velProfile[t-1, i]

                break

            
        if keypoints[-1] != self.N-2:
            keypoints[-1] = self.N-2

        return keypoints

    def get_keypoints_iterative_error(self, x, u, interpolation_method):
        """
        Calculates keypoints at which to calcualte derivatives by checking
        middle of the inteprolation versus the real value. If the approximation 
        is valid, no further subdivisions are required. If the approximation is 
        bad, then further subdivisions are required.

        Updates:
            fx:      at certain timesteps
            fu:      at certain timesteps
        Returns:
            keypoints:  list of keypoints to compute dynamics gradients at via autodiff
        """
        keypoints = []
        binsComplete = False 
        
        start_index = 0
        end_index = self.N-2

        initial_index_tuple = utils_derivs_interpolation.index_tuple(start_index, end_index)
        list_indices_to_check = [initial_index_tuple]
        sub_list_with_midpoints = []

        while not binsComplete:
            sub_list_indices = []
            all_checks_passed = True
            for i in range(len(list_indices_to_check)):

                approximation_good = self.check_one_matrix_error(list_indices_to_check[i], x, u, interpolation_method)
                mid_index = int((list_indices_to_check[i].start_index + list_indices_to_check[i].end_index)/2)

                if not approximation_good:
                    sub_list_indices.append(utils_derivs_interpolation.index_tuple(list_indices_to_check[i].start_index, mid_index))
                    sub_list_indices.append(utils_derivs_interpolation.index_tuple(mid_index, list_indices_to_check[i].end_index))
                    all_checks_passed = False

                else:
                    sub_list_with_midpoints.append(list_indices_to_check[i].start_index)
                    sub_list_with_midpoints.append(mid_index)
                    sub_list_with_midpoints.append(list_indices_to_check[i].end_index)

            if(all_checks_passed):
                binsComplete = True

            list_indices_to_check = sub_list_indices
            sub_list_indices = []

        for i in range(self.N-1):
            if(self.deriv_calculated_at_index[i]):
                keypoints.append(i)

        return keypoints

    def check_one_matrix_error(self, indices, x, u, interpolation_method):
        """
        Checks the mean sqaured sum error of two dynamics partials matrices
        If the error is above the set threshold, the approximation is bad 
        and false is returend. This leads to further subdivisions in the iterative
        error method.

        Updates:
            fx at certain timesteps
            fu at certain timesteps

        Returns:
            approximation_good:     boolean indicating if the approximation is good
        """
        approximation_good = True

        if(indices.end_index - indices.start_index <= interpolation_method.minN):
            return approximation_good

        start_index = indices.start_index
        mid_index = int((indices.start_index + indices.end_index)/2)
        end_index = indices.end_index

        if(not self.deriv_calculated_at_index[start_index]):
            # Calculate the graident matrices at this index
            self.fx[:,:,start_index], self.fu[:,:,start_index] = self._calc_dynamics_partials(x[:,start_index], u[:,start_index])
            self.deriv_calculated_at_index[start_index] = True

        if(not self.deriv_calculated_at_index[mid_index]):
            # Calculate the graident matrices at this index
            self.fx[:,:,mid_index], self.fu[:,:,mid_index] = self._calc_dynamics_partials(x[:,mid_index], u[:,mid_index])
            self.deriv_calculated_at_index[mid_index] = True

        if(not self.deriv_calculated_at_index[end_index]):
            # Calculate the graident matrices at this index
            self.fx[:,:,end_index], self.fu[:,:,end_index] = self._calc_dynamics_partials(x[:,end_index], u[:,end_index])
            self.deriv_calculated_at_index[end_index] = True

        #calculate mid index via interpolation
        fx_mid_lin = (self.fx[:,:,end_index] + self.fx[:,:,start_index] ) / 2


        sumSqDiff = 0
        for i in range(self.n):
            for j in range(self.n):
                sumSqDiff += (fx_mid_lin[i,j] - self.fx[i,j,mid_index])**2

        average_sq_diff = sumSqDiff / (2 * self.n)

        if(average_sq_diff > interpolation_method.iterative_error_threshold):
            approximation_good = False

        return approximation_good


    def interpolate_derivatives(self, keyPoints):
        """
        Interpolate the dynamics partials (fx, fu) by linealry
        interpolating the calculated values at the set keypoints.

        Updates:
            fx:      dynamcis partial wrt state at each timestep
            fu:      dynamcis partial wrt control at each timestep
        """

        # Interpoalte whole matrices
        for i in range(len(keyPoints) - 1):
            startIndex = keyPoints[i]
            endIndex = keyPoints[i+1]

            startVals_fx = self.fx[:,:,startIndex]
            endVals_fx = self.fx[:,:,endIndex]
            startVals_fu = self.fu[:,:,startIndex]
            endVals_fu = self.fu[:,:,endIndex]

            diff_fx = endVals_fx - startVals_fx
            diff_fu = endVals_fu - startVals_fu

            for j in range(startIndex, endIndex):
                self.fx[:,:,j] = startVals_fx + (endVals_fx - startVals_fx) * (j - startIndex) / (endIndex - startIndex)
                self.fu[:,:,j] = startVals_fu + (endVals_fu - startVals_fu) * (j - startIndex) / (endIndex - startIndex)

    def _error_in_trajectory(self):
        error = 0.0

        for t in range(self.N - 1):
            error += self._one_matrix_error(self.fx[:,:,t], self.fx_baseline[:,:,t])


        return error

    def _one_matrix_error(self, matrix1, matrix2):
        error = 0.0

        for i in range(self.n):
            for j in range(self.n):
                error += (matrix1[i,j] - matrix2[i,j])**2

        error /= (self.n * self.n)


        return error
    
    def _backward_pass(self):
        """
        Compute a quadratic approximation of the optimal cost-to-go
        by simulating the system backward in time. Use this quadratic 
        approximation and a first-order approximation of the system 
        dynamics to compute the feedback controller

            u = u_bar - eps*kappa - K*(x-x_bar).

        Updates:
            kappa:      feedforward control term at each timestep
            K:          feedback control term at each timestep
            dV_coeff:   coefficients for expected change in cost
        """
        # Store gradient and hessian of cost-to-go
        Vx, Vxx = self._terminal_cost_partials(self.x_bar[:,-1])

        # Do the backwards sweep
        for t in np.arange(self.N-2,-1,-1):
            x = self.x_bar[:,t]
            u = self.u_bar[:,t]

            # Get second(/first) order approximation of cost(/dynamics)
            lx, lu, lxx, luu, lux = self._running_cost_partials(x,u)
            fx = self.fx[:,:,t]
            fu = self.fu[:,:,t]

            # Construct second-order approximation of cost-to-go
            Qx = lx + fx.T@Vx
            Qu = lu + fu.T@Vx
            Qxx = lxx + fx.T@Vxx@fx
            Quu = luu + fu.T@Vxx@fu
            Quu_inv = np.linalg.inv(Quu)
            Qux = lux + fu.T@Vxx@fx

            # Derive controller parameters
            self.kappa[:,t] = Quu_inv@Qu
            self.K[:,:,t] = Quu_inv@Qux

            # Derive cost reduction parameters
            self.dV_coeff[t] = Qu.T@Quu_inv@Qu

            # Update gradient and hessian of cost-to-go
            Vx = Qx - Qu.T@Quu_inv@Qux
            Vxx = Qxx - Qux.T@Quu_inv@Qux

    def Solve(self):
        """
        Solve the optimization problem and return the (locally) optimal
        state and input trajectories. 

        Return:
            x:              (n,N) numpy array containing optimal state trajectory
            u:              (m,N-1) numpy array containing optimal control tape
            solve_time:     Total solve time in seconds
            optimal_cost:   Total cost associated with the (locally) optimal solution
        """
        # Store total cost and improvement in cost
        L = np.inf
        improvement = np.inf

        # Print labels for debug info
        print("----------------------------------------------------------------------------------------------------------------------------------")
        print("|    iter    |    cost    |    eps    |    ls    | derivs time | derivs '%'  | bp time  | fp time  |   iter time    |    time    |")
        print("----------------------------------------------------------------------------------------------------------------------------------")

        # iteration counter
        i = 0
        print(self.derivs_interpolation)
        print(f' number of interpolation methods: {len(self.derivs_interpolation)}')

        outer_loop_derivs_method_counter = len(self.derivs_interpolation) - 1

        st = time.time()
        percentage_derivs = []

        while outer_loop_derivs_method_counter >= 0:
            print(f' before interpolation method: {outer_loop_derivs_method_counter}')
            L, _percentage_derivs, i = self.inner_loop_optimisation(self.derivs_interpolation[outer_loop_derivs_method_counter], L, improvement, i, st)
            percentage_derivs += _percentage_derivs

            # Go to a more accurate approximation
            outer_loop_derivs_method_counter -= 1

        cost_reduction = 1 - (L / self.initialCost)
        total_time = time.time() - st
        print(percentage_derivs)

        avg_percent_derivs = sum(percentage_derivs) / len(percentage_derivs)

        return self.x_bar, self.u_bar, total_time, L, cost_reduction, i, avg_percent_derivs

    def inner_loop_optimisation(self, interpolation_method, L, improvement, iteration_num, st):
        percent_derivs = []
        while improvement > self.delta:
            st_iter = time.time()

            L_new, eps, ls_iters = self._forward_pass(L, interpolation_method)
            if(self.initialCost == None):
                self.initialCost = L_new

            if self.save_trajecInfo:
                self.saveTrajecInfo(self.taskName, self.saveIndex, self.fx, self.fu, self.x_bar, self.u_bar)

            bp_time_start = time.time()
            self._backward_pass()
            bp_end_time = time.time()
            self.time_backwardsPass = bp_end_time - bp_time_start

            iter_time = time.time() - st_iter
            total_time = time.time() - st

            print(f"{iteration_num:^14}{L_new:11.4f}  {eps:^12.4f}{ls_iters:^11}   {self.time_getDerivs:1.5f}         {self.percentage_derivs:.1f}       {self.time_backwardsPass:1.5f}    {self.time_fp:1.5f}      {iter_time:1.5f}          {total_time:4.2f}")
            percent_derivs.append(self.percentage_derivs)

            improvement = L - L_new
            L = L_new
            iteration_num += 1
            self.saveIndex += 1

        return L, percent_derivs, iteration_num

    def SaveSolution(self, fname):
        """
        Save the stored solution, including target state x_bar
        nominal control input u_bar, feedback gains K, and timesteps
        t in the given file, where the feedback control

            u = u_bar - K*(x-x_bar)

        locally stabilizes the nominal trajectory.

        Args:
            fname:  npz file to save the data to.
        """
        dt = self.system.GetSubsystemByName("plant").time_step()
        T = (self.N-1)*dt
        t = np.arange(0,T,dt)

        x_bar = self.x_bar[:,:-1]  # remove last timestep
        u_bar = self.u_bar
        K = self.K

        np.savez(fname, t=t, x_bar=x_bar, u_bar=u_bar, K=K)

    def saveTrajecInfo(self, task, index, A, B, x, u):
        """
        Save the trajectory information for a given 
        """
        save_index = index + self.saveFileStartIndex

        base_folder_name = f"savedTrajecInfo/{task}/{save_index}"
        print(f"Saving trajectory information to {base_folder_name}")

        if(not os.path.exists(base_folder_name)):
            os.makedirs(base_folder_name)

        trajecLength = A.shape[2]

        # reshape matrices
        # drop the top half of the matrices
        topHalfIndex = int(ceil(self.n/2))
        A = A[topHalfIndex:,:,:]
        B = B[topHalfIndex:,:,:]
        # reshape to 1D
        A = A.reshape(-1, A.shape[-1])
        B = B.reshape(-1, B.shape[-1])

        A = A.transpose()
        B = B.transpose()
        x = x.transpose()
        u = u.transpose()

        custom_delimiter = ','

        # Save A matrices
        file_name = base_folder_name + '/A_matrices.csv'

        formatted_data = "\n".join([",".join(map(str, row)) + "," for row in A])

        # Write the formatted data to the CSV file
        with open(file_name, 'w') as file:
            file.write(formatted_data)

        # Save B matrices
        file_name = base_folder_name + '/B_matrices.csv'

        formatted_data = "\n".join([",".join(map(str, row)) + "," for row in B])

        # Write the formatted data to the CSV file
        with open(file_name, 'w') as file:
            file.write(formatted_data)

        # Save controls
        file_name = base_folder_name + '/controls.csv'

        formatted_data = "\n".join([",".join(map(str, row)) + "," for row in u])

        # Write the formatted data to the CSV file
        with open(file_name, 'w') as file:
            file.write(formatted_data)

        # Save states
        file_name = base_folder_name + '/states.csv'

        formatted_data = "\n".join([",".join(map(str, row)) + "," for row in x])

        # Write the formatted data to the CSV file
        with open(file_name, 'w') as file:
            file.write(formatted_data)