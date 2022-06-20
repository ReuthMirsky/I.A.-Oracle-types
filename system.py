import random
import dynet as dy
import numpy as np
import math


def softmax(l):
    soft = l.copy()
    expl = np.exp(l)
    for i in range(len(l)):
        soft[i] = expl[i] / np.sum(expl)
    return soft


class Transition:
    def __init__(self, name, start, end):
        self.name = name
        self.start = start
        self.end = end



class process:
    # finite automaton where transitions can be shared with other processes.
    def __init__(self, name, states=[], internal=[], shared=[], initial_state=None, update_states=[]):
        self.name = name
        self.states = states  # list of names of the states.
        self.initial_state = initial_state
        self.current_state = initial_state
        self.internal = internal  # transitions that are specific to the process.
        self.shared = shared  # transitions of both processes, initialized empty and filled by add_transition.
        self.internal_transitions = []  # same as above, but the names of the transitions.
        self.shared_transitions = []  # initialized as empty lists that will be filled using list_transitions.
        self.all_transitions = []  # all_transitions will be updated as the concatenation of the two previous lists.
        self.update_states = update_states  # only states on which a learning pass can be done. empty = all states.
        self.pr_dict={}
    def add_state(self, name):
        self.states.append(name)
        if self.current_state is None:
            self.initial_state = name
            self.current_state = name
            
    def define_update_state(self, name):
        if name not in self.states:
            raise ValueError("no state named", name)
        else:
            self.update_states.append(name)  # new update possible state, if it is a valid state of the process.
            
    def list_update_states(self, name_list):
        for name in name_list:
            self.define_update_state(name)  # define a list of update possible states.
            
    def reinitialize(self):
        self.current_state = self.initial_state  # return to initial state.
        if self.update_states is []:  # if list of update states is undefined, all states will be updated.
            self.list_update_states(self.states)
            
    def add_transition(self, name, start, end, internal=True):
        if internal:
            self.internal.append(Transition(name, start, end))  # add a new transition from state start to state end.
        else:
            self.shared.append(Transition(name, start, end))

    def trigger_transition(self, tr_name):
        try:  # move the process according to the transition tr_name, printing an error if not possible.
            self.current_state = next(tr.end for tr in self.internal if tr_name == tr.name and tr.start == self.current_state)
        except StopIteration:
            try:  # we trigger it in internal, but if not exist, in shared.
                self.current_state = next(tr.end for tr in self.shared if tr_name == tr.name and tr.start == self.current_state)
            except StopIteration:
                print("No transition named", tr_name, "from state", self.current_state)
      
    def list_transitions(self):
        for tr in self.internal:
            if tr.name not in self.internal_transitions:  # update the names of currently defined transitions.
                self.internal_transitions.append(tr.name)
        for tr in self.shared:
            if tr.name not in self.shared_transitions:
                self.shared_transitions.append(tr.name)
        self.all_transitions = self.internal_transitions + self.shared_transitions
        
    def available_transitions(self):
        available = []
        for tr in self.internal + self.shared:
            if tr.name not in available and tr.start == self.current_state:
                available.append(tr.name)
        return available  # returns a list of names of transitions that can be triggered in the current state.


    #gets a state and return a list of names of transitions that can be triggered in the current state
    def available_trans(self,state):
        available = []
        for tr in self.internal + self.shared:
            if tr.name not in available and tr.start == state:
                available.append(tr.name)
        return available  # returns a list of names of transitions that can be triggered in the current state.

    #gets a state and action and return the probability for this action
    def prob(self, state, trans):
        available = self.available_trans(state)
        if (trans in available):
            p = 1. / len(available)
        else:
            p = 0
        return p
    def prob_byRnn(self, state, trans):
        available = self.all_transitions
        idx = [i for i, s in enumerate(available) if trans in s]
        return self.pr_dict[state][idx[0]]

    #gets a state, action and oracle and return the probablity for this state prob(system=p_trans|oracle=o_trans)
    def conditional_prob(self, state, p_trans,o_trans):
        available = self.available_trans(state)

        if (p_trans in available and o_trans in available):
            if (p_trans== o_trans):
                p = 1
            else:
                p=0
        elif (p_trans in available and o_trans not in available):
            p = 1./len(available)
        elif p_trans not in available:
            p=0
        return p

    ######################################### H query uniform   #########################


    #gets a atate and oracle and returns H(oracle) for the given state
    #H(oracle=o_trans)=1/len(actions)*[sigma (p(system=t|oracle=o_trans)*log(p(system=t|oracle=o_trans))] for all action t
    def H_with_query_uniform(self,state,o_trans):
        sum=0
        available = self.all_transitions
        for action in available:
            p=self.conditional_prob(state,action,o_trans)
            if (p!=0):
                sum+=p*math.log2(p)
            else:
                sum+=p
        if (sum!=0):
            sum=-sum
        return (1./len(available))*sum

    # gets a state and return a list of all IG (for all actions), for example if the actions are a,b,c so for state g1 returns list=[H(a),H(b),H(c)]
    def H_query_for_state_uniform(self, state):
        list = []
        trans = self.all_transitions
        for i in trans:
            t = self.H_with_query_uniform(state, i)  # H(i)
            list.append(t)
        return list

    # returns a dictionary. Each key is a state ands its values are a list of H for all the actions
    def H_query_for_all_states_uniform(self):
        list = []
        states = sorted(self.states)
        d = dict.fromkeys(states, [])
        for i in d:
            p = self.H_query_for_state_uniform(i)
            d[i] = p
        return d

    ######################################### H query with normal rnn #########################
    def H_query_for_all_states_normal(self):
        list = []
        states = sorted(self.states)
        d = dict.fromkeys(states, [])
        for i in d:
            p = self.H_query_for_state_normal(i)
            d[i] = p
        return d

    def H_query_for_state_normal(self, state):
        list=[]
        trans = self.all_transitions
        for i in trans:
            t=self.H_with_queryO_normal(state,i) #H(i)
            list.append(t)
        return list

    def H_with_queryO_normal(self,state,o_trans):
        p=0
        result=0
        available = self.all_transitions
        actions=self.available_trans(state)
        probs=(self.pr_dict)[state]
        sum = 0
        for action in actions:
            idx = [i for i, s in enumerate(available) if action in s]
            sum += probs[idx[0]]
        idx = [i for i, s in enumerate(available) if o_trans in s]
        if(o_trans in actions):
            p = probs[idx[0]] / sum
            for action in available:
                cond_p=self.conditional_prob(state,action,o_trans)
                if (cond_p!=0):
                    result+=cond_p*math.log2(cond_p)
        if (result!=0):
            result=-result
        return result*p



    ########################### H not querying uniform ######################################################

    #gets a state and return H(not quering)=sigma(psystem=t) * log(p(system=t) for all action t
    def H_not_query_uniform(self, state):
        sum = 0
        available = self.all_transitions
        for action in available:
            p = self.prob(state, action)
            if (p != 0):
                sum += p * math.log2(p)
        if sum!=0:
            sum=-sum
        return sum

    # Return a dictionary.Each key is a state and its value is a value H(not query) for the same state
    def H_all_states_not_query_uniform(self):
        list = []
        states = sorted(self.states)
        d = dict.fromkeys(states, [])
        for i in d:
            h = self.H_not_query_uniform(i)
            d[i] = h
        return d
    ################################## H not querying rnn #################################
    def H_not_query_byRnn(self, state):
        sum = 0
        available = self.all_transitions
        for action in available:
            p = self.prob_byRnn(state, action)
            if (p != 0):
                sum += p * math.log2(p)
        if sum!=0:
            sum=-sum
        return sum

    def H_all_states_not_query_byRnn(self):
        list = []
        states = sorted(self.states)
        d = dict.fromkeys(states, [])
        for i in d:
            h = self.H_not_query_byRnn(i)
            d[i] = h
        return d
    ##########################################################################
    def Ig_final_not_trained(self):
        H_q=self.H_query_for_all_states_uniform()
        for key, values in H_q.items():
            H_q[key] = sum(values)
        h=self.H_all_states_not_query_uniform()
        Ig=H_q.copy()
        for key, values in H_q.items():
            Ig[key] =h[key]-H_q[key]
        return Ig,h,H_q

    def Ig_final_trained_BothwithRnn(self):
        H_q=self.H_query_for_all_states_normal()
        for key, values in H_q.items():
            H_q[key] = sum(values)
        h=self.H_all_states_not_query_byRnn()
        Ig = H_q.copy()
        for key, values in H_q.items():
            Ig[key] =h[key]-H_q[key]
        return Ig,h,H_q





class System:
    def __init__(self, name, processes):  # a system is a compound of processes.
        self.name = name
        self.processes = processes  # a list of the processes in the compound.
        self.shared_transitions = []  # list of the shared transitions of the different processes
        self.networks = None  # Neural networks associated to the processes (listed in the same order as the processes)
        self.R = None  # R, bias, parameters and trainer are respective lists of the parameters for the NNs
        self.bias = None
        self.parameters = None
        self.trainer = None
        
    def reinitialize(self):
        for pr in self.processes:  # reinitialize all processes in the system
            pr.reinitialize()
    
    def get_process(self, name):  # returns the process with that name
        return next(proc for proc in self.processes if name == proc.name)
    
    def add_process(self, process):
        self.processes.append(process)  # add a new process to the system
        
    def add_transition(self, name, pr_list, start_list, end_list):
        if len(pr_list) == 1:   # add a new transition shared between processes in pr_list
            is_internal = True  # for process pr_list[i], the transition goes from state start_list[i] to state end_list[i]
        else:
            is_internal = False
            self.shared_transitions.append((name, pr_list, start_list, end_list))  # parameters aren't changed.
        for i in range(len(pr_list)):
            start = start_list[i]
            end = end_list[i]
            for j in range(len(start)):
                self.get_process(pr_list[i]).add_transition(name, start[j], end[j], is_internal)


class plant_environment(System):
    def __init__(self, name, plant: process, environment: process, model="correctly_guess", layers=1, hidden_dim=5, oracle={}):
        self.plant = plant  # define the system with processes plant and environment
        self.environment = environment
        System.__init__(self, name, [self.plant, self.environment])
        self.plant.list_transitions()
        self.layers = layers  # parameters of the neural network that will be used by the plant
        self.hidden_dim = hidden_dim
        self.model = model
        self.oracle = oracle

    def create_RNN(self):
        self.plant.list_transitions()
        self.parameters = dy.ParameterCollection()
        input_dim = (len(self.plant.internal_transitions)+len(self.plant.shared_transitions))*len(self.plant.states)
        output_dim = len(self.plant.all_transitions)
        self.R = self.parameters.add_parameters((output_dim,self.hidden_dim))
        self.bias = self.parameters.add_parameters((output_dim))
        self.network = dy.VanillaLSTMBuilder(self.layers,input_dim,self.hidden_dim,self.parameters,forget_bias = 1.0)
        self.trainer = dy.SimpleSGDTrainer(self.parameters)
    
    def RNN_input(self, last_transition):
        v = [0]*((len(self.plant.internal_transitions)+len(self.plant.shared_transitions))*len(self.plant.states))
        i = next(i for i in range(len(self.plant.states)) if self.plant.states[i] == self.plant.current_state)
        if last_transition is not None:
            if last_transition[:4] == "fail":
                failed_action = ""
                current_char_index = 5
                while last_transition[current_char_index] != ")":
                        failed_action += last_transition[current_char_index]
                        current_char_index += 1
                j = next(j for j in range(len(self.plant.shared_transitions)) if failed_action == self.plant.shared_transitions[j])
                v[((len(self.plant.internal_transitions) + j))*len(self.plant.states)+i] = -1    
            else:
                j = next(j for j in range(len(self.plant.all_transitions)) if last_transition == self.plant.all_transitions[j])
                v[(len(self.plant.internal_transitions) + j)*len(self.plant.states)+i] = 1
        return v       

    def RNN_output(self, output):
        available = self.plant.available_transitions()
        if len(available) > 1:
            next_transition = random.choices(available,
                                         ([output[i] for i,tr in enumerate(self.plant.all_transitions) if tr in available]))[0]
        else:
            next_transition = available[0]
            available = self.plant.available_transitions()
        return next_transition

    def check_transition(self, plant_transition):
        available = self.environment.available_transitions()
        if plant_transition in available:
            return [plant_transition, plant_transition]
        else:
            return ["fail("+plant_transition+")", random.choice(available)]

    def trigger_transition(self,transition):
        if transition[0][:4] != "fail":
            self.plant.trigger_transition(transition[0])
        self.environment.trigger_transition(transition[1])

    def random_transition(self):
        plant_action = random.choice([tr for tr in self.plant.internal + self.plant.shared if tr.start == self.plant.current_state]).name
        environment_action = random.choice([tr for tr in self.environment.internal + self.environment.shared if tr.start == self.environment.current_state]).name
        return [plant_action,environment_action]
    #################################################
    def check_transition1(self, transition):
        available1 = self.environment.available_transitions()
        available2 = self.plant.available_transitions()
        if transition in available1 and transition in available2 :
            return [transition, transition]
        elif transition in available2:
            return ["fail("+transition+")", random.choice(available1)]
        else:
            return ["fail(" + random.choice(available2) + ")", transition]

    def trigger_transition1(self,transition):
        if transition[0][:4] != "fail":
            self.plant.trigger_transition(transition[0])
        self.environment.trigger_transition(transition[1])

    def random_transition1(self):
        plant_action = random.choice([tr for tr in self.plant.internal + self.plant.shared if tr.start == self.plant.current_state]).name
        environment_action = random.choice([tr for tr in self.environment.internal + self.environment.shared if tr.start == self.environment.current_state]).name
        return [plant_action,environment_action]
    #################################################
    def generate_random_execution(self, steps):
        execution = []
        for s in range(steps):
            tr = self.random_transition()
            tr = self.check_transition(tr[0])
            self.trigger_transition(tr)
            execution.append(tr)
        return execution

    def generate_controlled_execution(self, steps,use_oracle=False):
        execution = []
        dy.renew_cg()
        state = self.network.initial_state()
        last_transition = None
        for step in range(steps):
            network_input = self.RNN_input(last_transition)
            input_vector = dy.inputVector(network_input)
            state = state.add_input(input_vector)
            output = dy.softmax(self.R * state.output() + self.bias).value()
            if(use_oracle):
                oracle_state = self.plant.current_state + self.environment.current_state
                if oracle_state in self.oracle:
                    if (len(self.oracle[oracle_state]) > 1):
                        next_plant_action = random.choice(self.oracle[oracle_state])
                    else:
                        next_plant_action = self.oracle[oracle_state]
                else:
                    next_plant_action = self.RNN_output(output)
            else:
                next_plant_action = self.RNN_output(output)
            tr = self.check_transition1(next_plant_action)
            self.trigger_transition(tr)
            execution.append(tr)
            last_transition = tr[0]
        return execution

    def generate_training_execution(self, steps=50, lookahead=1,compare_loss = False, epsilon=0,use_oracle=False):
        rollout = [None] * (lookahead + 1)
        rollout_error = [None] * (lookahead + 1)

        def rollout_update(rollout, new_state):
            return rollout[1:]+[new_state]

        def rollout_error_update(rollout_errors, error):
            return rollout_errors[1:]+[error]

        def get_rollouts(tr, rollout, rollout_error):
            plant_tr = tr[0]
            if plant_tr[:4] == "fail":
                plant_tr = plant_tr[5]
                rollout_error = rollout_error_update(rollout_error, True)
            else:
                rollout_error = rollout_error_update(rollout_error, False)
            # store information about the output to compute the loss

            rollout = rollout_update(rollout, (output, self.plant.available_transitions(), plant_tr))
            i_train = next(i for i in range(len(rollout)) if rollout[i] is not None)
            return rollout, rollout_error, i_train

        def get_loss(p: process, rollout, rollout_error, i_train, loss):
            if rollout[i_train] is not None:
                nb_failures = rollout_error.count(True)  # count successes and failures in lookahead window
                nb_successes = 1 + lookahead - nb_failures
                for i in range(len(self.plant.all_transitions)):
                    if self.plant.all_transitions[i] in rollout[i_train][1]:
                        if self.plant.all_transitions[i] == rollout[i_train][2]:  # chosen action
                            loss.append((nb_successes/(lookahead+1))*dy.pickneglogsoftmax(rollout[i_train][0],i))
                        else:  # not chosen action
                            loss.append((nb_failures/(lookahead+1))*dy.pickneglogsoftmax(rollout[i_train][0],i))
            return loss

        execution = []
        last_transition = None
        dy.renew_cg()
        state = self.network.initial_state()
        loss = [dy.scalarInput(0)]
        for step in range(steps):
            network_input = self.RNN_input(last_transition)
            input_vector = dy.inputVector(network_input)
            state = state.add_input(input_vector)
            output = dy.softmax(self.R*state.output() + self.bias)
            output_value = output.value()
            if(use_oracle):
                oracle_state = self.plant.current_state + self.environment.current_state
                if oracle_state in self.oracle:
                    #next_plant_action = self.oracle[oracle_state]

                    if (len(self.oracle[oracle_state]) > 1):
                        next_plant_action = random.choice(self.oracle[oracle_state])
                    else:
                        next_plant_action = self.oracle[oracle_state]

                elif random.random() < epsilon:
                        next_plant_action = self.random_transition()[0]
                else:
                    next_plant_action = self.RNN_output(output_value)
            else:
                next_plant_action = self.RNN_output(output_value)
            tr = self.check_transition1(next_plant_action)
            # update the information for the loss with lookahead: remember the successes and failures
            rollout, rollout_error, i_train = get_rollouts(tr, rollout, rollout_error)
            loss = get_loss(self.plant, rollout, rollout_error, i_train, loss)

            loss_compute = dy.esum(loss)
            loss_compute.value()
            loss_compute.backward()
            self.trainer.update()
            loss = [dy.scalarInput(0)]
            self.trigger_transition1(tr)
            execution.append(tr)
            last_transition = tr[0]
        return execution,loss
    def choose_type(self,Ig_type):
        if(Ig_type==1):
            Ig, h_not_query_rnn, h_quering=self.plant.Ig_final_not_trained()
        if (Ig_type == 2):
            Ig, h_not_query_rnn, h_quering = self.plant.Ig_final_trained_BothwithRnn()
        if (Ig_type == 3):
            Ig, h_not_query_rnn, h_quering = self.Ig_final_withRnn_and_oracle()

        return Ig, h_not_query_rnn, h_quering

    def generate_training_execution_with_Ig(self, steps=50, lookahead=0,compare_loss = False, epsilon=0,use_oracle=False,Ig_type=4,threshold=0.3):
        rollout = [None] * (lookahead + 1)
        rollout_error = [None] * (lookahead + 1)

        def rollout_update(rollout, new_state):
            return rollout[1:]+[new_state]

        def rollout_error_update(rollout_errors, error):
            return rollout_errors[1:]+[error]

        def get_rollouts(tr, rollout, rollout_error):
            plant_tr = tr[0]
            if plant_tr[:4] == "fail":
                plant_tr = plant_tr[5]
                rollout_error = rollout_error_update(rollout_error, True)
            else:
                rollout_error = rollout_error_update(rollout_error, False)
            # store information about the output to compute the loss

            rollout = rollout_update(rollout, (output, self.plant.available_transitions(), plant_tr))
            i_train = next(i for i in range(len(rollout)) if rollout[i] is not None)
            return rollout, rollout_error, i_train

        def get_loss(p: process, rollout, rollout_error, i_train, loss):
            if rollout[i_train] is not None:
                nb_failures = rollout_error.count(True)  # count successes and failures in lookahead window
                nb_successes = 1 + lookahead - nb_failures
                for i in range(len(self.plant.all_transitions)):
                    if self.plant.all_transitions[i] in rollout[i_train][1]:
                        if self.plant.all_transitions[i] == rollout[i_train][2]:  # chosen action
                            loss.append((nb_successes/(lookahead+1))*dy.pickneglogsoftmax(rollout[i_train][0],i))
                        else:  # not chosen action
                            loss.append((nb_failures/(lookahead+1))*dy.pickneglogsoftmax(rollout[i_train][0],i))
            return loss

        Ig, h_not_query_rnn, h_quering=self.choose_type(Ig_type)
        #Ig, h_not_query_rnn,h_quering = self.Ig_final_withRnn_with_oracle()
        execution = []
        last_transition = None
        dy.renew_cg()
        state = self.network.initial_state()
        loss = [dy.scalarInput(0)]
        for step in range(steps):
            network_input = self.RNN_input(last_transition)
            input_vector = dy.inputVector(network_input)
            state = state.add_input(input_vector)
            output = dy.softmax(self.R*state.output() + self.bias)
            output_value = output.value()
            if (Ig[self.plant.current_state] > threshold):
                use_oracle = True
            else:
                use_oracle = False
            if(use_oracle):
                oracle_state = self.plant.current_state + self.environment.current_state
                if oracle_state in self.oracle:
                    #next_plant_action = self.oracle[oracle_state]

                    if (len(self.oracle[oracle_state]) > 1):
                        next_plant_action = random.choice(self.oracle[oracle_state])
                    else:
                        next_plant_action = self.oracle[oracle_state]

                elif random.random() < epsilon:
                        next_plant_action = self.random_transition()[0]
                else:
                    next_plant_action = self.RNN_output(output_value)
            else:
                next_plant_action = self.RNN_output(output_value)
            tr = self.check_transition1(next_plant_action)
            # update the information for the loss with lookahead: remember the successes and failures
            rollout, rollout_error, i_train = get_rollouts(tr, rollout, rollout_error)
            loss = get_loss(self.plant, rollout, rollout_error, i_train, loss)

            loss_compute = dy.esum(loss)
            loss_compute.value()
            loss_compute.backward()
            self.trainer.update()
            loss = [dy.scalarInput(0)]
            self.trigger_transition1(tr)
            execution.append(tr)
            last_transition = tr[0]
        return execution,loss

    ########################################################################################################################


    def generate_controlled_execution1(self, steps,use_oracle=False):
        execution = []
        dy.renew_cg()
        state = self.network.initial_state()
        last_transition = None
        Ig, h_not_query_rnn,h_quering=self.Ig_final_withRnn_with_oracle()
        for step in range(steps):
            network_input = self.RNN_input(last_transition)
            input_vector = dy.inputVector(network_input)
            state = state.add_input(input_vector)
            output = dy.softmax(self.R * state.output() + self.bias).value()
            x=Ig[self.plant.current_state]
            if(Ig[self.plant.current_state] >0):
                use_oracle=True
            else:
                use_oracle=False
            if(use_oracle):
                oracle_state = self.plant.current_state + self.environment.current_state
                if oracle_state in self.oracle:
                    if (len(self.oracle[oracle_state]) > 1):
                        next_action = random.choice(self.oracle[oracle_state])
                    else:
                        next_action = self.oracle[oracle_state]
                else:
                    next_action = self.RNN_output(output)
            else:
                next_action = self.RNN_output(output)
            tr = self.check_transition1(next_action)
            self.trigger_transition1(tr)
            execution.append(tr)
            last_transition = tr[0]
        return execution

    def generate_training_execution1(self,dict, steps=50, lookahead=0,compare_loss = False, epsilon=0,use_oracle=False):
        rollout = [None] * (lookahead + 1)
        rollout_error = [None] * (lookahead + 1)

        def rollout_update(rollout, new_state):
            return rollout[1:]+[new_state]

        def rollout_error_update(rollout_errors, error):
            return rollout_errors[1:]+[error]

        def get_rollouts(tr, rollout, rollout_error):
            plant_tr = tr[0]
            if plant_tr[:4] == "fail":
                plant_tr = plant_tr[5]
                rollout_error = rollout_error_update(rollout_error, True)
            else:

                rollout_error = rollout_error_update(rollout_error, False)
            # store information about the output to compute the loss
            rollout = rollout_update(rollout, (output, self.plant.available_transitions(), plant_tr))
            i_train = next(i for i in range(len(rollout)) if rollout[i] is not None)
            return rollout, rollout_error, i_train

        def get_loss(p: process, rollout, rollout_error, i_train, loss):
            if rollout[i_train] is not None:
                nb_failures = rollout_error.count(True)  # count successes and failures in lookahead window
                nb_successes = 1 + lookahead - nb_failures
                for i in range(len(self.plant.all_transitions)):
                    if self.plant.all_transitions[i] in rollout[i_train][1]:
                        if self.plant.all_transitions[i] == rollout[i_train][2]:  #if the action that was chosen is available from this state
                            loss.append((nb_successes/(lookahead+1))*dy.pickneglogsoftmax(rollout[i_train][0],i))
                        else:  # not chosen action
                            loss.append((nb_failures/(lookahead+1))*dy.pickneglogsoftmax(rollout[i_train][0],i))
            return loss

        execution = []
        last_transition = None
        dy.renew_cg()
        state = self.network.initial_state()
        loss = [dy.scalarInput(0)]
        for step in range(steps):
            network_input = self.RNN_input(last_transition)
            input_vector = dy.inputVector(network_input)
            state = state.add_input(input_vector)
            output = dy.softmax(self.R*state.output() + self.bias)
            output_value = output.value()

            if(use_oracle):
                oracle_state = self.plant.current_state + self.environment.current_state
                if oracle_state in self.oracle:
                    #next_plant_action = self.oracle[oracle_state]

                    if (len(self.oracle[oracle_state]) > 1):
                        next_plant_action = random.choice(self.oracle[oracle_state])
                    else:
                        next_plant_action = self.oracle[oracle_state]

                elif random.random() < epsilon:
                        next_plant_action = self.random_transition()[0]
                else:
                    next_plant_action = self.RNN_output(output_value)
            else:
                next_plant_action = self.RNN_output(output_value)
            tr = self.check_transition1(next_plant_action)
            # update the information for the loss with lookahead: remember the successes and failures
            rollout, rollout_error, i_train = get_rollouts(tr, rollout, rollout_error)
            loss = get_loss(self.plant, rollout, rollout_error, i_train, loss)

            loss_compute = dy.esum(loss)
            loss_compute.value()
            loss_compute.backward()
            self.trainer.update()
            loss = [dy.scalarInput(0)]
            self.trigger_transition1(tr)
            execution.append(tr)
            last_transition = tr[0]
        return dict #loss,rollout_error.count(True)

    #like generate_training_execution1 but without making a dictionary
    def generate_training_execution11(self,steps=50, lookahead=1,compare_loss = False, epsilon=0,use_oracle=False):


        rollout = [None] * (lookahead + 1)
        rollout_error = [None] * (lookahead + 1)

        def rollout_update(rollout, new_state):
            return rollout[1:]+[new_state]

        def rollout_error_update(rollout_errors, error):
            return rollout_errors[1:]+[error]

        def get_rollouts(tr, rollout, rollout_error):
            plant_tr = tr[0]
            if plant_tr[:4] == "fail":
                plant_tr = plant_tr[5]
                rollout_error = rollout_error_update(rollout_error, True)
            else:
                rollout_error = rollout_error_update(rollout_error, False)
            # store information about the output to compute the loss
            rollout = rollout_update(rollout, (output, self.plant.available_transitions(), plant_tr))
            i_train = next(i for i in range(len(rollout)) if rollout[i] is not None)
            return rollout, rollout_error, i_train

        def get_loss(p: process, rollout, rollout_error, i_train, loss):
            if rollout[i_train] is not None:
                nb_failures = rollout_error.count(True)  # count successes and failures in lookahead window
                nb_successes = 1 + lookahead - nb_failures
                for i in range(len(self.plant.all_transitions)):
                    if self.plant.all_transitions[i] in rollout[i_train][1]:
                        if self.plant.all_transitions[i] == rollout[i_train][2]:  # chosen action
                            loss.append((nb_successes/(lookahead+1))*dy.pickneglogsoftmax(rollout[i_train][0],i))
                        else:  # not chosen action
                            loss.append((nb_failures/(lookahead+1))*dy.pickneglogsoftmax(rollout[i_train][0],i))
            return loss

        execution = []
        last_transition = None
        dy.renew_cg()
        state = self.network.initial_state()
        loss = [dy.scalarInput(0)]
        for step in range(steps):
            network_input = self.RNN_input(last_transition)
            input_vector = dy.inputVector(network_input)
            state = state.add_input(input_vector)
            output = dy.softmax(self.R*state.output() + self.bias)
            output_value = output.value()
            if(use_oracle):
                oracle_state = self.plant.current_state + self.environment.current_state
                if oracle_state in self.oracle:
                    #next_plant_action = self.oracle[oracle_state]

                    if (len(self.oracle[oracle_state]) > 1):
                        next_plant_action = random.choice(self.oracle[oracle_state])
                    else:
                        next_plant_action = self.oracle[oracle_state]

                elif random.random() < epsilon:
                        next_plant_action = self.random_transition()[0]
                else:
                    next_plant_action = self.RNN_output(output_value)
            else:
                next_plant_action = self.RNN_output(output_value)
            tr = self.check_transition1(next_plant_action)
            # update the information for the loss with lookahead: remember the successes and failures
            rollout, rollout_error, i_train = get_rollouts(tr, rollout, rollout_error)
            loss = get_loss(self.plant, rollout, rollout_error, i_train, loss)

            loss_compute = dy.esum(loss)
            loss_compute.value()
            loss_compute.backward()
            self.trainer.update()
            loss = [dy.scalarInput(0)]
            self.trigger_transition1(tr)
            execution.append(tr)
            last_transition = tr[0]
        return execution
    def generate_controlled_execution_with_Ig(self,steps,df,tests=5,use_oracle=False,Ig_type=1,threshold=0.3):
            execution = []
            dy.renew_cg()
            state = self.network.initial_state()
            last_transition = None
            #Ig, h_not_query_rnn, h_quering=self.choose_type(Ig_type)
            ig=df.loc[tests]
            Ig_test=ig[Ig_type].replace('\\','')
            Ig_test = eval(Ig_test)
            for step in range(steps):
                network_input = self.RNN_input(last_transition)
                input_vector = dy.inputVector(network_input)
                state = state.add_input(input_vector)
                output = dy.softmax(self.R * state.output() + self.bias).value()

                if (Ig_test[self.plant.current_state] > threshold):
                    use_oracle = True
                else:
                    use_oracle = False
                if(use_oracle):
                    oracle_state = self.plant.current_state + self.environment.current_state
                    if oracle_state in self.oracle:
                        if (len(self.oracle[oracle_state]) > 1):
                            next_action = random.choice(self.oracle[oracle_state])
                        else:
                            next_action = self.oracle[oracle_state]
                    else:
                        next_action = self.RNN_output(output)
                else:
                    next_action = self.RNN_output(output)

                tr = self.check_transition1(next_action)
                self.trigger_transition1(tr)
                execution.append(tr)
                last_transition = tr[0]
            return execution

    def generate_training_for_dict(self,dict, steps):
        execution = []
        dy.renew_cg()
        state = self.network.initial_state()
        last_transition = None
        for step in range(steps):
            network_input = self.RNN_input(last_transition)
            input_vector = dy.inputVector(network_input)
            state = state.add_input(input_vector)
            output = dy.softmax(self.R * state.output() + self.bias).value()
            dict[self.plant.current_state]=output
            next_action = self.RNN_output(output)
            tr = self.check_transition1(next_action)
            self.trigger_transition1(tr)
            execution.append(tr)
            last_transition = tr[0]
        return dict

    ###################################H query with oracle ############################
    def H_query_for_all_states_with_oracle(self):
        list = []
        states = sorted(self.plant.states)
        d = dict.fromkeys(states, [])

        for i in d:
            p = self.H_query_for_state_with_oracle(i)
            d[i] = p
        return d

    # calc for each state the probabilities for each action by the oracle dictionary
    def calc_prob_for_States_by_oracle(self,state):
        oracle_dict = self.oracle
        l = [v for k, v in oracle_dict.items() if k.startswith(state)]
        state_dict = dict((x, l.count(x)) for x in set(l))
        result = {}
        for k, v in state_dict.items():
            for k_i in k:
                if ((k_i) in result):
                    result[k_i] += (v) / len(k)
                else:
                    result[k_i] = (v) / len(k)
        s = sum(result.values())
        for key in result.keys():
            result[key] /= s
        return result

    # H(oracle=o_trans)=1/len(actions)*[sigma (p(system=t|oracle=o_trans)*log(p(system=t|oracle=o_trans))] for all action t
    def H_with_queryO_using_oracle_dict(self, state, o_trans):
        sum = 0
        prob = 0
        state_dict = self.calc_prob_for_States_by_oracle(state) #dictionary of the  actions that are chosen by the oracle for this state
        if o_trans in state_dict:
            prob = state_dict[o_trans]                                                         #and its probability, for example {'b':0.8,'c':0.2}
        actions=state_dict.keys() # actions available in the oracle dictionary for the state
        available = self.plant.all_transitions
        for action in available:
            cond_p = self.plant.conditional_prob(state, action, o_trans)
            if (cond_p != 0):
                sum += cond_p * math.log2(cond_p)
        if (sum != 0):
            sum = -sum
        return  sum*prob

    # gets a state and return a list of all IG (for all actions), for example if the actions are a,b,c so for state g1 returns list=[IG(a),IG(b),IG(c)]
    def H_query_for_state_with_oracle(self, state):

        list = []
        trans = self.plant.all_transitions
        for i in trans:
            t = self.H_with_queryO_using_oracle_dict(state, i)  # H(i)
            list.append(t)
        return list
     ######################################################################################################
    """
    def Ig_final_with_oracle(self):
        H_q=self.H_query_for_all_states_with_oracle()
        for key, values in H_q.items():
            H_q[key] = sum(values)
        h=self.plant.H_all_states_not_query()
        Ig=H_q.copy()
        for key, values in H_q.items():
            Ig[key] =h[key]-H_q[key]
        return Ig,h,H_q
    """
    def Ig_final_withRnn_and_oracle(self):
        H_q=self.H_query_for_all_states_with_oracle()
        for key, values in H_q.items():
            H_q[key] = sum(values)
        h=self.plant.H_all_states_not_query_byRnn()
        Ig = H_q.copy()
        for key, values in H_q.items():
            Ig[key] =h[key]-H_q[key]
        return Ig,h,H_q
    def Ig_final_unifrom_and_oracle(self):
        H_q=self.H_query_for_all_states_with_oracle()
        for key, values in H_q.items():
            H_q[key] = sum(values)
        h=self.plant.H_all_states_not_query_uniform()
        Ig = H_q.copy()
        for key, values in H_q.items():
            Ig[key] =h[key]-H_q[key]
        return Ig,h,H_q


