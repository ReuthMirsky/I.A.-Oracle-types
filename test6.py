import pandas as pd

from examples_with_oracle import *


# import random
# import dynet as dy
# import numpy as np
C = 100
L_C = 200
L_C_short = 50
model = "any_available"


def count_failures(execution):
    counter = 0
    for step in execution:
        if step[0][:4] == "fail":
            counter += 1
    return counter


def createByNum(num_model):
    model = "any_available"
    if (num_model == 0):
        pve = create_fig_cases_teacher_oracle(model)  # this generates the system and environment
    if (num_model == 1):
        pve = create_fig_strategy_teacher_oracle(model)  # this generates the system and environment
    if (num_model == 2):
        pve = create_fig_combination_lock_teacher_oracle(model)  # this generates the system and environment
    if (num_model == 3):
        pve = create_fig_cases_expert_oracle(model)  # this generates the system and environment
    if (num_model == 4):
        pve = create_fig_strategy_expert_oracle(model)  # this generates the system and environment
    if (num_model == 5):
        pve = create_fig_combination_lock_expert_oracle(model)  # this generates the system and environment
    # if (num_model == 6):
    #    pve = create_fig1_4_schedule_oracle(model)
    # if (num_model == 7):
    #    pve = create_fig1_5_circle_abc(model)
    # if (num_model == 8):
    #    pve = create_fig1_6_choise_scc_oracle(model)
    return pve


def print_use_case_name(num_model):
    if (num_model == 0):
        print("CASES TEACHER")
    if (num_model == 1):
        print("STRATEGY TEACHER")
    if (num_model == 2):
        print("COMBINATION LOCK TEACHER")
    if (num_model == 3):
        print("CASES EXPERT")
    if (num_model == 4):
        print("STRATEGY EXPERT")
    if (num_model == 5):
        print("COMBINATION LOCK EXPERT")


def print_model_name(num_model):
    if (num_model == 0):
        print("CASES teacher oracle")
    if (num_model == 1):
        print("STRATEGY teacher oracle")
    if (num_model == 2):
        print("COMBINATION LOCK teacher oracle")
    if (num_model == 3):
        print("CASES expert oracle")
    if (num_model == 4):
        print("STRATEGY expert oracle")
    if (num_model == 5):
        print("COMBINATION LOCK expert oracle")
    # if (num_model == 6):
    #    print("SCHEDULE")
    # if (num_model == 7):
    #    print("PERMITTE
    #    D")  # abc cyrcle
def model_name(num_model):
    if (num_model == 0):
        name="CASES teacher oracle"
    if (num_model == 1):
        name="STRATEGY teacher oracle"
    if (num_model == 2):
        name="COMBINATION LOCK teacher oracle"
    if (num_model == 3):
        name="CASES expert oracle"
    if (num_model == 4):
        name="STRATEGY expert oracle"
    if (num_model == 5):
        name="COMBINATION LOCK expert oracle"
    return name
    # if (num_model == 6):
def sumDicts(dict1,dict_dest,tests):
    if (tests == 0):
        dict_dest = dict1.copy()
    else:
        dict_dest = {key: dict_dest[key] + dict1[key] for key in dict1}
    return dict_dest


def run1(use_oracle_training=False, use_oracle_testing=False, num_model=0):
    tests_num = 5
    y = ['test' + i for i in map(str, range(1, tests_num + 1))]
    y.append('average')
    df_ig = pd.DataFrame(index=y ,columns=('IG1 (not trained)', 'IG2(trained, rnn+normal rnn)','IG3(trained,rnn+oracle)','IG4(trained,uniform+oracle)'))
    df_h_not_query = pd.DataFrame(columns=('H1_UNION', 'H2_RNN'))
    df_querying = pd.DataFrame(columns=('H1_UNION', 'H2_NORMAL_RNN', 'H3_ORACLE'))
    df_dict=pd.DataFrame(index=['test' + i for i in map(str, range(1, tests_num+1))])
    Ig1_sum=Ig2_sum=Ig3_sum=Ig4_sum=0
    all_dictionaries={}
    # to use a particular example, uncomment its line and comment all the others

    # If using a non-combined example, uncomment the next three lines
    number_of_epochs = 40  # this is what we called T so far
    training_execution_length = 20  # this is what we called L so far
    runs = 1  # will be >1 only for combined examples

    # If using a combined example, comment the three lines number_of_epochs and training_execution_length and runs above, and uncomment the ones below that correspond to the example:

    # from combine_scc_cycle_simplified import *
    # number_of_epochs = 100 #this is what we called T so far
    # training_execution_length = 8 #this is what we called L so far
    # runs = 5

    # from combine_schedule_cycle import *
    # number_of_epochs = 10 #this is what we called T so far
    # training_execution_length = 50 #this is what we called L so far
    # runs = 2

    # number of controlled tests after training, and length of control executions
    # C = 100
    # L_C = 200
    # L_C_short = 50
    model = "any_available"

    # T is the number of "epochs" we'll do the training on; if T is iterated on a
    # list of numbers, we'll do a different training with each number of epochs chosen

    results = []
    results_short = []

    best = L_C
    best_short = L_C_short
    worst = 0
    worst_short = 0
    print_use_case_name(num_model)
    print("Computation for", number_of_epochs, "epochs and training executions of length", training_execution_length)
    lookahead = 0
    epsilon = 0

    for tests in range(tests_num):
        print("\nTest ", tests+1,":")
        pve = createByNum(num_model)
        dict = {}
        for r in range(1):
            # iterating over the epochs
            for training in range(number_of_epochs):
                # iterating over the training sequences from length 1 to length L
                for length in range(1, training_execution_length):
                    for i in range(1, 5):
                        pve.reinitialize()  # return system and environment to initial states
                        # Now we generate a training sequence.
                        dict = pve.generate_training_execution1(dict, length, lookahead=lookahead, epsilon=epsilon,
                                                                use_oracle=use_oracle_training)  # compare_loss = False)

                        # pve.generate_training_execution1(length, lookahead=lookahead, epsilon=epsilon,use_oracle=use_oracle_training)  # compare_loss = False)
        dict2 = {}
        for r in range(1):
            # iterating over the epochs
            for training in range(number_of_epochs):
                # iterating over the training sequences from length 1 to length L
                for length in range(1, 40):
                    for i in range(1, 5):
                        pve.reinitialize()  # return system and environment to initial states
                        # Now we generate a training sequence.
                        dict2 = pve.generate_training_for_dict(dict2, length)
                        all_dictionaries[tests]=dict2
        pve.plant.pr_dict = dict2
        print("dict",dict2)

        Ig1, h_not_query1, h_querying1 = pve.plant.Ig_final_not_trained()
        print("Ig1:", Ig1, "\nh not query1:", h_not_query1, "\nh query1:", h_querying1)
        Ig2, h_not_query2, h_querying2 = pve.plant.Ig_final_trained_BothwithRnn()
        print("Ig2:", Ig2, "\nh not query2:", h_not_query2, "\nh query2:", h_querying2)
        Ig3, h_not_query3, h_querying3 = pve.Ig_final_withRnn_and_oracle()
        print("Ig3:", Ig3, "\nh not query3:", h_not_query3, "\nh query3:", h_querying3)

        Ig4, h_not_query4, h_querying4 = pve.Ig_final_unifrom_and_oracle()
        print("Ig4:", Ig4, "\nh not query4:", h_not_query4, "\nh query4:", h_querying4)

        # d = pve.plant.Ig_final()
        # dict = pve.generate_training_execution11(length, lookahead=lookahead, epsilon=epsilon,
        df_ig.loc[y[tests]]=[Ig1,Ig2,Ig3,Ig4]
        df_h_not_query.loc[y[tests]]=[h_not_query1,h_not_query2]
        df_querying.loc[y[tests]]=[h_querying1,h_querying2,h_querying3]
        df_dict.loc[y[tests]]=[dict2]

        #print("dict:",dict2)
        #print("df dict:",df_dict)
    #Ig1_sum = {k: v / tests for k, v in Ig1_sum.}

        Ig1_sum = sumDicts(Ig1, Ig1_sum, tests)
        Ig2_sum = sumDicts(Ig2, Ig2_sum, tests)
        Ig3_sum = sumDicts(Ig3, Ig3_sum, tests)
        Ig4_sum = sumDicts(Ig4, Ig4_sum, tests)

    Ig1_sum = {k: v / tests_num for k, v in Ig1_sum.items()}
    Ig2_sum = {k: v / tests_num for k, v in Ig2_sum.items()}
    Ig3_sum = {k: v / tests_num for k, v in Ig3_sum.items()}
    Ig4_sum = {k: v / tests_num for k, v in Ig4_sum.items()}
    df_ig.loc[y[tests+1]] = [Ig1_sum, Ig2_sum, Ig3_sum,Ig4_sum]
    df_dict = pd.DataFrame.from_dict(all_dictionaries).transpose()
    df_dict.index = ['test' + i for i in map(str, range(1, tests_num + 1))]
    print("df:",df_dict)
    """
    print("number of tests:", tests_num )
    print("Ig1 avg:", Ig1_sum/ tests_num)
    print("Ig2 avg:", Ig2_sum / tests_num)
    print("Ig3 avg:", Ig3_sum / tests_num)
    
    print("############  Global results:  ############")
    print("_Short control_")
    print("Best result:", best_short * 100, "%")
    print("Worst result:", worst_short * 100, "%")
    print("Average", sum(results_short) / len(results_short) * 100, "%")

    print("_Long control_")
    """
    name=model_name(num_model)+'.csv'
    df_ig.to_csv('Ig'+name)
    df_h_not_query.to_csv('H_not_query'+name)
    df_querying.to_csv('H_query'+name)
    df_dict.to_csv('RNN Dictionary'+name)



def run11(use_oracle_training=False, use_oracle_testing=False, num_model=0):
    # to use a particular example, uncomment its line and comment all the others

    # If using a non-combined example, uncomment the next three lines
    number_of_epochs = 40  # this is what we called T so far
    training_execution_length = 20  # this is what we called L so far
    runs = 1  # will be >1 only for combined examples
    all_dictionaries={}
    # If using a combined example, comment the three lines number_of_epochs and training_execution_length and runs above, and uncomment the ones below that correspond to the example:

    # from combine_scc_cycle_simplified import *
    # number_of_epochs = 100 #this is what we called T so far
    # training_execution_length = 8 #this is what we called L so far
    # runs = 5

    # from combine_schedule_cycle import *
    # number_of_epochs = 10 #this is what we called T so far
    # training_execution_length = 50 #this is what we called L so far
    # runs = 2

    # number of controlled tests after training, and length of control executions
    # C = 100
    # L_C = 200
    # L_C_short = 50
    model = "any_available"

    # T is the number of "epochs" we'll do the training on; if T is iterated on a
    # list of numbers, we'll do a different training with each number of epochs chosen

    results = []
    results_short = []

    best = L_C
    best_short = L_C_short
    worst = 0
    worst_short = 0
    print_use_case_name(num_model)
    print("Computation for", number_of_epochs, "epochs and training executions of length", training_execution_length)

    for tests in range(10):
        pve = createByNum(num_model)
        s = pve.plant.states
        # d = pve.plant.Ig_final()
        dict = {}
        for r in range(runs):

            for (lookahead, epsilon) in [(0, 0)]:
                # print("lookahead",lookahead,"and epsilon",epsilon)

                # iterating over the epochs
                for training in range(number_of_epochs):

                    # iterating over the training sequences from length 1 to length L
                    for length in range(1, training_execution_length):
                        for i in range(1, 5):
                            pve.reinitialize()  # return system and environment to initial states
                            # Now we generate a training sequence.
                            # dict = pve.generate_training_execution11(dict,length, lookahead=lookahead, epsilon=epsilon,use_oracle=use_oracle_training)  # compare_loss = False)

                            # pve.generate_training_execution1(length, lookahead=lookahead, epsilon=epsilon,use_oracle=use_oracle_training)  # compare_loss = False)

        pve.plant.pr_dict = dict
        Ig, h_not_query = pve.plant.Ig_final()
        Ig2, h_not_query_rnn = pve.plant.Ig_final_withRnn()
        # d = pve.plant.Ig_final()
        # dict = pve.generate_training_execution11(length, lookahead=lookahead, epsilon=epsilon,
        #                                         use_oracle=use_oracle_training)  # compare_loss = False)
        failures = []
        failures_short = []
        """
        for control in range(C):
            pve.reinitialize()
            execution = pve.generate_controlled_execution(L_C_short,
                                                          use_oracle=use_oracle_testing)  # ,print_probs = False)
            failures_short.append(count_failures(execution))
        """
        dict1 = {}
        for control in range(C):
            pve.reinitialize()
            # execution = pve.generate_controlled_execution(L_C,
            #                                              use_oracle=use_oracle_testing)  # ,print_probs = False)
            execution, dict1 = pve.generate_controlled_execution1(dict1, L_C,
                                                                  use_oracle=use_oracle_testing)  # ,print_probs = False)
            failures.append(count_failures(execution))

        pve.plant.pr_dict = dict1
        Ig, h_not_query = pve.plant.Ig_final()
        Ig2, h_not_query_rnn = pve.plant.Ig_final_withRnn()
        percentage = 0
        percentage_short = 0
        for i in range(C):
            percentage += failures[i] / L_C
            # percentage_short += failures_short[i] / L_C_short
        percentage /= C
        # percentage *= 100
        # percentage_short /= C
        # percentage_short *= 100

        if percentage > worst:
            worst = percentage
        if percentage < best:
            best = percentage
        """
        if percentage_short > worst_short:
            worst_short = percentage_short
        if percentage_short < best_short:
            best_short = percentage_short
        """
        results.append(percentage)
        # results_short.append(percentage_short)

        print("test number", tests + 1, percentage * 100, "%")
        # print("short control", percentage_short * 100, "%")
        #
        # print("long control", percentage * 100, "%")
    """
    print("############  Global results:  ############")
    print("_Short control_")
    print("Best result:", best_short * 100, "%")
    print("Worst result:", worst_short * 100, "%")
    print("Average", sum(results_short) / len(results_short) * 100, "%")

    print("_Long control_")
    """
    print("Best result:", best * 100, "%")
    print("Worst result:", worst * 100, "%")
    print("Average", sum(results) / len(results) * 100, "%")

    """
    def run(pve, steps=50, print_first=False, print_probs=False):
        pve.reinitialize()
        if print_first:
            print(pve.generate_controlled_execution(steps, print_probs=print_probs)[0][0])
        else:
            print(pve.generate_controlled_execution(steps, print_probs=print_probs))


    # same to test on a large amount of trainings
    def test(number_of_tests, number_of_runs, size, print_probs=False, random_exploration=False, new_loss=False,
             lookahead=1, epsilon=0, compare_loss=False):
        results = []

        T = number_of_runs
        L = size

        for test in range(number_of_tests):
            for T in [number_of_runs]:
                for L in [size]:
                    pve = create(model)
                    for training in range(T):
                        for length in range(1, L):
                            pve.reinitialize()
                            pve.generate_training_execution(length, print_probs=False,
                                                            random_exploration=random_exploration, new_loss=new_loss,
                                                            lookahead=T, epsilon=epsilon, compare_loss=compare_loss,use_oracle=False)

                            # print("____________")
                        for length in range(1, L):
                            pve.reinitialize()
                            pve.generate_training_execution(length, print_probs=False,
                                                            random_exploration=random_exploration, new_loss=new_loss,
                                                            lookahead=0, epsilon=epsilon, compare_loss=compare_loss,use_oracle=False)

                    failures = []
                    for control in range(C):
                        pve.reinitialize()
                        execution = pve.generate_controlled_execution(L_C, print_probs=print_probs,)
                        # print("____________")
                        failures.append(count_failures(execution))
                    percentage = 0
                    for i in range(C):
                        percentage += (failures[i] / L_C) * 100
                    percentage /= C
                    # print("test number",test+1,"(",T,",",L,")",percentage*100,"%")
                    # run(pve,True)
                    run(pve, False, print_probs=True)

            results.append(percentage)

        average = 0
        for r in results:
            average += r
        return average / len(results)


# =============================================================================
# for l in [0, 3, 20]:
#     for e in [0, 0.2, 0.5]:
#         print("lookahead = ", l, "epsilon = ", e)
#         print("new_loss", test(50, 50, 50, new_loss=True, lookahead=l, epsilon=e, compare_loss=False))
#         print("old_loss", test(50, 50, 50, new_loss=True, lookahead=l, epsilon=e, compare_loss=True))
# =============================================================================
    """
    """
# random
    pve_rand = create(model)
    failures = []
    for control in range(C):
        pve_rand.reinitialize()
        execution = pve_rand.generate_random_execution(L_C)
        failures.append(count_failures(execution))
    percentage = 0
    for i in range(C):
        percentage += failures[i] / L_C
        percentage /= C
    print("(random)", percentage * 100, "%")
    """


# run2 runs the use_case with oracle only in training and with oracle in training+testing

def run2(num_model=0):
    # to use a particular example, uncomment its line and comment all the others

    # If using a non-combined example, uncomment the next three lines
    number_of_epochs = 40  # this is what we called T so far
    training_execution_length = 20  # this is what we called L so far
    runs = 1  # will be >1 only for combined examples

    # If using a combined example, comment the three lines number_of_epochs and training_execution_length and runs above, and uncomment the ones below that correspond to the example:

    # from combine_scc_cycle_simplified import *
    # number_of_epochs = 100 #this is what we called T so far
    # training_execution_length = 8 #this is what we called L so far
    # runs = 5

    # from combine_schedule_cycle import *
    # number_of_epochs = 10 #this is what we called T so far
    # training_execution_length = 50 #this is what we called L so far
    # runs = 2

    # number of controlled tests after training, and length of control executions
    # C = 100
    # L_C = 200
    # L_C_short = 50
    model = "any_available"

    # T is the number of "epochs" we'll do the training on; if T is iterated on a
    # list of numbers, we'll do a different training with each number of epochs chosen

    results1 = []
    results2 = []
    results_short = []

    best1 = best2 = L_C
    best_short = L_C_short
    worst1 = worst2 = 0
    worst_short = 0
    print_model_name(num_model)
    print("Computation for", number_of_epochs, "epochs and training executions of length", training_execution_length)

    for tests in range(10):
        pve = createByNum(num_model)
        # pve1 = createByNum(0)
        # pve2 = createByNum(1)
        for r in range(runs):

            for (lookahead, epsilon) in [(0, 0)]:
                # print("lookahead",lookahead,"and epsilon",epsilon)

                # iterating over the epochs
                for training in range(number_of_epochs):

                    # iterating over the training sequences from length 1 to length L
                    for length in range(1, training_execution_length):
                        for t in range(1, 5):
                            pve.reinitialize()  # return system and environment to initial states
                            # pve1.reinitialize()
                            # pve2.reinitialize()
                            # Now we generate a training sequence.
                            execution1, loss1, nb_failures1 = pve.generate_training_execution1(length,
                                                                                               lookahead=lookahead,
                                                                                               epsilon=epsilon,
                                                                                               use_oracle=True)

                            # ex2=pve2.generate_training_execution1(length, lookahead=lookahead, epsilon=epsilon,use_oracle=True)

        failures1 = []
        failures2 = []
        failures_short = []
        """
        for control in range(C):
            pve.reinitialize()
            execution = pve.generate_controlled_execution(L_C_short,
                                                          use_oracle=use_oracle_testing)  # ,print_probs = False)
            failures_short.append(count_failures(execution))
        """
        for control in range(C):
            pve.reinitialize()
            execution1 = pve.generate_controlled_execution1(L_C, use_oracle=False)  # ,print_probs = False)
            failures1.append(count_failures(execution1))

            pve.reinitialize()
            execution2 = pve.generate_controlled_execution1(L_C, use_oracle=True)  # ,print_probs = False)
            failures2.append(count_failures(execution2))
        percentage1 = 0
        percentage2 = 0
        percentage_short = 0
        for i in range(C):
            percentage1 += failures1[i] / L_C
            percentage2 += failures2[i] / L_C
            # percentage_short += failures_short[i] / L_C_short
        percentage1 /= C
        percentage2 /= C
        # percentage *= 100
        # percentage_short /= C
        # percentage_short *= 100

        if percentage1 > worst1:
            worst1 = percentage1
        if percentage1 < best1:
            best1 = percentage1

        if percentage2 > worst2:
            worst2 = percentage2
        if percentage2 < best2:
            best2 = percentage2
        """
        if percentage_short > worst_short:
            worst_short = percentage_short
        if percentage_short < best_short:
            best_short = percentage_short
        """
        results1.append(percentage1)
        results2.append(percentage2)

        # results_short.append(percentage_short)

        print("test number", tests + 1, "oracle only in train : ", percentage1 * 100, "%")
        print("test number", tests + 1, "oracle in train and test : ", percentage2 * 100, "%")
        # print("short control", percentage_short * 100, "%")
        #
        # print("long control", percentage * 100, "%")
    """
    print("############  Global results:  ############")
    print("_Short control_")
    print("Best result:", best_short * 100, "%")
    print("Worst result:", worst_short * 100, "%")
    print("Average", sum(results_short) / len(results_short) * 100, "%")

    print("_Long control_")
    """
    print("result only oracle in train:")
    print("Best:", best1 * 100, "%")
    print("Worst:", worst1 * 100, "%")
    print("Average:", sum(results1) / len(results1) * 100, "%")
    print("result oracle in train and test:")
    print("Best:", best2 * 100, "%")
    print("Worst:", worst2 * 100, "%")
    print("Average:", sum(results2) / len(results2) * 100, "%")


# random
def run_random(num_model):
    pve_rand = createByNum(num_model)
    failures = []
    for control in range(C):
        pve_rand.reinitialize()
        execution = pve_rand.generate_random_execution(L_C)
        failures.append(count_failures(execution))
    percentage = 0
    for i in range(C):
        percentage += failures[i] / L_C
    percentage /= C
    print("(random)", percentage * 100, "%")


def run_model(num):
    print("***********No oracle**********")
    run1(False, False, num)
    print("***********only training oracle*******")
    run1(True, False, num)
    print("***********Training and testing oracle*****")
    run1(True, True, num)
    run_random(num)


# run_model(2)
# run_model(3)
# run_model(1)
# run1(False,False,2)


# run2(0)
# print("NEW ORACLE")
run1(False, False, 0)
run1(False, False, 1)
run1(False, False, 2)
run1(False, False, 3)
run1(False, False, 4)
run1(False, False, 5)




# run2(1)
# run2(3)
# run1(False,False,3)
# run2(3)
# run1(False,False,1)
# run2(1)