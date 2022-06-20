from examples_with_oracle import *
import pandas as pd
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
def run12(use_oracle_training=False, use_oracle_testing=False, num_model=0):
    # to us
    # def run1(use_oracle_training=False, use_oracle_testing=False, num_model=0):
    #     # to use a particular example, uncomment its line and comment all the others
    #
    #     # If using a non-combined example, uncomment the next three lines
    #     number_of_epochs = 40  # this is what we called T so far
    #     training_execution_length = 20  # this is what we called L so far
    #     runs = 1  # will be >1 only for combined examples
    #
    #     # If using a combined example, comment the three lines number_of_epochs and training_execution_length and runs above, and uncomment the ones below that correspond to the example:
    #
    #     # from combine_scc_cycle_simplified import *
    #     # number_of_epochs = 100 #this is what we called T so far
    #     # training_execution_length = 8 #this is what we called L so far
    #     # runs = 5
    #
    #     # from combine_schedule_cycle import *
    #     # number_of_epochs = 10 #this is what we called T so far
    #     # training_execution_length = 50 #this is what we called L so far
    #     # runs = 2
    #
    #     # number of controlled tests after training, and length of control executions
    #     # C = 100
    #     # L_C = 200
    #     # L_C_short = 50
    #     model = "any_available"
    #
    #     # T is the number of "epochs" we'll do the training on; if T is iterated on a
    #     # list of numbers, we'll do a different training with each number of epochs chosen
    #
    #     results = []
    #     results_short = []
    #
    #     best = L_C
    #     best_short = L_C_short
    #     worst = 0
    #     worst_short = 0
    #     print_use_case_name(num_model)
    #     print("Computation for", number_of_epochs, "epochs and training executions of length", training_execution_length)
    #     lookahead = 0
    #     epsilon = 0
    #     for tests in range(10):
    #         pve = createByNum(num_model)
    #         dict = {}
    #         for r in range(1):
    #             # iterating over the epochs
    #             for training in range(number_of_epochs):
    #                 # iterating over the training sequences from length 1 to length L
    #                 for length in range(1, training_execution_length):
    #                     for i in range(1, 5):
    #                         pve.reinitialize()  # return system and environment to initial states
    #                         # Now we generate a training sequence.
    #                         dict = pve.generate_training_execution1(dict, length, lookahead=lookahead, epsilon=epsilon,
    #                                                                 use_oracle=use_oracle_training)  # compare_loss = False)
    #
    #                         # pve.generate_training_execution1(length, lookahead=lookahead, epsilon=epsilon,use_oracle=use_oracle_training)  # compare_loss = False)
    #         dict2 = {}
    #         for r in range(1):
    #             # iterating over the epochs
    #             for training in range(number_of_epochs):
    #                 # iterating over the training sequences from length 1 to length L
    #                 for length in range(1, 40):
    #                     for i in range(1, 5):
    #                         pve.reinitialize()  # return system and environment to initial states
    #                         # Now we generate a training sequence.
    #                         dict2 = pve.generate_training_for_dict(dict2, length)
    #
    #         pve.plant.pr_dict = dict2
    #         print("dict:", dict2)
    #         Ig1, h_not_query1, h_quering1 = pve.plant.Ig_final_not_trained()
    #         print("Ig1:", Ig1, "\nh not query1:", h_not_query1, "\nh query1:", h_quering1)
    #         Ig2, h_not_query2, h_quering2 = pve.plant.Ig_final_trained_BothwithRnn()
    #         print("Ig2:", Ig2, "\nh not query2:", h_not_query2, "\nh query2:", h_quering2)
    #         Ig3, h_not_query3, h_quering3 = pve.Ig_final_withRnn_and_oracle()
    #         print("Ig3:", Ig3, "\nh not query3:", h_not_query3, "\nh query3:", h_quering3)
    #         # d = pve.plant.Ig_final()
    #         # dict = pve.generate_training_execution11(length, lookahead=lookahead, epsilon=epsilon,
    #         #                                         use_oracle=use_oracle_training)  # compare_loss = False)
    #         failures = []
    #
    #         ig_type = 2
    #         threshold = 0.25
    #         dict = {}
    #         for r in range(1):
    #             # iterating over the epochs
    #             for training in range(number_of_epochs):
    #                 # iterating over the training sequences from length 1 to length L
    #                 for length in range(1, training_execution_length):
    #                     for i in range(1, 5):
    #                         pve.reinitialize()  # return system and environment to initial states
    #                         # Now we generate a training sequence.
    #                         execution, loss = pve.generate_training_execution_with_Ig(length, lookahead=lookahead,
    #                                                                                   epsilon=epsilon,
    #                                                                                   use_oracle=use_oracle_training,
    #                                                                                   Ig_type=ig_type,
    #                                                                                   threshold=0.3)  # compare_loss = False)
    #                         # execution = pve.generate_training_execution11(length, lookahead=lookahead,compare_loss = False, epsilon=0,use_oracle=False)
    #         for control in range(C):
    #             pve.reinitialize()
    #             # execution = pve.generate_controlled_execution(L_C,use_oracle=use_oracle_testing)#'g4' = {list: 3} [0.994458794593811, 0.0034357940312474966, 0.0021054658573120832]  # ,print_probs = False)
    #             execution = pve.generate_controlled_execution_with_Ig(L_C, use_oracle=use_oracle_testing, Ig_type=ig_type,
    #                                                                   threshold=0.3)
    #             failures.append(count_failures(execution))
    #
    #         percentage = 0
    #         percentage_short = 0
    #         for i in range(C):
    #             percentage += failures[i] / L_C
    #             # percentage_short += failures_short[i] / L_C_short
    #         percentage /= C
    #         # percentage *= 100
    #         # percentage_short /= C
    #         # percentage_short *= 100
    #
    #         if percentage > worst:
    #             worst = percentage
    #         if percentage < best:
    #             best = percentage
    #         """
    #         if percentage_short > worst_short:
    #             worst_short = percentage_short
    #         if percentage_short < best_short:
    #             best_short = percentage_short
    #         """
    #         results.append(percentage)
    #         # results_short.append(percentage_short)
    #
    #         print("test number", tests + 1, percentage * 100, "%")
    #         # print("short control", percentage_short * 100, "%")
    #         #
    #         # print("long control", percentage * 100, "%")
    #     """
    #     print("############  Global results:  ############")
    #     print("_Short control_")
    #     print("Best result:", best_short * 100, "%")
    #     print("Worst result:", worst_short * 100, "%")
    #     print("Average", sum(results_short) / len(results_short) * 100, "%")
    #
    #     print("_Long control_")
    #     """
    #     print("Best result:", best * 100, "%")
    #     print("Worst result:", worst * 100, "%")
    #     print("Average", sum(results) / len(results) * 100, "%")e a particular example, uncomment its line and comment all the others

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
    failures = []
    #

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
    tests_num = 5
    for tests in range(10):
        pve = createByNum(num_model)
        """
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

        pve.plant.pr_dict = dict2
        print("dict:", dict2)
        Ig1, h_not_query1, h_quering1 = pve.plant.Ig_final_not_trained()
        print("Ig1:", Ig1, "\nh not query1:", h_not_query1, "\nh query1:", h_quering1)
        Ig2, h_not_query2, h_quering2 = pve.plant.Ig_final_trained_BothwithRnn()
        print("Ig2:", Ig2, "\nh not query2:", h_not_query2, "\nh query2:", h_quering2)
        Ig3, h_not_query3, h_quering3 = pve.Ig_final_withRnn_and_oracle()
        print("Ig3:", Ig3, "\nh not query3:", h_not_query3, "\nh query3:", h_quering3)
        # d = pve.plant.Ig_final()
        # dict = pve.generate_training_execution11(length, lookahead=lookahead, epsilon=epsilon,
        #                                         use_oracle=use_oracle_training)  # compare_loss = False)
        failures = []

        ig_type = 2
        threshold = 0.25
        dict = {}
        for r in range(1):
            # iterating over the epochs
            for training in range(number_of_epochs):
                # iterating over the training sequences from length 1 to length L
                for length in range(1, training_execution_length):
                    for i in range(1, 5):
                        pve.reinitialize()  # return system and environment to initial states
                        # Now we generate a training sequence.
                        execution, loss = pve.generate_training_execution_with_Ig(length, lookahead=lookahead,
                                                                                  epsilon=epsilon,
                                                                                  use_oracle=use_oracle_training,
                                                                                  Ig_type=ig_type,
                                                                                  threshold=0.3)  # compare_loss = False)
                        # execution = pve.generate_training_execution11(length, lookahead=lookahead,compare_loss = False, epsilon=0,use_oracle=False)
        """
        name = 'Ig'+model_name(num_model) + '.csv'
        df1= pd.read_csv(name)
        failures = []

        ig_type = 1
        threshold = 0.25
        for control in range(C):
            pve.reinitialize()
            #execution = pve.generate_controlled_execution(L_C,use_oracle=use_oracle_testing)#'g4' = {list: 3} [0.994458794593811, 0.0034357940312474966, 0.0021054658573120832]  # ,print_probs = False)
            execution = pve.generate_controlled_execution_with_Ig(L_C,df1,tests=tests_num,use_oracle=use_oracle_testing,Ig_type=ig_type,threshold=0.3)
            failures.append(count_failures(execution))

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




# =============================================================================
# for l in [0, 3, 20]:
#     for e in [0, 0.2, 0.5]:
#         print("lookahead = ", l, "epsilon = ", e)
#         print("new_loss", test(50, 50, 50, new_loss=True, lookahead=l, epsilon=e, compare_loss=False))
#         print("old_loss", test(50, 50, 50, new_loss=True, lookahead=l, epsilon=e, compare_loss=True))
# =============================================================================

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
    run12(False, False, num)
    print("***********only training oracle*******")
    run12(True, False, num)
    print("***********Training and testing oracle*****")
    run12(True, True, num)
    run_random(num)


# run_model(2)
# run_model(3)
# run_model(1)
# run1(False,False,2)


run12(False, False, 0)
run12(False, False, 1)
run12(False, False, 2)
run12(False, False, 3)
run12(False, False, 4)
run12(False, False, 5)