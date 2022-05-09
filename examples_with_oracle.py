from system import *
import itertools

def create_fig_cases_teacher_oracle(model):
#def create_fig1_1_cases_oracle(model):
    dict = {
        "g1e1": "a",
        "g1e2": "a",
        "g1e3": "a",
        "g1e4": "a",
        "g1e5": "a",
        "g2e1": "b",
        "g2e2": "b",
        "g2e3": "c",
        "g2e4": "b",
        "g2e5": "b"
    }
    plant = process("plant", ["g1", "g2"], [], [], "g1", update_states=["g2"])
    environment = process("environment", ["e1", "e2", "e3", "e4", "e5"], [], [], "e1")

    pve = plant_environment("syst", plant, environment, model=model, oracle=dict)

    pve.add_transition("a", ["plant", "environment"], [["g1"], ["e2", "e5"]], [["g2"], ["e4", "e3"]])
    pve.add_transition("b", ["plant", "environment"], [["g2"], ["e1", "e4"]], [["g1"], ["e2", "e2"]])
    pve.add_transition("c", ["plant", "environment"], [["g2"], ["e1", "e3"]], [["g1"], ["e3", "e5"]])

    pve.create_RNN()
    pve.reinitialize()
    return pve
def create_fig_strategy_teacher_oracle(model):
#def create_fig1_2_strategy_oracle(model):
    dict = {
        "g2e4": "b",
        "g2e5": "c"
    }

    plant = process("plant", ["g1", "g2", "g3", "g4"], [], [], "g1")
    environment = process("environment", ["e1", "e2", "e3", "e4", "e5", "e6", "e7"], [], [], "e1")

    pve = plant_environment("syst", plant, environment, model=model, oracle=dict)

    pve.add_transition("a", ["plant", "environment"], [["g1"], ["e2", "e3"]], [["g2"], ["e4", "e5"]])
    pve.add_transition("b", ["plant", "environment"], [["g2", "g3"], ["e1", "e5", "e6"]],
                       [["g3", "g3"], ["e2", "e7", "e6"]])
    pve.add_transition("c", ["plant", "environment"], [["g2", "g4"], ["e1", "e4", "e7"]],
                       [["g4", "g4"], ["e3", "e6", "e7"]])

    pve.create_RNN()
    pve.reinitialize()
    return pve

def create_fig_combination_lock_teacher_oracle(model):
#def create_fig1_3_combination_lock_oracle(model):
    plant = process("plant",["g0","g1","g2","g3"],[],[],"g0",update_states = ["g3"])
    environment = process("environment",["s0","s1","s2","s3","s4"],[],[],"s0")

    dict = {
        "g0s0":"a",
        "g1s1":"b",
        "g2s2":"a",
        "g3s3":"c"
    }

    pve = plant_environment("syst",plant,environment,model = model,oracle=dict)

    pve.add_transition("a",["plant","environment"],[["g0","g1","g2"],["s0","s1","s2","s4"]],[["g1","g2","g3"],["s1","s4","s3","s4"]])
    pve.add_transition("b",["plant","environment"],[["g0","g1","g2"],["s0","s1","s2","s4"]],[["g1","g2","g3"],["s4","s2","s4","s4"]])
    pve.add_transition("c",["plant","environment"],[["g3"],["s3"]],[["g3"],["s3"]])

    pve.create_RNN()
    pve.reinitialize()
    return pve
def create_fig_cases_expert_oracle(model):
#def create_fig2_1_cases_oracle(model):
    dict = {
        "g1e1": ["b", "c"],
        "g1e2": "a",
        "g1e3": "c",
        "g1e5": "a",
        "g2e3": "c",
        "g2e4": "b"

        # "g2e5": "b"
    }
    plant = process("plant", ["g1", "g2"], [], [], "g1", update_states=["g2"])
    environment = process("environment", ["e1", "e2", "e3", "e4", "e5"], [], [], "e1")

    pve = plant_environment("syst", plant, environment, model=model, oracle=dict)

    pve.add_transition("a", ["plant", "environment"], [["g1"], ["e2", "e5"]], [["g2"], ["e4", "e3"]])
    pve.add_transition("b", ["plant", "environment"], [["g2"], ["e1", "e4"]], [["g1"], ["e2", "e2"]])
    pve.add_transition("c", ["plant", "environment"], [["g2"], ["e1", "e3"]], [["g1"], ["e3", "e5"]])

    pve.create_RNN()
    pve.reinitialize()
    return pve
def create_fig_strategy_expert_oracle(model):
#def create_fig2_2_strategy_oracle(model):
    dict = {
        "g1e1": ["b", "c"],  # **** from g1 can do only a, from e1 can do b or c
        "g1e2": "a",  # agree with plant,the only option
        "g1e3": "a",  # agree with plant,the only option
        "g2e4": "c",  # old
        "g2e5": "b",  # old
        "g3e7": "c",  # from g3 can do only b, from e7 can do only c
        "g4e6": "b",  # from g4 can do only c, from e6 can do only b

    }

    plant = process("plant", ["g1", "g2", "g3", "g4"], [], [], "g1")
    environment = process("environment", ["e1", "e2", "e3", "e4", "e5", "e6", "e7"], [], [], "e1")

    pve = plant_environment("syst", plant, environment, model=model, oracle=dict)

    pve.add_transition("a", ["plant", "environment"], [["g1"], ["e2", "e3"]], [["g2"], ["e4", "e5"]])
    pve.add_transition("b", ["plant", "environment"], [["g2", "g3"], ["e1", "e5", "e6"]],
                       [["g3", "g3"], ["e2", "e7", "e6"]])
    pve.add_transition("c", ["plant", "environment"], [["g2", "g4"], ["e1", "e4", "e7"]],
                       [["g4", "g4"], ["e3", "e6", "e7"]])

    pve.create_RNN()
    pve.reinitialize()
    return pve
#combination lock automaton aba
def create_fig_combination_lock_expert_oracle(model):
#def create_fig2_3_combination_lock_oracle(model):
    plant = process("plant",["g0","g1","g2","g3"],[],[],"g0",update_states = ["g3"])
    environment = process("environment",["s0","s1","s2","s3","s4"],[],[],"s0")

    dict = {
        "g0s0":["a","b"],
        "g1s1":["a","b"],
        "g2s4":["a","b"],
        "g3s4":["a","b"],
        "g1s4":["a","b"],
        "g2s2":["a","b"],
        "g3s3":"c"
    }

    pve = plant_environment("syst",plant,environment,model = model,oracle=dict)

    pve.add_transition("a",["plant","environment"],[["g0","g1","g2"],["s0","s1","s2","s4"]],[["g1","g2","g3"],["s1","s4","s3","s4"]])
    pve.add_transition("b",["plant","environment"],[["g0","g1","g2"],["s0","s1","s2","s4"]],[["g1","g2","g3"],["s4","s2","s4","s4"]])
    pve.add_transition("c",["plant","environment"],[["g3"],["s3"]],[["g3"],["s3"]])

    pve.create_RNN()
    pve.reinitialize()
    return pve












def create_fig2_schedule(model):
    plant = process("plant", ["g1", "g2", "g3"], [], [], "g1")
    environment = process("environment", ["e1", "e2", "e3", "e4"], [], [], "e1")

    pve = plant_environment("syst", plant, environment, model=model)

    pve.add_transition("a", ["plant", "environment"], [["g1"], ["e1", "e4"]], [["g2"], ["e3", "e3"]])
    pve.add_transition("b", ["plant", "environment"], [["g1", "g2"], ["e1"]], [["g3", "g1"], ["e2"]])
    pve.add_transition("c", ["plant", "environment"], [["g3"], ["e2", "e3"]], [["g1"], ["e1", "e4"]])

    pve.create_RNN()
    pve.reinitialize()
    return pve


def create_fig3_cases(model):
    plant = process("plant", ["g1", "g2"], [], [], "g1", update_states=["g2"])
    environment = process("environment", ["e1", "e2", "e3", "e4", "e5"], [], [], "e1")

    pve = plant_environment("syst", plant, environment, model=model)

    pve.add_transition("a", ["plant", "environment"], [["g1"], ["e2", "e5"]], [["g2"], ["e4", "e3"]])
    pve.add_transition("b", ["plant", "environment"], [["g2"], ["e1", "e4"]], [["g1"], ["e2", "e2"]])
    pve.add_transition("c", ["plant", "environment"], [["g2"], ["e1", "e3"]], [["g1"], ["e3", "e5"]])

    pve.create_RNN()
    pve.reinitialize()
    return pve

def create_fig1_4_schedule_oracle(model):
    dict = {
        "g1e1": "b"
    }
    plant = process("plant", ["g1", "g2", "g3"], [], [], "g1")
    environment = process("environment", ["e1", "e2", "e3", "e4"], [], [], "e1")

    pve = plant_environment("syst", plant, environment, model, oracle=dict)

    pve.add_transition("a", ["plant", "environment"], [["g1"], ["e1", "e4"]], [["g2"], ["e3", "e3"]])
    pve.add_transition("b", ["plant", "environment"], [["g1", "g2"], ["e1"]], [["g3", "g1"], ["e2"]])
    pve.add_transition("c", ["plant", "environment"], [["g3"], ["e2", "e3"]], [["g1"], ["e1", "e4"]])

    pve.create_RNN()
    pve.reinitialize()
    return pve

def create_fig1_5_circle_abc(model):
    print("cycle_abc_oracle")
    plant = process("plant", ["p0"], [], [], "p0")
    environment = process("environment", ["e0", "e1", "e2"], [], [], "e0")

    dict = {
        "p0e0": "a",
        "p0e1": "b",
        "p0e2": "c"
    }
    pve = plant_environment("syst", plant, environment, model=model, oracle=dict)

    pve.add_transition("a", ["plant", "environment"], [["p0"], ["e0"]], [["p0"], ["e1"]])
    pve.add_transition("b", ["plant", "environment"], [["p0"], ["e1"]], [["p0"], ["e2"]])
    pve.add_transition("c", ["plant", "environment"], [["p0"], ["e2"]], [["p0"], ["e0"]])

    pve.create_RNN()
    pve.reinitialize()
    return pve

def create_fig1_6_choise_scc_oracle(model):
    print("choice_scc_oracle")
    dict = {
        "g1e0": "d"
    }
    plant = process("plant", ["g1", "g2"], [], [], "g1", update_states=["g1"])
    environment = process("environment", ["e0", (l + str(i) for (l, i) in itertools.product("abcd", range(1, 7)))],
                          [], [], "e0")

    pve = plant_environment("syst", plant, environment, model=model, oracle=dict)

    # pve.add_transition("a",["plant","environment"],[["g1"],["e0"]],[["g2"],["a1"]])
    pve.add_transition("b", ["plant", "environment"], [["g1"], ["e0"]], [["g2"], ["b1"]])
    pve.add_transition("c", ["plant", "environment"], [["g1"], ["e0"]], [["g2"], ["c1"]])
    pve.add_transition("d", ["plant", "environment"], [["g1"], ["e0"]], [["g2"], ["d1"]])
    pve.add_transition("e", ["plant", "environment"],
                       [["g2"], ["a1", "a2", "a3", "b1", "b2", "b4", "c1", "c4", "c5", "d4", "d5", "d6"]],
                       [["g2"], ["a2", "a3", "a4", "b2", "b3", "b5", "c2", "c5", "c6", "d5", "d6", "d4"]])
    pve.add_transition("f", ["plant", "environment"],
                       [[], ["a4", "a5", "a6", "b3", "b5", "b6", "c2", "c3", "c6", "d1", "d2", "d3"]],
                       [[], ["a5", "a6", "a4", "b4", "b6", "b4", "c3", "c4", "c4", "d2", "d3", "d4"]])

    pve.create_RNN()
    pve.reinitialize()
    return pve

    """
def create_fig4_unlucky(model):
    plant = process("plant", ["g1", "g2"], [], [], "g1")
    environment = process("environment", ["e1", "e2", "e3", "e4", "e5"], [], [], "e1")

    pve = plant_environment("syst", plant, environment, model=model)

    pve.add_transition("a", ["plant", "environment"], [["g1"], ["e2", "e3"]], [["g2"], ["e4", "e5"]])
    pve.add_transition("b", ["plant", "environment"], [["g2"], ["e1", "e4"]], [["g1"], ["e2", "e2"]])
    pve.add_transition("c", ["plant", "environment"], [["g2"], ["e1", "e5"]], [["g1"], ["e3", "e3"]])

    pve.create_RNN()
    pve.reinitialize()
    return pve


def create_fig5_least_failures(model):
    plant = process("plant", ["g1", "g2"], [], [], "g1")
    environment = process("environment", ["e1", "e2", "e3", "e4"], [], [], "e1")

    pve = plant_environment("syst", plant, environment, model=model)

    pve.add_transition("a", ["plant", "environment"], [["g1"], ["e1"]], [["g2"], ["e2"]])
    pve.add_transition("b", ["plant", "environment"], [["g1"], ["e1"]], [["g2"], ["e3"]])
    pve.add_transition("c", ["plant", "environment"], [[], ["e2", "e3", "e5"]], [[], ["e4", "e5", "e4"]])
    pve.add_transition("d", ["plant", "environment"], [["g2"], ["e4"]], [["g1"], ["e1"]])

    pve.create_RNN()
    pve.reinitialize()
    return pve


def create_fig6_good_failure(model):
    plant = process("plant", ["s1", "s2", "s3"], [], [], "s1")
    environment = process("environment", ["e1", "e2", "e3"], [], [], "e1")

    pve = plant_environment("syst", plant, environment, model=model)

    pve.add_transition("a", ["plant", "environment"], [["s1"], ["e1"]], [["s2"], ["e2"]])
    pve.add_transition("b", ["plant", "environment"], [["s1", "s3"], ["e2", "e3"]], [["s3", "s3"], ["e3", "e3"]])
    pve.add_transition("c", ["plant", "environment"], [["s2"], []], [["s2"], []])

    pve.create_RNN()
    pve.reinitialize()
    return pve


def create_fig7_combination_lock(model):
    plant = process("plant", ["g0", "g1", "g2", "g3"], [], [], "g0", update_states=["g3"])
    environment = process("environment", ["s0", "s1", "s2", "s3", "s4"], [], [], "s0")

    pve = plant_environment("syst", plant, environment, model=model)

    pve.add_transition("a", ["plant", "environment"], [["g0", "g1", "g2"], ["s0", "s1", "s2", "s4"]],
                       [["g1", "g2", "g3"], ["s1", "s4", "s3", "s4"]])
    pve.add_transition("b", ["plant", "environment"], [["g0", "g1", "g2"], ["s0", "s1", "s2", "s4"]],
                       [["g1", "g2", "g3"], ["s4", "s2", "s4", "s4"]])
    pve.add_transition("c", ["plant", "environment"], [["g3"], ["s3"]], [["g3"], ["s3"]])

    pve.create_RNN()
    pve.reinitialize()
    return pve


def create_fig8_choice_scc(model):
    plant = process("plant", ["g1", "g2"], [], [], "g1", update_states=["g1"])
    environment = process("environment", ["e0", (l + str(i) for (l, i) in itertools.product("abcd", range(1, 7)))], [],
                          [], "e0")

    pve = plant_environment("syst", plant, environment, model=model)

    # pve.add_transition("a",["plant","environment"],[["g1"],["e0"]],[["g2"],["a1"]])
    pve.add_transition("b", ["plant", "environment"], [["g1"], ["e0"]], [["g2"], ["b1"]])
    pve.add_transition("c", ["plant", "environment"], [["g1"], ["e0"]], [["g2"], ["c1"]])
    pve.add_transition("d", ["plant", "environment"], [["g1"], ["e0"]], [["g2"], ["d1"]])
    pve.add_transition("e", ["plant", "environment"],
                       [["g2"], ["a1", "a2", "a3", "b1", "b2", "b4", "c1", "c4", "c5", "d4", "d5", "d6"]],
                       [["g2"], ["a2", "a3", "a4", "b2", "b3", "b5", "c2", "c5", "c6", "d5", "d6", "d4"]])
    pve.add_transition("f", ["plant", "environment"],
                       [[], ["a4", "a5", "a6", "b3", "b5", "b6", "c2", "c3", "c6", "d1", "d2", "d3"]],
                       [[], ["a5", "a6", "a4", "b4", "b6", "b4", "c3", "c4", "c4", "d2", "d3", "d4"]])

    pve.create_RNN()
    pve.reinitialize()
    return pve



def create_fig10_schedule_cycle(model):
    plant = process("plant", ["g1", "g2"], [], [], "g1")
    environment = process("environment", ["e1", "e2", "e3", "e4"], [], [], "e1")

    pve = plant_environment("syst", plant, environment, model=model)

    pve.add_transition("a", ["plant", "environment"], [["g1"], ["e1", "e4"]], [["g2"], ["e4", "e4"]])
    pve.add_transition("b", ["plant", "environment"], [["g1", "g2"], ["e1"]], [["g1", "g2"], ["e2"]])
    pve.add_transition("c", ["plant", "environment"], [["g1"], ["e2"]], [["g1"], ["e3"]])
    pve.add_transition("d", ["plant", "environment"], [["g1"], ["e3"]], [["g1"], ["e1"]])

    pve.create_RNN()
    pve.reinitialize()
    return pve


def create_fig11_cycle_scc(model):
    plant = process("plant", ["p0", "p1"], [], [], "p0")
    environment = process("environment", ["e" + str(i) for i in range(25)], [], [], "e0")

    pve = plant_environment("syst", plant, environment, model=model)

    pve.add_transition("a", ["plant", "environment"], [["p0"], ["e0"]], [["p1"], ["e1"]])
    pve.add_transition("b", ["plant", "environment"], [["p0"], ["e0"]], [["p1"], ["e7"]])
    pve.add_transition("c", ["plant", "environment"], [["p0"], ["e0"]], [["p1"], ["e13"]])
    pve.add_transition("d", ["plant", "environment"], [["p0"], ["e0"]], [["p1"], ["e19"]])
    pve.add_transition("f", ["plant", "environment"],
                       [[], ["e" + str(i) for i in [1, 2, 3, 7, 8, 10, 13, 16, 17, 22, 23, 24]]],
                       [[], ["e" + str(i) for i in [2, 3, 4, 8, 9, 11, 14, 17, 18, 23, 24, 22]]])
    pve.add_transition("x", ["plant", "environment"], [["p1"], ["e" + str(i) for i in [4, 9, 14, 15, 19, 20, 21]]],
                       [["p1"], ["e" + str(i) for i in [5, 10, 15, 16, 20, 21, 22]]])
    pve.add_transition("y", ["plant", "environment"], [["p1"], ["e" + str(i) for i in [5, 11, 9, 14, 15, 19, 20, 21]]],
                       [["p1"], ["e" + str(i) for i in [6, 12, 10, 15, 16, 20, 21, 22]]])
    pve.add_transition("z", ["plant", "environment"],
                       [["p1"], ["e" + str(i) for i in [6, 12, 18, 9, 14, 15, 19, 20, 21]]],
                       [["p1"], ["e" + str(i) for i in [4, 10, 16, 10, 15, 16, 20, 21, 22]]])

    pve.create_RNN()
    pve.reinitialize()
    return pve
    """