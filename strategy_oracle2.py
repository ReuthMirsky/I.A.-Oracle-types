
from system import *


# strategy from Doron's paper
def create(model):
    dict = {
        "g1e1":["b","c"], # **** from g1 can do only a, from e1 can do b or c
        "g1e2":"a", # agree with plant,the only option
        "g1e3":"a", # agree with plant,the only option
        "g2e4": "c",
        "g2e5": "b",
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