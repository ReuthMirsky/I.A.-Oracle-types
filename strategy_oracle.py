
from system import *

#strategy from Doron's paper
def create(model):
    dict = {
        "g2e4": "b",
        "g2e5": "c"
    }

    plant = process("plant",["g1","g2","g3","g4"],[],[],"g1")
    environment = process("environment",["e1","e2","e3","e4","e5","e6","e7"],[],[],"e1")
    
    pve = plant_environment("syst",plant,environment,model = model, oracle=dict)
    
    pve.add_transition("a",["plant","environment"],[["g1"],["e2","e3"]],[["g2"],["e4","e5"]])
    pve.add_transition("b",["plant","environment"],[["g2","g3"],["e1","e5","e6"]],[["g3","g3"],["e2","e7","e6"]])
    pve.add_transition("c",["plant","environment"],[["g2","g4"],["e1","e4","e7"]],[["g4","g4"],["e3","e6","e7"]])

    pve.create_RNN()
    pve.reinitialize()
    return pve