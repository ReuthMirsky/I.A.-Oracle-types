

from system import *

#first easy example
def create(model):
    print("cycle_abc")
    plant = process("plant",["p0"],[],[],"p0")
    environment = process("environment",["e0","e1","e2"],[],[],"e0")
    
    pve = plant_environment("syst",plant,environment,model = model)
    
    pve.add_transition("a",["plant","environment"],[["p0"],["e0"]],[["p0"],["e1"]])
    pve.add_transition("b",["plant","environment"],[["p0"],["e1"]],[["p0"],["e2"]])
    pve.add_transition("c",["plant","environment"],[["p0"],["e2"]],[["p0"],["e0"]])

    pve.create_RNN()
    pve.reinitialize()
    return pve