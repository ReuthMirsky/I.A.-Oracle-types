

from system import *

# choose the most successes path
def create(model):
    print("choice3")
    plant = process("plant",["g1","g2","g3"],[],[],"g1",update_states = ["g1"])
    environment = process("environment",["e1","e2","e3","e4","e5","e6"],[],[],"e1")
    
    pve = plant_environment("syst",plant,environment,model = model)
    
    pve.add_transition("a",["plant","environment"],[["g1"],["e1"]],[["g3"],["e2"]])
    pve.add_transition("b",["plant","environment"],[["g1"],["e1"]],[["g2"],["e5"]])
    pve.add_transition("c",["plant","environment"],[[],["e2","e5"]],[[],["e3","e6"]])
    pve.add_transition("d",["plant","environment"],[["g3"],["e3"]],[["g2"],["e4"]])
    pve.add_transition("e",["plant","environment"],[["g2"],["e4","e6"]],[["g1"],["e1","e1"]])

    pve.create_RNN()
    pve.reinitialize()
    return pve


# =============================================================================
# # choose the least failures path
# def create(model):        
#     plant = process("plant",["g1","g2"],[],[],"g1")
#     environment = process("environment",["e1","e2","e3","e4"],[],[],"e1")
#     
#     pve = plant_environment("syst",plant,environment,model = model)
#     
#     pve.add_transition("a",["plant","environment"],[["g1"],["e1"]],[["g2"],["e2"]])
#     pve.add_transition("b",["plant","environment"],[["g1"],["e1"]],[["g2"],["e3"]])
#     pve.add_transition("c",["plant","environment"],[[],["e3"]],[[],["e4"]])
#     pve.add_transition("d",["plant","environment"],[["g2"],["e2","e4"]],[["g1"],["e1","e1"]])
# 
#     pve.create_RNN()
#     pve.reinitialize()
#     return pve
# =============================================================================
