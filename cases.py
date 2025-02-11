


from system import *

#cases from Doron's paper
def create(model):
    print("cases")
    plant = process("plant",["g1","g2"],[],[],"g1",update_states=["g2"])
    environment = process("environment",["e1","e2","e3","e4","e5"],[],[],"e1")
    
    pve = plant_environment("syst",plant,environment,model = model)
    
    pve.add_transition("a",["plant","environment"],[["g1"],["e2","e5"]],[["g2"],["e4","e3"]])
    pve.add_transition("b",["plant","environment"],[["g2"],["e1","e4"]],[["g1"],["e2","e2"]])
    pve.add_transition("c",["plant","environment"],[["g2"],["e1","e3"]],[["g1"],["e3","e5"]])

    pve.create_RNN()
    pve.reinitialize()
    return pve
