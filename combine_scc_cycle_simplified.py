


from system import *


def create(model):        
    plant = process("plant",["p0","p1"],[],[],"p0")
    environment = process("environment",["e"+str(i) for i in range(18)],[],[],"e0")
    
    pve = plant_environment("syst",plant,environment,model = model)
    
    pve.add_transition("a",["plant","environment"],[["p0"],["e0"]],[["p1"],["e1"]])
    pve.add_transition("b",["plant","environment"],[["p0"],["e0"]],[["p1"],["e7"]])
    pve.add_transition("c",["plant","environment"],[["p0"],["e0"]],[["p1"],["e13"]])
    pve.add_transition("f",["plant","environment"],[[],["e"+str(i) for i in [1,2,3,7,8,10,13,16,17]]],[[],["e"+str(i) for i in [2,3,4,8,9,11,14,17,18]]])
    pve.add_transition("x",["plant","environment"],[["p1"],["e"+str(i) for i in [4,9,14,15]]],[["p1"],["e"+str(i) for i in [5,10,15,16]]])
    pve.add_transition("y",["plant","environment"],[["p1"],["e"+str(i) for i in [5,11,9,14,15]]],[["p1"],["e"+str(i) for i in [6,12,10,15,16]]])
    pve.add_transition("z",["plant","environment"],[["p1"],["e"+str(i) for i in [6,12,18,9,14,15]]],[["p1"],["e"+str(i) for i in [4,10,16,10,15,16]]])

    pve.create_RNN()
    pve.reinitialize()
    return pve