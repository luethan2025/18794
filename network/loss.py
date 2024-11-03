import torch.nn.functional as F

def compute_occ_loss(W, Z, GT):
    return F.binary_cross_entropy(W * Z, GT)

def compute_loss(L_detect, L_occluder_B, L_occluder_S, L_occludee_B, L_occludee_S, learning_rates):
    eta1, eta2, eta3, eta4, eta5 = learning_rates
    return eta1 * L_detect + \
           eta2 * L_occluder_B + \
           eta3 * L_occluder_S + \
           eta4 * L_occludee_B + \
           eta5 * L_occludee_S
