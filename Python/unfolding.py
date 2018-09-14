import numpy as np

def unfold3D_ref(tensor): # Using numpy library for reference
    X1 = np.moveaxis(tensor, 0, 0).reshape(tensor.shape[0], -1)
    X2 = np.moveaxis(tensor, 1, 0).reshape(tensor.shape[1], -1)
    X3 = np.moveaxis(tensor, 2, 0).reshape(tensor.shape[2], -1)
    return X1, X2, X3

def unfold3D(tensor):
    dim1 = tensor.shape[0] # Rows
    dim2 = tensor.shape[1] # Columns
    dim3 = tensor.shape[2] # Matrices
    X1 = np.zeros([dim1, dim2*dim3])
    X2 = np.zeros([dim2, dim1*dim3])
    X3 = np.zeros([dim3, dim1*dim2])
    x1_j, x2_i, x3_i, x3_j, mat = 0, 0, 0, 0, 0
    
    for k in range(0, dim3):
        for i in range(0, dim1):
            for j in range(0, dim2):
                X1[i, x1_j+mat] = tensor[i, j ,k]
                X2[j, x2_i+mat] = tensor[i, j, k]
                X3[x3_i, x3_j] = tensor[i, j, k]
                x1_j += dim3
                x3_j += 1
            
            x1_j = 0
            x2_i += dim3
        
        x2_i = 0
        x3_i += 1
        x3_j = 0
        mat += 1
        
    return X1, X2, X3


# Create a test tensor
tensor = np.zeros([3, 4, 5]) # Row, column, matrix
for k in range(0, 5): # For every matrix
    for i in range(0, 3): # For every row
        for j in range(0, 4): # For every column
            value = str(i) + str(j) + str(k)
            tensor[i][j][k] = int(value)
            
            
# Create reference unfoldings
X1_ref, X2_ref, X3_ref = unfold3D_ref(tensor)

# Test the unfolding algorithm
X1, X2, X3 = unfold3D(tensor)

# Print settings
np.set_printoptions(linewidth=10000, suppress=False, formatter={'all': lambda x: "({0:03d})".format(int(x))})

print("This is our tensor with shape = {0}:\n".format(tensor.shape))
for i in range(0, tensor.shape[2]):
    print(tensor[:,:,i])
    print("")

print("This is our first reference unfolding with shape = {0}:\n".format(X1_ref.shape))
print(X1_ref)
print("\nThis is our first unfolding with shape = {0}:\n".format(X1.shape))
print(X1)

print("\nThis is our second reference unfolding with shape = {0}:\n".format(X2_ref.shape))
print(X2_ref)
print("\nThis is our first unfolding with shape = {0}:\n".format(X2.shape))
print(X2)

print("\nThis is our third reference unfolding with shape = {0}:\n".format(X3_ref.shape))
print(X3_ref)
print("\nThis is our first unfolding with shape = {0}:\n".format(X3.shape))
print(X3)

print("\n")

if np.array_equal(X1, X1_ref):
    print("X1 -> correct")
else:
    print("X1 -> wrong")
    
if np.array_equal(X2, X2_ref):
    print("X2 -> correct")
else:
    print("X2 -> wrong")
    
if np.array_equal(X3, X3_ref):
    print("X3 -> correct")
else:
    print("X3 -> wrong")




##### OUTPUT:

'''

This is our tensor with shape = (3, 4, 5):

[[(000) (010) (020) (030)]
 [(100) (110) (120) (130)]
 [(200) (210) (220) (230)]]

[[(001) (011) (021) (031)]
 [(101) (111) (121) (131)]
 [(201) (211) (221) (231)]]

[[(002) (012) (022) (032)]
 [(102) (112) (122) (132)]
 [(202) (212) (222) (232)]]

[[(003) (013) (023) (033)]
 [(103) (113) (123) (133)]
 [(203) (213) (223) (233)]]

[[(004) (014) (024) (034)]
 [(104) (114) (124) (134)]
 [(204) (214) (224) (234)]]

This is our first reference unfolding with shape = (3, 20):

[[(000) (001) (002) (003) (004) (010) (011) (012) (013) (014) (020) (021) (022) (023) (024) (030) (031) (032) (033) (034)]
 [(100) (101) (102) (103) (104) (110) (111) (112) (113) (114) (120) (121) (122) (123) (124) (130) (131) (132) (133) (134)]
 [(200) (201) (202) (203) (204) (210) (211) (212) (213) (214) (220) (221) (222) (223) (224) (230) (231) (232) (233) (234)]]

This is our first unfolding with shape = (3, 20):

[[(000) (001) (002) (003) (004) (010) (011) (012) (013) (014) (020) (021) (022) (023) (024) (030) (031) (032) (033) (034)]
 [(100) (101) (102) (103) (104) (110) (111) (112) (113) (114) (120) (121) (122) (123) (124) (130) (131) (132) (133) (134)]
 [(200) (201) (202) (203) (204) (210) (211) (212) (213) (214) (220) (221) (222) (223) (224) (230) (231) (232) (233) (234)]]

This is our second reference unfolding with shape = (4, 15):

[[(000) (001) (002) (003) (004) (100) (101) (102) (103) (104) (200) (201) (202) (203) (204)]
 [(010) (011) (012) (013) (014) (110) (111) (112) (113) (114) (210) (211) (212) (213) (214)]
 [(020) (021) (022) (023) (024) (120) (121) (122) (123) (124) (220) (221) (222) (223) (224)]
 [(030) (031) (032) (033) (034) (130) (131) (132) (133) (134) (230) (231) (232) (233) (234)]]

This is our first unfolding with shape = (4, 15):

[[(000) (001) (002) (003) (004) (100) (101) (102) (103) (104) (200) (201) (202) (203) (204)]
 [(010) (011) (012) (013) (014) (110) (111) (112) (113) (114) (210) (211) (212) (213) (214)]
 [(020) (021) (022) (023) (024) (120) (121) (122) (123) (124) (220) (221) (222) (223) (224)]
 [(030) (031) (032) (033) (034) (130) (131) (132) (133) (134) (230) (231) (232) (233) (234)]]

This is our third reference unfolding with shape = (5, 12):

[[(000) (010) (020) (030) (100) (110) (120) (130) (200) (210) (220) (230)]
 [(001) (011) (021) (031) (101) (111) (121) (131) (201) (211) (221) (231)]
 [(002) (012) (022) (032) (102) (112) (122) (132) (202) (212) (222) (232)]
 [(003) (013) (023) (033) (103) (113) (123) (133) (203) (213) (223) (233)]
 [(004) (014) (024) (034) (104) (114) (124) (134) (204) (214) (224) (234)]]

This is our first unfolding with shape = (5, 12):

[[(000) (010) (020) (030) (100) (110) (120) (130) (200) (210) (220) (230)]
 [(001) (011) (021) (031) (101) (111) (121) (131) (201) (211) (221) (231)]
 [(002) (012) (022) (032) (102) (112) (122) (132) (202) (212) (222) (232)]
 [(003) (013) (023) (033) (103) (113) (123) (133) (203) (213) (223) (233)]
 [(004) (014) (024) (034) (104) (114) (124) (134) (204) (214) (224) (234)]]


X1 -> correct
X2 -> correct
X3 -> correct

'''



