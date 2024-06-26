
# Importing necessary modules
import argparse
import numpy as np
import pandas as pd
from copy import deepcopy

# Argument parser for command-line arguments
parser = argparse.ArgumentParser(description="Generate brane webs")
parser.add_argument("--n_legs", type=int, default=4, help="Number of legs")
parser.add_argument("--min_charge", type=int, default=-4, help="Minimum charge for the initial 7-branes")
parser.add_argument("--max_charge", type=int, default=4, help="Maximum charge for the initial 7-branes")
parser.add_argument("--max_multiplicity", type=int, default=3, help="Maximum initial multiplicity of 5-branes connected to the same 7-brane")
parser.add_argument("--max_range", type=int, default=1000, help="Maximum range")
parser.add_argument("--n_classes", type=int, default=10, help="Number of strong equivalence classes")
parser.add_argument("--n_equivalent", type=int, default=10000, help="Number of equivalent webs per class")

args = parser.parse_args()

# Parameters
N_LEGS = args.n_legs
MIN_CHARGE, MAX_CHARGE = args.min_charge, args.max_charge
MAX_MULTIPLICITY = args.max_multiplicity
MAX_RANGE = args.max_range
N_CLASSES = args.n_classes
N_EQUIVALENT = args.n_equivalent
FILENAME = f"dataset_{N_LEGS}_legs_{N_CLASSES}_classes_.pkl"


np.random.seed(0)

S, T = np.array([[0,-1],[1,0]]), np.array([[0,-1],[1,1]]) #two useful matrices

def generate_web():
    while True:
        matrix = [] # Initialize an empty list to store the rows
        while len(matrix) < N_LEGS - 1: # Generate a new row within the bounds
            row = np.random.randint(MIN_CHARGE, MAX_CHARGE + 1, 2)
            if np.all(row == 0): # Check if the row is (0, 0)
                continue
            aligns = False # Check if the row aligns with any of the previously added rows
            for existing_row in matrix:
                if row[0] * existing_row[1] == row[1] * existing_row[0]:
                    aligns = True
                    break
            if not aligns: # If the row does not align with any existing rows, add it to the matrix
                n_branes = np.random.randint(1,MAX_MULTIPLICITY)
                matrix.append(n_branes*row)
        last_row = -np.sum(matrix, axis=0) # Calculate the last row such that the sum of each column is zero
        if np.all(last_row == 0): # Check if the last row is (0, 0) or aligns with any existing rows
            continue
        aligns = False
        for existing_row in matrix:
            if last_row[0] * existing_row[1] == last_row[1] * existing_row[0]:
                aligns = True
                break
        if aligns:
            continue
        # If the last row meets the criteria, append it to the matrix and break the loop
        matrix.append(last_row)
        web = np.array(matrix)
        web = web.take(np.argsort(np.arctan2(web[:,1],web[:,0])),axis=0) #sort the branes in anticlockwise order from -pi
        n_branes = np.gcd(web[:,0],web[:,1]) #multiplicity corrected with the gcd of the charges
        det = web@S@(web.T) 
        I = np.abs(det[np.triu_indices(N_LEGS)].sum())-np.dot(n_branes,n_branes) #self-intersection
        if I > -2:
            break
    return web

def generate_web_dict(web):
    '''Function to generate a dict containing a web and properties useful to its classification: the self-intersection
    number, the monodromy matrices, the total monodromy matrix and its trace, the charge invariant
    Input: a web as an np array of shape ((N_LEGS,2) containing int
    Output: a dict containing - 'web': tuple of size 2*N_LEGS containing int       
                              - 'self-intersection': int
                              - 'monodromy': a np array of shape (N_LEGS,2,2) 
                              - 'total-monodromy': tuple of size N_LEGS containing int
                              - 'total-monodromy-trace': int
                              - 'charge-invariant': int'''
    web_dict = {}
    n_branes = np.gcd(web[:,0],web[:,1]) #multiplicity corrected with the gcd of the charges
    det = web@S@(web.T) 
    I = np.abs(det[np.triu_indices(N_LEGS)].sum())-np.dot(n_branes,n_branes) #self-intersection
    web_dict['web'] = tuple(web.flatten())
    web_dict['self-intersection'] = I
    TM = np.identity(2,dtype='int') #total monodromy
    M = np.zeros((N_LEGS,2,2),dtype='int') #monodromy matrices
    for j in range(N_LEGS): #compute monodromy of each brane
        charges = web[j].reshape((2,1))
        coprime_charges = np.array(charges/n_branes[j],dtype='int')
        M[j,:,:] = monodromy_matrix(coprime_charges)
        TM = TM@M[j,:,:]
    web_dict['monodromy'] = M
    web_dict['total-monodromy-trace'] = np.trace(TM)
    web_dict['total-monodromy'] = tuple(TM.flatten())
    reduced_web = (web/n_branes.reshape((N_LEGS,1))).astype('int')
    det = reduced_web@S@(reduced_web.T)
    l = np.gcd.reduce(det.reshape(-1)) #charge invariant
    web_dict['charge-invariant'] = l
    return web_dict


def monodromy_matrix(charge):
    '''Function to compute the monodromy matrix of a charge tuple (p,q)
    Input: np.array of shape (2,)
    Output: np.array of shape (2,2)'''
    return np.identity(2,dtype='int')-charge@charge.T@S

def generate_transformation():
    '''Function that provides a random transformation by returning a SL(2,Z) matrix and a choice of leg for a HW move
    Input: int (number of legs of our webs)
    Output: tuple (np (2,2)-array, int between 0 and N_LEGS-1)'''
    sl_mat = np.identity(2,dtype='int') #start with the identity and multiply it by a random sequence of SL(2,Z) generators
    n = np.random.randint(40)
    sequence = np.random.randint(0,2,size=(n))
    for i in range(n):
        if sequence[i]:
            sl_mat = S@sl_mat
        else: 
            sl_mat = T@sl_mat
    hw_move = np.random.randint(0,N_LEGS) #generate a random HW-move
    return sl_mat, hw_move

def apply_transformation(row, sl_mat, hw_move):
    '''Function that transforms a dict or pd.Series with a given Hanany-Witten move and SL(2,Z) matrix
    Input: - dict containing all necessary data as defined in generate_web()
           - np.array of shape (2,2) in SL(2,Z)
           - int between 0 and N_LEGS
    Output: - dict as in generate_web()'''
    new_row = deepcopy(row)
    new_web = np.array(new_row['web']).reshape((N_LEGS,2))
    new_mon = deepcopy(row['monodromy'])
    new_web = np.roll(new_web,-hw_move,axis=0) #rotate web to place HW-moved leg first
    chosen_leg = new_web[0,:]
    index = 0
    while np.arctan2(chosen_leg[0]*new_web[index+1,1]-chosen_leg[1]*new_web[index+1,0],np.dot(chosen_leg,new_web[index+1,:])) > 0: #sweep leg anticlockwise by pi rad
        index += 1
    new_web[:index,:] = new_web[1:index+1,:]@(row['monodromy'][hw_move,:,:].T) #update all sweeped legs
    new_web[index,:]= -new_web[:index,:].sum(axis=0)-new_web[index+1:,:].sum(axis=0) #update HW-moved leg
    if (new_web[index,:] == np.array([0,0])).all(): #return None if the leg detached
        return None
    new_web = np.roll(new_web,hw_move,axis=0) #rotate back
    new_web = new_web@sl_mat #apply SL(2,Z) matrix
    if np.max(new_web) > MAX_RANGE or np.min(new_web) < -MAX_RANGE: #return None if new web too large
        return None
    n_branes = np.gcd(new_web[:,0],new_web[:,1])
    TM = np.identity(2,dtype='int')
    for j in range(N_LEGS):
        charge = new_web[j].reshape((2,1))
        coprime_charges = np.array(charge/n_branes[j],dtype='int')
        new_mon[j,:,:] = monodromy_matrix(coprime_charges)
        TM = TM@new_mon[j,:,:]
    new_row['web'] = tuple(new_web.flatten())
    new_row['monodromy'] = new_mon 
    new_row['total-monodromy'] = tuple(TM.flatten())
    return new_row

def transform_df(df):
    '''Function that applies a random transformation to a dataframe and returns the transformed dataframe
    Input: - pd.DataFrame with columns from generate_web()
    Output: - pd.DataFrame'''
    new_df = pd.DataFrame(columns=['web','self-intersection','monodromy','total-monodromy','total-monodromy-trace','charge-invariant'])
    sl_mat, hw_move = generate_transformation() #choose random transformation to apply
    for _, row in df.iterrows():
        new_row = apply_transformation(row, sl_mat, hw_move)
        if new_row is not None:
            new_df = pd.concat([new_df,pd.Series(new_row).to_frame().T],ignore_index=True)
    return new_df

dataset = pd.DataFrame(columns=['web','self-intersection','monodromy','total-monodromy','total-monodromy-trace','charge-invariant','class'])

class_id = 0
while len(dataset.groupby('class')) < N_CLASSES: #while not enough classes in the dataframe, add new ones
    web = generate_web() #create the first web of the class
    row = generate_web_dict(web)
    row['class'] = class_id
    class_dataset = pd.Series(row).to_frame().T
    if len(pd.merge(dataset,class_dataset,on=['self-intersection','total-monodromy-trace','charge-invariant'])) > 0: #check that there are no webs with same invariants
        continue
    class_id += 1
    count = 0
    while len(class_dataset) < N_EQUIVALENT: #while number of equivalent branes in the class not enough, add equivalent branes
        if count > N_EQUIVALENT: #safe stop for the while loop: if too long, discard this class (happens when web grows too fast)
            class_id -= 1
            break
        new_df = transform_df(class_dataset) #apply transformation to all branes of the class (doubles the size of the class)
        class_dataset = pd.concat([class_dataset,new_df]).drop_duplicates(subset=['web']) #concatenate and discard duplicate webs
        count += 1
    if len(class_dataset) >= N_EQUIVALENT: #if class large enough, add it to dataset, otherwise discard and create new class
        dataset = pd.concat([dataset,class_dataset])
dataset.reset_index(drop=True)
dataset.to_pickle('datasets/'+FILENAME) #save to pickle format