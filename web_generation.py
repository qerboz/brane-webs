#PARAMETERS TO SET BEFORE RUNNING
N_LEGS = 4 #number of legs
MIN_CHARGE, MAX_CHARGE = -4,4 #min and max charges for the initial 7-branes (will change with the transformations)
MAX_MULTIPLICITY = 3 #max initial multiplicity of 5-branes connected to the same 7-brane (will change with the transformations)
MAX_RANGE = 1000
N_CLASSES = 10 #number of strong equivalence classes
N_EQUIVALENT = 10000 #number of equivalent webs per class
SAVED_DATA = ['web','class'] #choose the data to be saved: 
                             #web, class, self-intersection, monodromy, total-monodromy, total-monodromy-trace, charge-invariant
FILENAME = 'dataset_10_classes.pkl' #name of the file to be saved with .pkl at the end

import numpy as np
import pandas as pd
from copy import deepcopy
np.random.seed(0)

S, T = np.array([[0,-1],[1,0]]), np.array([[0,-1],[1,1]]) #two useful matrices

def generate_web():
    '''Function to generate a dict containing a web and properties useful to its classification: the self-intersection
    number, the monodromy matrices, the total monodromy matrix and its trace, the charge invariant 
    Output: a dict containing - 'web': tuple of size 2*N_LEGS containing int       
                              - 'self-intersection': int
                              - 'monodromy': a np array of shape (N_LEGS,2,2) 
                              - 'total-monodromy': tuple of size N_LEGS containing int
                              - 'total-monodromy-trace': int
                              - 'charge-invariant': int'''
    web_dict = {}
    web = np.zeros((N_LEGS,2)) #initial definition of the web, to be replaced by random numbers
    while 0 in web[:,0]**2 + web[:,1]**2:
        web = np.random.randint(MIN_CHARGE,MAX_CHARGE+1,size=(N_LEGS,2)) #generation of the charges
        n_branes = np.random.randint(MAX_MULTIPLICITY,size=(N_LEGS,1)) #generation of the multiplicity
        web = n_branes*web
        web[-1,:] = np.zeros((2,)) - web[:-1,:].sum(axis=0) #ensuring a null total charge
    web = web.take(np.argsort(np.arctan2(web[:,1],web[:,0])),axis=0) #sort the branes in anticlockwise order from -pi
    n_branes = np.gcd(web[:,0],web[:,1]) #multiplicity corrected with the gcd of the charges
    det = web@S@(web.T) 
    I = np.abs(det[np.triu_indices(N_LEGS)].sum())-np.dot(n_branes,n_branes) #self-intersection
    if I > -2: #check supersymmetry
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
        reduced_web = (web/n_branes.reshape((4,1))).astype('int')
        det = reduced_web@S@(reduced_web.T)
        l = np.gcd.reduce(det.reshape(-1)) #charge invariant
        web_dict['charge-invariant'] = l
        return web_dict
    else: #if web not supersymmetric, try again
        return generate_web()

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
    row = generate_web() #create the first web of the class
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
dataset[SAVED_DATA].to_pickle(FILENAME) #save to pickle format