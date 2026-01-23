############
# Packages #
############
import numpy as np
from matplotlib.colors import ListedColormap


#############
# functions #
#############
'''
linear path of the zero-start protocol
'''
def linear_path0(a_i, b_i, a_f, b_f, s_temp):
    a = a_i + (a_f - a_i) * np.tanh(s_temp)
    b = b_i + (b_f - b_i) * np.tanh(s_temp) 
    
    return a, b  


'''
Compute the stable fixed points of henon map period 1 directly with formula
Input:
    a: (float) parrameter a
    b: (float) parrameter b
Output:
    x: x-coordinate of the fixed point
    y: y-coordinate of the fixed point
'''
def direct_compute_stable_fixed_point(a, b):
    disc = max((1 - b)**2 + 4 * a, 0)
    if disc < 0 or a == 0:
        return None, None
    x_fp = (-(1 - b) + np.sqrt(disc)) / (2 * a)
    y_fp = b * x_fp
    return x_fp, y_fp


"""
Compute the next value of the henon map, with detection whether any orbit goes to infinity
Input:
    a: (float) parrameter a
    b: (float) parrameter b
    x_current: (float) the current value of x
    y_current: (float) the current value of y
    infinity_tipping_detected: (bool) whether it detects values go to infinity
    MAX_VAL: (float) threshold to indicate the mao is going to infinity
Output:
    x_next: (float) the next value of x
    y_next: (float) the next value of y
"""
def henon_map_next(a, b, x_current, y_current, MAX_VAL, infinity_tipping_detected): 
    x_next = 1 - a * x_current**2 + y_current
    y_next = b * x_current

    # Works with both scalar and array input
    overflow = (
        ~np.isfinite(x_next) | ~np.isfinite(y_next) |
        (np.abs(x_next) > MAX_VAL) | (np.abs(y_next) > MAX_VAL)
    )
    
    if np.any(overflow):
        x_next = np.where(overflow, np.nan, x_next)
        y_next = np.where(overflow, np.nan, y_next)
        infinity_tipping_detected = True

    return x_next, y_next, infinity_tipping_detected


'''
Input a value of r, see whether the system will tip with the given functions from (a_i, b_i) to (a_f, b_f)
Output:
    True: if the input r value inducing tipping to inifinity
    False: if not
    None: ambiguous case, tip to other orbits
'''
def r_inducing_tipping(func, a_i, b_i, a_f, b_f, r, x0, y0, 
                       N=1500, MAX_VAL=1e4, div_steps=60, diverge_threshold=1e-2,
                       converge_threshold=1e-4, converge_steps=30):
    
    min_steps = min(N-converge_steps, N-div_steps)

    x, y = x0, y0
    dist_ls = []
    escaped = False
    x_fp, y_fp = direct_compute_stable_fixed_point(a_f, b_f)
    
    # N_prev = 15    # for the infinite-start protocol, decide by a test round
    N_prev = 0 # zero-start protocol
    for n in range(-N_prev, N):
        if np.isinf(r):
            a, b = a_f, b_f
        else:
            s_temp = r * n
            a, b = func(a_i, b_i, a_f, b_f, s_temp)
            
        x, y, escaped = henon_map_next(a, b, x, y, MAX_VAL, escaped)
        if escaped or not np.isfinite(x) or not np.isfinite(y):
            return True

        # x_fp, y_fp = direct_compute_stable_fixed_point(a, b) # if using end-point check definition
        # if x_fp is None and y_fp is None:
        #     continue

        dist = np.hypot(x - x_fp, y - y_fp)
        if not np.isfinite(dist) or dist > MAX_VAL:
            return True
        
        # ignore earlier steps
        if n < min_steps:
            continue
        
        dist_ls.append(dist)
        
    # Check convergence
    if len(dist_ls) >= converge_steps:
        recent = np.array(dist_ls[-converge_steps:])
        if all(d < converge_threshold for d in recent):
            return False # strict convergence - so r does not induced tipping

        k = converge_steps // 3
        if k >= 3:
            is_converging = (
                np.mean(recent[0:k]) > np.mean(recent[k:2*k]) > np.mean(recent[2*k:])
            )
            if is_converging:
                return False
            
    # Check divergence  
    if len(dist_ls) >= div_steps:
        recent = np.array(dist_ls[-div_steps:])
        
        # just in case it moves away and then come back
        if k >= 3: # too few points to decide
            m1 = np.mean(recent[0:k])
            m2 = np.mean(recent[k:2*k])
            m3 = np.mean(recent[2*k:])
            all_large = (
                m1 > diverge_threshold and
                m2 > diverge_threshold and
                m3 > diverge_threshold
            )
            margin, factor = 1.05, 2
            strickly_increasing = (m1 * margin < m2) and (m2 * margin < m3)
            overall_increase = m3 > factor * m1
            if all_large and strickly_increasing and overall_increase:
                return True

    return None


'''
Check whether the fixed point (x_i, y_i) is inside the basin of attraction of (x_f, y_f)
Input:
    a_f, b_f: (float) the (a_f, b_f) that generates (x_f, y_f)
    x_i, y_i: the initial condition
    x_f, y_f: the end stable fixed point
Output:
    True: is inside
    False: not inside
    None: tip to other orbits
'''
def is_in_basin_of_attraction(
    a_f, b_f, x_i, y_i, x_f, y_f,
    N=1500, MAX_VAL=1e3,
    converge_threshold=1e-4, converge_steps=30,
    diverge_threshold=1e-2, diverge_steps=60
):
    
    min_steps = min(N-converge_steps, N-diverge_steps)
    x, y = x_i, y_i
    distances = []
    escaped = False
    for i in range(N):
        x, y, escaped = henon_map_next(a_f, b_f, x, y, MAX_VAL, escaped)

        if escaped or not np.isfinite(x) or not np.isfinite(y):
            return False

        dist = np.hypot(x - x_f, y - y_f)
        if not np.isfinite(dist) or dist > MAX_VAL:
            return False
        
        # ignore earlier steps
        if i < min_steps:
            continue
        
        distances.append(dist)


    # --- Check convergence ---
    if len(distances) >= converge_steps:
        recent = np.array(distances[-converge_steps:])
        if all(d < converge_threshold for d in recent):
            return True  # strict convergence

        k = converge_steps // 3
        if k >= 3:
            is_converging = (
                np.mean(recent[0:k]) > np.mean(recent[k:2*k]) > np.mean(recent[2*k:])
            )

            if is_converging:
                return True

            
    # --- Check diverge/tip to infinity ---
    if len(distances) >= diverge_steps:
        recent = np.array(distances[-diverge_steps:])
        
        # just in case it moves away and then come back
        k = diverge_steps // 3
        if k >= 3: # too few points to decide
            m1 = np.mean(recent[0:k])
            m2 = np.mean(recent[k:2*k])
            m3 = np.mean(recent[2*k:])
            all_large = (
                m1 > diverge_threshold and
                m2 > diverge_threshold and
                m3 > diverge_threshold
            )
            margin, factor = 1.05, 2
            strickly_increasing = (m1 * margin < m2) and (m2 * margin < m3)
            overall_increase = m3 > factor * m1
            if all_large and strickly_increasing and overall_increase:
                return False
    
    # Neither converge nor diverge to infinity: period3, period6, 3band, etc.
    return None


"""
Find the jacobian matix for the henon map evaluted at point (x0, y0).
Input:
    a: (float) parrameter a
    b: (float) parrameter b
    x: (float) value of x
    y: (float) value of y
    period: (int) period of the map
Output:
    jac: (np array, float) a list of x values after discarding n_transient points
"""
def henon_jacobian(x0, y0, a, b):
    jac = np.array([[-2 * a * x0, 1], # the jacobian matrix of period 1 henon map
                     [b, 0]])
    return jac


"""
Check whether a fixed point (x, y) is stable.
Input:
    a: (float) parrameter a
    b: (float) parrameter b
    x: (float) value of x
    y: (float) value of y
Output:
    True if all |lambda| < 1; otherwise False
"""
def is_stable(x, y, a, b):
    J = henon_jacobian(x, y, a, b)
        
    eigenvals = np.linalg.eigvals(J)

    margin = 1e-10
    return np.all(np.abs(eigenvals) < 1 - margin)


"""
Select certain number of (a, b) pair from the parameter space
Input:
    b_min: (float) the minimum value of b
    b_max: (float) the maximum value of b
    a_min: (float) the minimum value of a; 
            defaule need to use None, means the whole parameter a ranging based on b,
            if you want to fill the whole parameter space;
            do not feed in the a even if you have computed it
    a_max: (float) the maximum value of a
    n_b: number of b values
    n_a: number of a values. So in total, get n_b*n_a point
Output:
    ab_list: (list of tuples) a list of (a, b) pairs
"""
def select_ab_list(b_min, b_max, n_b, n_a, a_min=None, a_max=None):
    n_b = int(n_b)
    n_a = int(n_a)

    ab_list = []

    if b_min == b_max: # select a fixed b
        b_list =[b_min]
    else:
        b_list = np.linspace(b_min, b_max, num=n_b+2)[1:-1] # do not consider the end because it may be too close to bifurcation

    if a_min is not None and a_max is not None: 
        a_list = np.linspace(a_min, a_max, num=n_a+2)[1:-1]

    eps=1e-5
    # for each b, compute range of a
    for b in b_list:
        if a_min is None and a_max is None:
            a_min_local= -(1-b)**2/4 + eps
            a_max_local = 3*(1-b)**2/4 - eps

            a_list = np.linspace(a_min_local, a_max_local, num=n_a+2)[1:-1]

        for a in a_list:
            # and make sure the fixed point is stable
            x_fp, y_fp = direct_compute_stable_fixed_point(a, b)
            if x_fp is not None and y_fp is not None and is_stable(x_fp, y_fp, a, b):  
                # make sure derminant works
                des = (1-b)**2 + 4*a # descriminant of x+
                if des >=0 and a != 0:
                    ab_list.append( (a, b) )
    
    return ab_list


"""
Given a list of initial (a_i,b_i) pairs, 
figure out whether there exist other (a_f, b_f) pairs in the whole parameter space, 
such that the fixed point of (a_i, b_i) do not converge to the fixed points of (a_f, b_f) 
- and r tipping can happen from (a_i, b_i) to (a_f, b_f); 
also return those converge pairs 
(i.e. in the basin of attraction, is_in_basin() returns True), or ambiguous pairs (is_in_basin() returns None)
Input:
    ab_list: a list of (a, b) pairs [(a, b), ...]
    b_min: (float) the minimum value of b
    b_max: (float) the maximum value of b
    a_min: (float) the minimum value of a; defaule None, means the whole parameter range based on b
    a_max: (float) the maximum value of a
    n_b: number of b values
    n_a: number of a values. So in total, get n_b*n_a point
Output:
    r_tipping_ab_pairs: (dict with (a_i, b_i) as key, list of tuples [(a_f, b_f), ...] as values)
    converge_ab_pairs: (dict with (a_i, b_i) as key,
"""
def find_fp_possible_r_tipping_ab(ab_list, b_min, b_max, a_min=None, a_max=None, n_b=30, n_a=30):
    n_b = int(n_b)
    n_a = int(n_a)    
    
    tipping_infty_ab_pairs = {}
    converge_ab_pairs = {} 
    none_ab_pairs = {} 
    notSufficient_ab_pairs = {}
    
    ab_full_list = select_ab_list(b_min, b_max, n_b, n_a, a_min, a_max)

    for a_i, b_i in ab_list:
        tipping_infty_ab_pairs[(a_i, b_i)] = []
        # tipping_period3_ab_pairs[(a_i, b_i)] = []
        converge_ab_pairs[(a_i, b_i)] = []
        none_ab_pairs[(a_i, b_i)] = []
        notSufficient_ab_pairs[(a_i, b_i)] = []
        
        x_i, y_i = direct_compute_stable_fixed_point(a_i, b_i)
        if x_i is None or y_i is None:
            continue

        for a_f, b_f in ab_full_list:
            if abs(a_f - a_i) < 1e-6 and abs(b_f - b_i) < 1e-6:
                continue

            x_f, y_f = direct_compute_stable_fixed_point(a_f, b_f)
            if x_f is None or y_f is None:
                continue

            # calculate the actual initial point to find basin of attraction
            # n < 0, (a, b) = (a_i, b_i); n = 0, (a, b) = ( (a_i+a_f)/2, (b_i+b_f)/2 ); n>0, (a, b) = (a_f, b_f)
            # assume the case r = inf, the IC (x_i^*, y_i^*) is henon( (x_i, y_i),  (a_i+a_f)/2, (b_i+b_f)/2)
            # a_star = (a_i+a_f)/2 # protocal from -infty
            # b_star = (b_i+b_f)/2

            a_star = a_i # protocol from 0
            b_star = b_i

            MAX_VAL = 1e3
            infinity_tipping_detected = False

            # Z_1
            x_i_star, y_i_star, infinity_tipping_detected = henon_map_next(a_star, b_star, x_i, y_i, MAX_VAL, infinity_tipping_detected) # fixed ai, bi
            if x_i_star is None or y_i_star is None or infinity_tipping_detected:
                tipping_infty_ab_pairs[(a_i, b_i)].append((a_f, b_f))
                continue
            
            if a_star >= (3*(1-b_star)**2) /4 or a_star <= (- (1-b_star)**2) / 4: # add not sufficient case
                is_in_basin = -2 # -2 not sufficient
            else:
                # Z_2 is merged in is_in_basin_of_attraction()
                # since Z_2 = H_{af, bf}(Z_1)
                is_in_basin = is_in_basin_of_attraction(a_f, b_f, x_i_star, y_i_star, x_f, y_f) #fixed a_i, b_i

            if is_in_basin is False:
                tipping_infty_ab_pairs[(a_i, b_i)].append((a_f, b_f))

            elif is_in_basin is True:
                converge_ab_pairs[(a_i, b_i)].append((a_f, b_f))

            elif is_in_basin is None:
                none_ab_pairs[(a_i, b_i)].append((a_f, b_f))
            
            elif is_in_basin == -2:
                notSufficient_ab_pairs[(a_i, b_i)].append((a_f, b_f))

    return tipping_infty_ab_pairs, converge_ab_pairs, none_ab_pairs, notSufficient_ab_pairs


# a version fixed (af, bf) ranging (ai, bi)
def find_fp_possible_r_tipping_ab_reverse(ab_list, b_min, b_max, a_min=None, a_max=None, n_b=30, n_a=30):
    n_b = int(n_b)
    n_a = int(n_a)    
    
    tipping_infty_ab_pairs = {}
    converge_ab_pairs = {} 
    none_ab_pairs = {} 
    notSufficient_ab_pairs = {}
    
    ab_full_list = select_ab_list(b_min, b_max, n_b, n_a, a_min, a_max)

    for a_f, b_f in ab_list:
        tipping_infty_ab_pairs[(a_f, b_f)] = []
        converge_ab_pairs[(a_f, b_f)] = []
        none_ab_pairs[(a_f, b_f)] = []
        notSufficient_ab_pairs[(a_f, b_f)] = []
        
        x_f, y_f = direct_compute_stable_fixed_point(a_f, b_f)
        if x_f is None or y_f is None:
            continue

        for a_i, b_i in ab_full_list:
            if abs(a_f - a_i) < 1e-6 and abs(b_f - b_i) < 1e-6:
                continue

            x_i, y_i = direct_compute_stable_fixed_point(a_i, b_i)
            if x_i is None or y_i is None:
                continue

            a_star = a_i # protocol from 0 fixed af, bf
            b_star = b_i

            MAX_VAL = 1e3
            infinity_tipping_detected = False

            # Z_1
            x_i_star, y_i_star, infinity_tipping_detected = henon_map_next(a_star, b_star, x_i, y_i, MAX_VAL, infinity_tipping_detected) # fixed af, bf
            if x_i_star is None or y_i_star is None or infinity_tipping_detected:
                tipping_infty_ab_pairs[(a_f, b_f)].append((a_i, b_i))
                continue
            
            if a_star >= (3*(1-b_star)**2) /4 or a_star <= (- (1-b_star)**2) / 4: # add not sufficient case
                is_in_basin = -2 # -2 not sufficient
            else:
                # Z_2 is merged in is_in_basin_of_attraction()
                # since Z_2 = H_{af, bf}(Z_1)
                is_in_basin = is_in_basin_of_attraction(a_f, b_f, x_i_star, y_i_star, x_f, y_f)

            if is_in_basin is False:
                tipping_infty_ab_pairs[(a_f, b_f)].append((a_i, b_i))

            elif is_in_basin is True:
                converge_ab_pairs[(a_f, b_f)].append((a_i, b_i))

            elif is_in_basin is None:
                none_ab_pairs[(a_f, b_f)].append((a_i, b_i))
            
            elif is_in_basin == -2: 
                notSufficient_ab_pairs[(a_f, b_f)].append((a_i, b_i))

    return tipping_infty_ab_pairs, converge_ab_pairs, none_ab_pairs, notSufficient_ab_pairs


'''
Find out the label('tipping', 'converge', 'ambiguous') of the given (a_f, b_f) for the parameter space:
    - assume r is infty
    - iterate (x_i, y_i) once to get (x_i^*, y_i^*)
    - see whether (x_i^*, y_i^*) is in the basin of attraction of (a_f, b_f)
Input:
    x_i, x_i: (float) the initial fixed point from the fixed (a_i, b_i)
    x_f, y_f: (float) the fixed point from (a_f, b_f)
    a_i, b_i: (float) the initial fixed (a_i, b_i)
    a_f, b_f: (float) the chose (a_f, b_f)
Output:
    None: no label found because (x_f, y_f) contains None
    0: this is a tipping situation to the infinity
    1: a converge situation
    -1: an ambiguous situation
'''
def find_label_numeric_para(x_i, y_i, x_f, y_f, a_i, b_i, a_f, b_f):
    label = None 
   
    # a_star = (a_i+a_f)/2 # protocol -infty
    # b_star = (b_i+b_f)/2 

    a_star = a_i # protocol 0
    b_star = b_i

    MAX_VAL = 1e6       
    infinity_tipping_detected = False    
    x_i_star, y_i_star, infinity_tipping_detected = henon_map_next(a_star, b_star, x_i, y_i, MAX_VAL, infinity_tipping_detected)
    if infinity_tipping_detected:
        label = 0
        return label
    
    r = float('inf')
    x_i, y_i = direct_compute_stable_fixed_point(a_i, b_i)
    induce_tipping = r_inducing_tipping(linear_path0, a_i, b_i, a_f, b_f, r, x_i, y_i)

    if induce_tipping is True:
        label = 0  

    elif induce_tipping is False:
        label = 1
    
    elif induce_tipping is None:
        label = -1
    
    return label

'''
Figure out the track/tip numerical labels of the values in the given val_space for plotting
Input:
    val_space: a list of values of in the free parameter space
    val_fixed: the fixed parameter
    x_i, y_i: input (x_i, y_i) from the selected (a_i, b_i)
    a_i, b_i: input selected (a_i, b_i)
    a_f, b_f: input selected (a_f, b_f)
output:
    labels: a list of label which contains
        - 0: tipping
        - 1: track/converge to given fixed point
        - -1: ambiguous
'''
def find_numeric_label_ls(val_space:list, val_fixed: float, 
                          x_i:float, y_i:float, 
                          a_i:float, b_i:float, a_f:float, b_f:float,
                          g_type:str):

    labels = []
    for val in val_space:
        
        if g_type == 'para':
            x_f, y_f = direct_compute_stable_fixed_point(val, val_fixed) 
            if None in (x_f, y_f):
                    continue
            label = find_label_numeric_para(x_i, y_i, x_f, y_f, a_i, b_i, val, val_fixed)
                   
        elif g_type == 'basin':
            x_f, y_f = direct_compute_stable_fixed_point(a_f, b_f) 

            r = float('inf')
            x_i, y_i = direct_compute_stable_fixed_point(a_i, b_i)
            induce_tipping = r_inducing_tipping(linear_path0, a_i, b_i, a_f, b_f, r, val_fixed, val)
            if induce_tipping is True:
                label = 0
            elif induce_tipping is False:
                label = 1
            elif induce_tipping is None:
                label = -1
        
        elif g_type == 'r':
            induce_tipping = r_inducing_tipping(linear_path0, a_i, b_i, a_f, b_f, val, x_i, y_i)  
            if induce_tipping is True:
                label = 0   
            elif induce_tipping is False:
                label = 1   
            elif induce_tipping is None:
                label = -1

        labels.append(label)

    return labels

'''
Plot heat map of fractal branch cut with input data labels. 0 - tipping to infinity (green); 1 - converge (pink)
Input:
    labels: (list or array) The label values. 0 - converge; 1 - tip to infinity
    val_min, val_max: (float) the range limit of the free parameter
    ax: (matplotlib.axes.Axes) Optional. If provided, plot into this axes. Otherwise a new figure is created.
Output:
    fig, ax : Figure and Axes objects
'''
def plot_frac_bar(labels, val_min, val_max, ax=None):
    # cmap = ListedColormap(['orange', 'darkseagreen', 'mistyrose'])
    cmap = ListedColormap(['orange', 'yellow', 'blue'])
    # cmap = ListedColormap(['whitesmoke', 'whitesmoke', 'whitesmoke'])

    ax.imshow([labels],
              aspect="auto",
              interpolation="nearest",
              vmin=-1, vmax=1, # 0 maps to the yellow (tipping/not converge), and 1 map to blue (track)
              cmap=cmap,
              extent=[val_min, val_max, 0, 1])

    ax.set_yticks([])

    return ax






