import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from tqdm.notebook import tqdm
import matplotlib.animation as animation

class boid():
    '''
    Class for each individual agent (bird, fish, shark, etc) in the swarm. 
    '''
    def __init__(self,x,y,max_force = .8,max_speed = 2.5,v_init = None):
        """_summary_

        Args:
            x (float): starting x coordinate
            y (float): starting y coordinate
            max_force (float, optional): Maximum change in velocity possible each time step. Defaults to .8.
            max_speed (float, optional): Maximum possible speed. Defaults to 2.5.
            v_init (tuple: (float,float)), optional): Initial velocity. Defaults to None, which triggers random initialization
        """        
        self.x = x
        self.y = y
        if v_init is None:
            self.v = np.array([np.random.uniform(-3,3),np.random.uniform(-3,3)]) #initialize random initial velocity
        else:
            self.v = v_init
        self.max_force = max_force
        self.max_speed = max_speed
        
    def update(self, acc_vec):
        """Updates the position and velocity of the boid given the acceleration vector

        Args:
            acc_vec (ndarray(2,),float): Acceleration vector with x and y component
        """        
        if np.linalg.norm(self.v) > self.max_force: #if the acceleration is more than the max allowed
            acc_vec *= (self.max_force/ (np.linalg.norm(self.v))) #normalize it to the maximum
        self.v += acc_vec #add acceleration to the former velocity (this results in something like momentum)
        if np.linalg.norm(self.v) > self.max_speed: #if the new velocity is greater than the max velocity
            self.v *= self.max_speed / (np.linalg.norm(self.v)) #normalize it to the max velocity
        self.x += self.v[0]
        self.y += self.v[1]

    def get_loc(self):
        """Returns the position and velocity of the boid

        Returns:
            (ndarray, (4,), float): [x coordinaate, y coordinate, x velocity, y velocity] of the boid
        """        
        return np.array([self.x,self.y,self.v[0],self.v[1]])

def hunt(n_boids = 1000, n_preds = 5,time_steps = 150, 
         pred_swarm = True,circle_start = True,anim= True,
        sight_boids = 20, sight_pred = 50,
        x_lim = [-200,200],y_lim = [-150,150],
        b_motivation = [2.5, 3, 2, 1], p_motivation = [2, 3, 1,1],
        anim_type = ".gif"):
    """
    Simulates a swarm of predators hunting and capturing a swarm of prey
    or just a swarm of prey (if pred_swarm is not True) over a set spatial
    and temporal domain.

    Args:
        n_boids (int, optional): Number of boids in swarm. Defaults to 1000.
        n_preds (int, optional): Number of predators in swarm. Defaults to 5.
        time_steps (int, optional): How long to run simulation. Defaults to 150.
        pred_swarm (bool, optional): If true, predators will be in system, otherwise, no predators. Defaults to True.
        circle_start (bool, optional): Start the predators in a circle? Defaults to True.
        anim (bool, optional): Output animation? Defaults to True.
        sight_boids (float, optional): Length of sight for boids. Defaults to 20
        sight_pred: (float,optional): Length of sight for predators. Defaults to 50
        x_lim (tuple (int, int)): Spatial width (x) of domain
        y_lim (tuple (int, int)): Spatial height (y) of domain
        b_motivation(tuple (float,float,float,float)): motivations of boids (separation, fleeing, conformity, cohesion)
        p_motivation(tuple (float, float, float, float)): motivations of predators (separation, hunding, conformity, cohesion)
        anim_type (string): set to ".gif" to output a GIF, ".mp4" to output an MP4 file.
    Returns:
        dead_count (int): Number of boids captured by predators during simulation
    """    
    np.random.seed(0) #set random seed for repeatability
    n_boids_orig = n_boids #save the original number of boids
    boids = []
    preds = []
    x_hist = np.ones((time_steps,n_boids))*500
    y_hist = np.ones((time_steps,n_boids))*500
    x_hist_2 = np.zeros((time_steps,n_preds))
    y_hist_2 = np.zeros((time_steps,n_preds))
    dead_count = [0]
    for i in range(n_boids):
        new_loc = np.random.uniform(-200,200,(2,))
        new_boid = boid(new_loc[0],new_loc[1],max_force = .7,max_speed = 3)
        boids.append(new_boid)

    for i in range(n_preds):
        if circle_start:
            theta = 2*np.pi * i/(n_preds)
            new_loc = [30*np.cos(theta),30*np.sin(theta)]
        else:
            new_loc = np.random.uniform(-100,100,(2,))
        #v_0 = [-20*np.sin(theta) , 20*np.cos(theta)]
        new_pred = boid(new_loc[0],new_loc[1],max_force = .5,
                        max_speed = 4)
        preds.append(new_pred)

    for k in tqdm(range(time_steps)):
        dead_count.append(dead_count[-1])
        #get data distances
        total_data = np.zeros((n_preds+n_boids,4))
        for l in range(n_boids):
            total_data[l] = boids[l].get_loc()
        for l in range(n_boids,n_boids + n_preds):
            total_data[l] = preds[l-n_boids].get_loc()
        d = squareform(pdist(total_data[:,:2])) 
        #print(d)
        d += np.eye(d.shape[0])*1000

        #get dead boids
        to_kill = []
        n_dead = 0
        for i,b in enumerate(preds):
            dead = reversed(list(np.where(d[n_boids+i] < 5)[0]))
            to_kill += list(dead)
            n_dead += len(list(dead)) 
        to_kill = np.array(to_kill)
        to_kill = to_kill[to_kill < n_boids]
        to_kill = list(to_kill)
        res = []
        [res.append(k) for k in to_kill if k not in res]
        to_kill = res
        for d in reversed(sorted(to_kill)):
            del boids[d]
            n_boids -= 1
            dead_count[-1] +=1

        #get updated distances
        total_data = np.zeros((n_preds+n_boids,4))
        for l in range(n_boids):
            total_data[l] = boids[l].get_loc()
        for l in range(n_boids,n_boids + n_preds):
            total_data[l] = preds[l-n_boids].get_loc()
        d = squareform(pdist(total_data[:,:2])) 
        d += np.eye(d.shape[0])*1000

        #calculate boid motion
        acc_vec = np.random.normal(0,.1,(n_boids+n_preds,2))
        for b in range(n_boids):
            #deal with edges by making boids bounce off of them
            if total_data[b][0] < x_lim[0] + 5:
                acc_vec[b] += [10,0]
            if total_data[b][0] > x_lim[1] -5:
                acc_vec[b] -= [10,0]
            if total_data[b][1] < y_lim[0] +5:
                acc_vec[b] += [0,10]
            if total_data[b][1] > y_lim[1] -5:
                acc_vec[b] -= [0,10]

            #separation
            too_close = np.array(np.where(d[b] < 6))[0]
            if too_close.shape[0] > 0:
                cluster_center = np.mean(total_data[too_close,:2],axis = 0)
                acc_vec[b] -= b_motivation[0]*((cluster_center - total_data[b][:2])/
                                 np.linalg.norm((cluster_center - total_data[b][:2])+.0001))

            #avoid predators
            too_close = np.array(np.where(d[b] < sight_boids))[0]
            too_close = too_close[too_close >= n_boids]
            if too_close.shape[0] > 0:
                cluster_center = np.mean(total_data[too_close,:2],axis = 0)
                acc_vec[b] -= b_motivation[1]*(cluster_center - total_data[b][:2])/np.linalg.norm((cluster_center - total_data[b][:2])+.0001)


            #conformity and cohesion
            cluster_inds = np.array(np.where(d[b] < sight_boids))
            cluster_inds = cluster_inds[cluster_inds < n_boids]
            if cluster_inds.shape[0] > 0:
                cluster_dir = np.mean(total_data[cluster_inds,2:],axis = 0)
                cluster_dir = cluster_dir / np.linalg.norm(cluster_dir)
                acc_vec[b] += b_motivation[2]*cluster_dir #conformity
                cluster_spot = np.mean(total_data[cluster_inds,:2],axis =0)
                acc_vec[b] += b_motivation[3]*((cluster_spot - total_data[b][:2])/
                        np.linalg.norm((cluster_spot - total_data[b][:2]))) #cohesion

        #calculate predator motion
        for b in range(n_preds):
            #deal with edges
            if total_data[b+n_boids][0] < x_lim[0] + 5:
                acc_vec[b+n_boids] += [3,0]
            if total_data[b+n_boids][0] > x_lim[1] -5:
                acc_vec[b+n_boids] -= [3,0]
            if total_data[b+n_boids][1] < y_lim[0] +5:
                acc_vec[b+n_boids] += [0,3]
            if total_data[b+n_boids][1] > y_lim[1] -5:
                acc_vec[b+n_boids] -= [0,3]

            #Find boids to hunt
            cluster_inds = np.where(d[n_boids+b] < sight_pred)
            cluster_inds = np.array(cluster_inds)
            cluster_inds = cluster_inds[cluster_inds < n_boids]
            cluster_ds = total_data[b] - total_data[cluster_inds]
            if cluster_inds.shape[0] > 0:
                cluster_inds = np.mean(total_data[cluster_inds,:2],axis = 0)
                acc_vec[b+n_boids] += p_motivation[1]*((cluster_inds - total_data[b+n_boids][:2])/(
                        np.linalg.norm((cluster_inds - total_data[b+n_boids][:2]))+.0001))
            
            if pred_swarm:
                #separation
                too_close = np.array(np.where(d[b+n_boids] < sight_pred/3))[0]
                too_close = too_close[too_close >= n_boids]
                if too_close.shape[0] > 0:
                    cluster_center = np.mean(total_data[too_close,:2],axis = 0)
                    acc_vec[b+n_boids] -= p_motivation[0]*((cluster_center - total_data[b+n_boids][:2])/
                            np.linalg.norm((cluster_center - total_data[b+n_boids][:2])+.0001))

                #cohesion and conformity
                cluster_inds = np.where(d[n_boids+b] < 2*sight_pred/3)
                cluster_inds = np.array(cluster_inds)
                cluster_inds = cluster_inds[cluster_inds >= n_boids]
                if cluster_inds.shape[0] > 0:
                    cluster_dir = np.mean(total_data[cluster_inds,2:],axis = 0)
                    cluster_dir = cluster_dir / np.linalg.norm(cluster_dir)
                    acc_vec[b+n_boids] += p_motivation[2]*cluster_dir 
                    cluster_spot = np.mean(total_data[cluster_inds,:2],axis =0)
                    acc_vec[b+n_boids] += p_motivation[3]*((cluster_spot - total_data[b+n_boids][:2])/
                            np.linalg.norm((cluster_spot - total_data[b+n_boids][:2]))) #cohesion


        for i,b in enumerate(boids):
            b.update(acc_vec[i])
            x_hist[k,i], y_hist[k,i] = b.get_loc()[:2]
        for i,b in enumerate(preds):
            b.update(acc_vec[i+n_boids])
            x_hist_2[k,i], y_hist_2[k,i] = b.get_loc()[:2]
    
    if anim:
        print("Animating...")
        fig,ax = plt.subplots(1,figsize = (10,10))

        def update(i):
            ax.clear()
            ax.text(x_lim[0]-10,y_lim[1]-10,str(dead_count[i]),{"size":50},alpha = .4)
            ax.set_xlim(x_lim[0]-20,x_lim[1]+20)
            ax.set_ylim(y_lim[0]-20,y_lim[1]+20)
            ax.scatter(x_hist[i],y_hist[i],s=10,c = np.arange(0,n_boids_orig), cmap ='winter')
            ax.scatter(x_hist_2[i],y_hist_2[i],s= 150,c = "red") 
            plt.gca().set_aspect('equal')

        animation.writer = animation.writers['ffmpeg']
        plt.ioff()  #turn off interaction to save file

        ani = animation.FuncAnimation(fig, update, frames = range(time_steps),interval = 30) #run the animation

        ani.save(f'{n_preds}hunters_{n_boids_orig}boids_flock{pred_swarm}time_s{time_steps}{anim_type}')
    return dead_count

    