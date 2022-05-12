# Predator-Prey Swarm Simulator and Animator
![alt text](https://github.com/DanielMortenson/PredatoryBoids/blob/ac8a066148b71792811d27a77cea986e943f7137/hunt.gif?raw=true)

## Abstract

Boid's Algorithm (Reynolds 1987) is the classical method for modeling the movement of animals in flocks or swarms. In it, each each bird-oid (boid) actor makes choices about how to move based on the movement of surrounding boids. In this report, I show that Boid's Algorithm can be extended to describe short-term predator-prey dynamics, where one or both of the groups (predator and prey) behave as flocks or swarms. Swarm-predator, swarm-prey behavior is most notably seen in sharks and schools of fish (shoiks and foish) and flocks of birds and flocks of insects. The benefits of flock vs lone hunting are also analyzed here.

## Making Simulations: 

Everything you need to make an animation of predator-prey system is included here. Use the RunSimulation.ipynb Jupyter Notebook to make a simulation in minutes. Requires Python3, Scikit-Learn, Juptyer Notebook, Numpy, Matplotlib, tqdm, and ffmpeg to run.

## Introduction

Boid's Algorithm models the behavior of animal swarms remarkably well, in that the pulses, ripples, and seemingly coordinated movement seen in real-life bird or fish swarms are visually similar to movement seen in a boid's simulation. The algorithm works under the assumption that each boid (bird-oid animal) makes a decision at each time step $t$ on where to steer based only on information that is visible to the bird (within some sight radius $r_s$). That decision is based on three goals:

- Conformity: The boid will try to match its flight direction to the average flight direction of all boids within r_s of the boid.
- Cohesion: The boid will steer toward the center (average position) of the surrounding boids.
- Separation: To avoid collisions, the boid will steer away from any boid within a second smaller radius r_c.

Each of these goals produces a vector. These vectors are then weighted by chosen parameters, summed, and scaled again by a maneuverability constant, before being added to the previous velocity vector of the boid.
This update rule is calculated for all boids at time step $t$, and then each of the boids' positions and velocities are updated. \par

Predator-prey dynamics have been studied extensively on the population scale, but far less research has been done to analyze the dynamics of a swarm of animals hunting another swarm of animals. In 2011, Muro et. al. produced an algorithm to model wolf-pack hunting, where wolves surround an animal surreptitiously before attacking it. Although that study did not involve boid's algorithm per se, it did show that the wolves did not need any more information than what they could see to execute the hunting behaviour, and that there was no need for a leader of the pack: each wolf acted by the same exact set of rules (one of which was separation). \par
The goal of this paper is to show that a  predator that hunts in a pack (hereafter referred to as a \textit{shoik}), governed by an an extended Boid's algorithm is more effective at hunting prey that moves in a swarm (hereafter referred to as \textit{foish} than lone hunters are. Although the names of these simulated animals come from "shark" and "fish," the behavior of dolphins hunting a school of fish is perhaps more similar. Doilphoin simply doesn't do it for me linguistically.

## Implementation

The setup of the shoik-foish simulation is relatively simple. First we establish a flock of foish, which obey the three Boid's laws and have an additional law: Move away from the average position of shoiks within sight. Because survival is important to the foish, steering away from the shoiks has heavier weight than the other three laws. In addition to the foish, we establish a flock of shoiks. Similar to foish, shoiks obey the three boid's laws, and have one more: Move toward the center (average position) of the foish within sight of the shoik. Finally, at the end of each time step, any foish that are touching any shoiks are tallied and eliminated from the simulation. 
At each update step, an acceleration vector (with an x and y velocity) is calculated for each of the four motivations of the foish and shoiks (separation, conformity, cohesion, and hunting/fleeing). This acceleration vector is then weighted by a motivation vector to give some behaviors more importance than others (for fish, fleeing is more important than conformity, for example).
These simulations are only interesting if the parameters are tuned to represent real life when animated. Consequentially, the parameters of the shoiks and foish are established as follows:
For Foish (prey)
- Max Speed : 3
- Max Acceleration: .7
- Conformity: 1
- Separation: 2.5
- Flee: 3
- Sight: 20
- Collision: 6

For Shoiks (predators)
- Max Speed : 4
- Max Acceleration: .5
- Conformity: 3
- Separation: 1
- Hunt: 2
- Sight: 50
- Collision: 16
- Capture Radius: 5
