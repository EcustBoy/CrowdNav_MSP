Based on https://github.com/vita-epfl/CrowdNav

To test the enviorements, run `crowdnav/tests.py` :

```shell
usage: tests.py [-h] [-l LAYOUT] [-n NUMBER] [-r RENDER] [-t TIMEH] [-m MAXN]
                [-nd NEIGHD]

optional arguments:
  -h, --help            show this help message and exit
  -l LAYOUT, --layout LAYOUT
                        Humans layout, default=circle_crossing
  -n NUMBER, --number NUMBER
                        The number of humans, default=10
  -r RENDER, --render RENDER
                        The render mode (video or traj), default=video
  -t TIMEH, --timeh TIMEH
                        The time horizon of the orca policy, default=5
  -m MAXN, --maxn MAXN  The maximum neighbor param of the orca policy,
                        default=10
  -nd NEIGHD, --neighd NEIGHD
                        The neighbor distance param of the orca policy,
                        default=10
```

Short description of the layouts : 
- 'circle_crossing' (default): Humans generated in a cricle with goal at the opposite of the circle
- 'square_crossing' : Humans with goals generated randomly on a square 
- 'cm_hall' : Humans travel vertically in both directions
- 'cm_hall_oneway' : Humans travel vertically but are only going down (towards the robot)
- 'line' : A vertical line of humans going horizontaly from right to left (10 humans makes the task impossible, try 7)
- 'line-td' : A horizontal line of humans going vertically from top to bottom (10 humans makes the task impossible, try 7)
- 'tri-td' : Same as 'line-td' but humans are in a trinagle formation which should make the task easier for the robot (try 7 humans)
- 'mixed' : Mix different raining simulation with certain distribution

To modify those layouts, look into the respective functions that generate the humans in the simulation file `crowd_sim/envs/crowd_sim.py` called in lines 101 to 193

First results on shifting the domain :

![Alt text](/crowd_nav/data/domain_tests/individual_shifts/individual_shifts_plot.png "Optional Title")

As we can see, little to no changes in robot performances with only individual orca policy changes at a time
