# About Repository
## Content
AdNDP 2.0 code associated with the publication:
Tkachenko N.V. and Boldyrev A.I. Chemical bonding analysis of excited states using the Adaptive Natural Density Partitioning method, Phys. Chem. Chem. Phys., 2019, 21, 9590-9596.

Original source code comes from [here](https://zenodo.org/record/3252298#.YxSJKXZBxD8).

## Reason
A friend on mine asked me to do some smaller fixes of the script and when I saw the code I couldn't resist to try it write better. Code style modified to match a little bit better to Python and refactor (or reorganize) the code so I could see what's happening there. That helped to abstract some parts, find possible issues, speed up some parts or add more abilities. I'm blindly moving code from side to side because it make sense from my point of view but I'm not a chemist and variable names in source code didn't give me much idea about what is what so I have to "image" a lot.

## Dependencies
- Python 3
- numpy


## TODOs
- [ ] cli commands
    - [x] create AdNDP
    - [x] analyse
    - [ ] search - is more complicated as requires more inputs
- [ ] log file reader more efficient and cache data
- [x] add ability to define working directory instead of using current work directory
