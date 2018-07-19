# PyLook
Eye and Head Based User interface / Mouse Actions

### Development Environment
_Note: Please install Cuda and Cudnn for GPU support._
1. install Anaconda
2. ```conda create -n pylook --file requirements_conda.txt```
3. ```source activate pylook```


### Final Product Usage

Usage can be turned on when the conditions are appropriate by
1. Bringing a popup on the side of the screen and asking to stare at the stare using pop up and perform an action. (This makes it totally hands free)
2. Usage can be stopped on alternative (touch / mouse) is used. 
3. Even a toggle button / gesture can be used to start using.

The most natural action of the head movement must be paired with the action on the object of interest (OI).

1. [Gestures](documentation/gestures.md)
2. [Triangulation](documentation/triangulation.md)

## Roadmap

Phase 1:
    First build the nearest OI triangulation. and figure out the lowest distance at which two buttons can be spaced apart. If it is not feasable try the classifier method.

Phase 2:
    Train seperately to identify gestures, make sure gesture detection is feasable.

Phase 2:
    Build the Complete software which taked the OI co-ordinates and gives the choice of OI.
    
Phase 4:
    Add gesture and package complete product.

Phase 5:
    Package as a chrome extention / in a website to test how people percieve it, collect feed back. make adjustments until the new UI seems very intuitive and natural  

Phase 6:
    Write Middleware for each operating system to broadcast objects of interests.

    
_Forgive me for my bad english. This is just a first draft. My english appears better after I iterate over 3-4 times._ 