# Generative Models for Face Detection

Developed code from scratch implementing various generative models for 
face detection using the [Faces in the Wild FDDB](http://vis-www.cs.umass.edu/fddb/) dataset. Models include 
Gaussian, Mixture of Gaussians, Factor Analyzer, and Student t model. 
Expectation maximization code written from scratch. 

## Use:
1. Convert FDDB elliptical annotations to rectangular bounding boxes with _convert\_annotation.py_
2. Extract faces and non-face image patches using _extract\_images.py_
3. Run python script of corresponding model to be tested
