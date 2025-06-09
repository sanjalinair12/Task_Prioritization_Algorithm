# Robot Task Assignment using AHP and VIKOR
==============================================

## Overview
------------

This project implements a robot task assignment system using the Analytic Hierarchy Process (AHP) and VIKOR method. The system takes into account multiple criteria and task types to assign robots to tasks.

## Features
------------

*   AHP-based weight calculation for criteria
*   TOPSIS based robot capability evaluation
*   VIKOR method for robot ranking
*   Task prioritization  and team formation based on criteria weights
*   Robot assignment to tasks based on rankings and task requirements.

## Requirements
---------------

*   Python 3.x
*   NumPy library

## Installation
------------

1.  Clone the repository: `git clone https://github.com/sanjalinair12/Task_Prioritization_Algorithm.git`
2.  Install the required libraries: `pip install numpy`
   
   ## Usage
---------

1.  Run the script: `python SearchAndRescue.py`
2.  Follow the prompts to input the number of robots, robot types, and task types.
3.  Enter the pairwise comparisons for the criteria to calculate the weights.
4.  Input the values for each robot and scores for each task.

   
## Google Colab Implementation
------------
1. Login to google colab using your google account.
2. Create a .ipynb file in google colab.
3. Copy the code from  SearchAndRescue.py file in Github to the newly created file.
4. Execute the code for the results.
   



## Output
----------

The system will output the following:

*   Criteria weights
*   Task priorities
*   Robot rankings
*   Assigned robots for each task
*   Decision matrix and normalized decision matrix
*   Ideal best and worst solutions
*   Distance calculations
*   VIKOR index (Q)


