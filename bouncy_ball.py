# -*- coding: utf-8 -*-
"""

BOUNCY BALL ASSIGNMENT - PHYS20161

The purpose of this code is to calculate (based on user inpt) the number of bounces
above some minimum height and the time it takes for the bouncy ball to complete
these bounces.
The code checks that the inputted values can be used for the calculations, does the
calculations and then prints the results.

Last updated on 27/10/2021

@author: L. Kovacs (student ID: 10735793)

"""

import numpy as np

def number_of_bounces_and_time():

    """
    This function calculates the number of bounces above the minimum height and
    the time taken to complete the bounces.
    It uses values that are inputted by the user within the function.
    """

    gravitational_acceleration = 9.81

    while True:
        initial_height = float(input("What is the initial height? (m) "))
        if initial_height > 0:
            break
        print("Please enter a positive value.")

    while True:
        h_min = float(input("What is the minimum height? (m) "))
        if initial_height >= h_min > 0:
            break
        print("Please enter a valid value (between 0 and the initial height).")

    while True:
        eta = float(input("What is eta, the efficiency of each bounce? (0 < eta < 1) "))
        if 0 < eta < 1:
            break
        print("Please enter a value between 0 and 1.")

    height = initial_height
    n_bounces = 0
    time = np.sqrt(2*height/gravitational_acceleration) #including the initial height

    while height > h_min: #m and g were not included since they cancel out
        height = height*eta
        n_bounces += 1
        time = time + 2*np.sqrt(2*height/gravitational_acceleration)

    print()
    print("The number of bounces from an initial height of " + str(initial_height) +
          " metres over the minimum height of " + str(h_min) + " metres is " + str(n_bounces) + ".")
    print("The time taken for the bounces to complete is {:.2f} seconds.".format(time))

print("Please input your values for the calculations.")

number_of_bounces_and_time()
