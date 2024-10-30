# Robust-tracking-control-based-on-bounds-on-dynamic-coefficients

## Abstract
In this work, we aim to design a robust control law for tracking a periodic joint space trajectory for a
3R spatial manipulator, based on bounds on its dynamic coefficients. To achieve this, we will derive the
dynamic model and extract a linear parameterization in terms of a minimal set of dynamic coefficients.
We will then apply robust control theory to the case in exam. Additionally, we will conduct simulations to
evaluate the performance of our designed control law, highlighting the main benefits of robust control in
contrast to a classic control law such as feedback linearization under both ideal and uncertain conditions.


## Introduction 
Manipulators often operate in dynamic environments where conditions change, leading to imprecision in their dynamic models. Robust control is essential for maintaining performance amid uncertainties and disturbances. Dynamic coefficients of robots can vary due to factors like component wear or additional payloads. Our robust control method adapts to these variations without requiring constant reparametrization, enhancing system reliability.
For manipulators facing predictable uncertainties, a robust control law can ensure consistent performance despite external disturbances. Traditional robust control methods have struggled with complex uncertainty bounds, often depending on inertia parameters and the reference trajectory. Building on the work of M. Spong, our approach simplifies robust control by focusing on dynamic coefficients, eliminating restrictive assumptions and offering an effective design for accurate trajectory tracking.

## A Practical Case Study: Designing Control Laws for the 3R Spatial Robot
For this study, we have analyzed a 3R Spatial Robot, as shown in the Figure below, We extractextracted the dynamic model of the manipulator with no special assumptions on the
distribution of link masses (CoM location, link inertia matrices), excluding friction and elastic forces to
simplify the calculations. However, it is important to note that a more accurate model can be obtained by
incorporating viscous friction and stiction forces and joint elasticity.

![Dynamic Manipulator](https://github.com/Emilianogith/Robust-tracking-control-based-on-bounds-on-dynamic-coefficients/blob/main/images/3R.png?raw=true)

From this model, we derived a linear parameterization and implemented a robust control law that directly utilizes bounds on the dynamic coefficients. This approach enhances the robot's resilience to uncertainties and external disturbances without requiring complex analyses or restrictive assumptions about the robot's inertia matrix and trajectories.

For a more comprehensive examination, please refer to the report.pdf, which includes performance evaluation graphics comparing our method with a classic feedback linearization control law under two different operating conditions.