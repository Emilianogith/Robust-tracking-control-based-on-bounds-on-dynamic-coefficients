clc 
clear all
close all

% DYNAMIC COEFFICIENTS
global l2
l1=1; l2=1; l3=1;

global theta_min theta0 theta1 theta2 theta3 theta4 theta5 theta6 theta7 theta8 theta9 theta10 theta11 theta12 theta13 theta14 theta15 

m1=10; m2=10; m3=10;
rcx1=+0.5; rcz1=-0.1;
rcx2=-0.5; rcy2=-0.3; rcz2=0.4;
rcx3=0.2; rcy3=0.2; rcz3=0.2;
Iyy1=0.5;
Ixx2=0.5; Iyy2=0.5; Ixy2=0.2; Ixz2=0.2; Iyz2=0.2; Izz2=0.5;
Ixx3=0.5; Iyy3=0.5; Ixy3=0.2; Ixz3=0.2; Iyz3=0.2; Izz3=0.5;


theta_min=compute_parameters(l2,l3,m1,m2,m3,rcx1,rcz1,rcx2,rcy2,rcz2,rcx3,rcy3,rcz3,Iyy1,Ixx2,Iyy2,Ixy2,Ixz2,Iyz2,Izz2,Ixx3,Iyy3,Ixy3,Ixz3,Iyz3,Izz3);
disp('theta_min')
disp(theta_min)


%introducing perturbations:
m_maximum_perturbation=10;
r_maximum_perturbation=1;
I_maximum_perturbation=1;

m1_max=m1+m_maximum_perturbation; m2_max=m2+m_maximum_perturbation; m3_max=m3+m_maximum_perturbation;
rcx1_max=rcx1+r_maximum_perturbation; rcz1_max=rcz1+r_maximum_perturbation;
rcx2_max=rcx2+r_maximum_perturbation; rcy2_max=rcy2+r_maximum_perturbation; rcz2_max=rcz2+r_maximum_perturbation;
rcx3_max=rcx3+r_maximum_perturbation; rcy3_max=rcy3+r_maximum_perturbation; rcz3_max=rcz3+r_maximum_perturbation;
Iyy1_max=Iyy1+I_maximum_perturbation;
Ixx2_max=Ixx2; Iyy2_max=Iyy2+I_maximum_perturbation; Ixy2_max=Ixy2; Ixz2_max=Ixz2; Iyz2_max=Iyz2; Izz2_max=Izz2+I_maximum_perturbation;
Ixx3_max=Ixx3; Iyy3_max=Iyy3+I_maximum_perturbation; Ixy3_max=Ixy3; Ixz3_max=Ixz3; Iyz3_max=Iyz3; Izz3_max=Izz3+I_maximum_perturbation;

theta_max=compute_parameters(l2,l3,m1_max,m2_max,m3_max,rcx1_max,rcz1_max,rcx2_max,rcy2_max,rcz2_max,rcx3_max,rcy3_max,rcz3_max,Iyy1_max,Ixx2_max,Iyy2_max,Ixy2_max,Ixz2_max,Iyz2_max,Izz2_max,Ixx3_max,Iyy3_max,Ixy3_max,Ixz3_max,Iyz3_max,Izz3_max);
disp('theta_max')
disp(theta_max)


theta0=(theta_max+theta_min)/2;
disp('theta0')
disp(theta0)

%perturbed parameters
seed=123;
rng(seed);

m1=m1+randi([0,m_maximum_perturbation]); m2=m2+randi([0,m_maximum_perturbation]); m3=m3+randi([0,m_maximum_perturbation]);
rcx1=rcx1+randi([0,r_maximum_perturbation]); rcz1=rcz1+randi([0,r_maximum_perturbation]);
rcx2=rcx2+randi([0,r_maximum_perturbation]); rcy2=rcy2+randi([0,r_maximum_perturbation]); rcz2=rcz2+randi([0,r_maximum_perturbation]);
rcx3=rcx3+randi([0,r_maximum_perturbation]); rcy3=rcy3+randi([0,r_maximum_perturbation]); rcz3=rcz3+randi([0,r_maximum_perturbation]);
Iyy1=Iyy1+randi([0,I_maximum_perturbation]);
Ixx2=Ixx2+randi([0,I_maximum_perturbation]); Iyy2=Iyy2+randi([0,I_maximum_perturbation]); Ixy2=Ixy2+randi([0,I_maximum_perturbation]); Ixz2=Ixz2+randi([0,I_maximum_perturbation]); Iyz2=Iyz2+randi([0,I_maximum_perturbation]); Izz2=Izz2+randi([0,I_maximum_perturbation]);
Ixx3=Ixx3+randi([0,I_maximum_perturbation]); Iyy3=Iyy3+randi([0,I_maximum_perturbation]); Ixy3=Ixy3+randi([0,I_maximum_perturbation]); Ixz3=Ixz3+randi([0,I_maximum_perturbation]); Iyz3=Iyz3+randi([0,I_maximum_perturbation]); Izz3=Izz3+randi([0,I_maximum_perturbation]);

theta_perturbed=compute_parameters(l2,l3,m1,m2,m3,rcx1,rcz1,rcx2,rcy2,rcz2,rcx3,rcy3,rcz3,Iyy1,Ixx2,Iyy2,Ixy2,Ixz2,Iyz2,Izz2,Ixx3,Iyy3,Ixy3,Ixz3,Iyz3,Izz3);
disp('theta_perturbed')
disp(theta_perturbed)


perturbed_conditions = true;
if perturbed_conditions == true

    theta1=theta_perturbed(1);
    theta2=theta_perturbed(2);
    theta3=theta_perturbed(3);
    theta4=theta_perturbed(4);
    theta5=theta_perturbed(5);
    theta6=theta_perturbed(6);
    theta7=theta_perturbed(7);
    theta8=theta_perturbed(8);
    theta9=theta_perturbed(9);
    theta10=theta_perturbed(10);
    theta11=theta_perturbed(11);
    theta12=theta_perturbed(12);
    theta13=theta_perturbed(13);
    theta14=theta_perturbed(14);
    theta15=theta_perturbed(15);

else
    theta1=theta_min(1);
    theta2=theta_min(2);
    theta3=theta_min(3);
    theta4=theta_min(4);
    theta5=theta_min(5);
    theta6=theta_min(6);
    theta7=theta_min(7);
    theta8=theta_min(8);
    theta9=theta_min(9);
    theta10=theta_min(10);
    theta11=theta_min(11);
    theta12=theta_min(12);
    theta13=theta_min(13);
    theta14=theta_min(14);
    theta15=theta_min(15);
end


%compute rho
global rho
rho=norm(theta0-theta_min)

%%
global T
% Simulation parameters
T = 10; 
dt = 0.01; 
t = 0:dt:T; 

% Desired trajectory
q_d = [sin(t); 5*cos(t); sin(t)];
dq_d = [cos(t); -5*sin(t); cos(t)]; 
ddq_d = [-sin(t); -5*cos(t); -sin(t)];

% State variables initialization
q0 = [0; 5; 0];
dq0 = [1; 0; 1];
y0 = [q0; dq0];

% Integration Routine with ode15s
options = odeset('RelTol',1e-5, 'AbsTol',1e-5); % Tolerances settings
[t, y] = ode15s(@(t, y) robot_dynamics_ode(t, y, q_d, dq_d, ddq_d), t, y0,options); 


q = y(:, 1:3)';
dq = y(:, 4:6)';


u = zeros(size(q));
% Compute the torque for each time step
for i = 1:length(t)
    u(:,i) = robust_control(q(:,i), dq(:,i), q_d(:,i), dq_d(:,i), ddq_d(:,i));
    %u(:,i) = feedbacklinearization(q(:,i), dq(:,i), q_d_t(:,i), dq_d_t(:,i), ddq_d_t(:,i));
end

% Plotting
figure('Position', [200, 250, 1100, 700]);
subplot(3, 3, 1);
plot(t, q_d(1,:), '--', t, q(1,:), 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Angle (rad)');
legend('q_d1','q1');
title('joint position1');

subplot(3, 3, 4);
plot(t, q_d(2,:), '--', t, q(2,:), 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Angle (rad)');
legend('q_d2','q2');
title('joint position2');

subplot(3, 3, 7);
plot(t, q_d(3,:), '--', t, q(3,:), 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Angle (rad)');
legend('q_d3','q3');
title('joint position3');

subplot(3, 3, 2);
plot(t, u(1,:), 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Torque ');
title('joint torque1');

subplot(3, 3, 5);
plot(t, u(2,:), 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Torque');
title('joint torque2');

subplot(3, 3,8);
plot(t, u(3,:), 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Torque');
title('joint torque3');

subplot(3, 3, 3);
plot(t, q_d(1,:)-q(1,:), 'LineWidth', 1.5);
xlabel('Time (s)');
title('position error1');

subplot(3, 3, 6);
plot(t, q_d(2,:)-q(2,:), 'LineWidth', 1.5);
xlabel('Time (s)');
title('position error2');

subplot(3, 3, 9);
plot(t, q_d(3,:)-q(3,:), 'LineWidth', 1.5);
xlabel('Time (s)');
title('position error3');


% Integration function for ode15s
function dydt = robot_dynamics_ode(t, y, q_d, dq_d, ddq_d)
    global T
    q = y(1:3);
    dq = y(4:6);

    % Interpolation of the desired values at time t
    q_d_t = interp1(linspace(0, T, length(q_d)), q_d', t, 'linear', 'extrap')';
    dq_d_t = interp1(linspace(0, T, length(dq_d)), dq_d', t, 'linear', 'extrap')';
    ddq_d_t = interp1(linspace(0, T, length(ddq_d)), ddq_d', t, 'linear', 'extrap')';


    %SIMULATION WITH ROBUST CONTROL
    u = robust_control(q, dq, q_d_t, dq_d_t, ddq_d_t);
    
    %SIMULATION WITH FEEDBACK LINEARIZATION
    %u = feedbacklinearization(q, dq, q_d_t, dq_d_t, ddq_d_t);

    ddq = robot_dynamics(q, dq, u);
    dydt = [dq; ddq];
end


function [ddq] = robot_dynamics(q, dq, u)
    global l2
    global theta1 theta2 theta3 theta4 theta5 theta6 theta7 theta8 theta9 theta10 theta11 theta12 theta13 theta14 theta15
    g0=9.8;
    

    %INERTIA MATRIX
   
    M = [theta1+theta2*sin(q(2))^2+theta3*sin(q(2))*cos(q(2))+theta4*sin(q(2)+q(3))^2+theta5*sin(q(2)+q(3))*cos(q(2)+q(3))+2*theta6*cos(q(2))*sin(q(2)+q(3))+2*theta7*cos(q(2))*cos(q(2)+q(3)), theta8*cos(q(2))+theta9*sin(q(2))+theta10*cos(q(2)+q(3))+theta11*sin(q(2)+q(3)), theta10*cos(q(2)+q(3))+theta11*sin(q(2)+q(3));
         theta8*cos(q(2))+theta9*sin(q(2))+theta10*cos(q(2)+q(3))+theta11*sin(q(2)+q(3)), theta12+2*theta7*cos(q(3))+2*theta6*sin(q(3)), theta13+theta7*cos(q(3))+theta6*sin(q(3));
         theta10*cos(q(2)+q(3))+theta11*sin(q(2)+q(3)), theta13+theta7*cos(q(3))+theta6*sin(q(3)), theta13];


    q2=q(2);
    q3=q(3);
    dq1=dq(1);
    dq2=dq(2);
    dq3=dq(3);

    %CORIOLIS AND CENTRIFUGAL TERMS
    C=[dq2^2*theta11*cos(q2 + q3) + dq3^2*theta11*cos(q2 + q3) - dq2^2*theta10*sin(q2 + q3) - dq3^2*theta10*sin(q2 + q3) + dq2^2*theta9*cos(q2) - dq2^2*theta8*sin(q2) + dq1*dq2*theta5*cos(2*q2 + 2*q3) + dq1*dq3*theta5*cos(2*q2 + 2*q3) + dq1*dq2*theta4*sin(2*q2 + 2*q3) + dq1*dq3*theta4*sin(2*q2 + 2*q3) + 2*dq2*dq3*theta11*cos(q2 + q3) - 2*dq2*dq3*theta10*sin(q2 + q3) + dq1*dq3*theta6*cos(q3) - dq1*dq3*theta7*sin(q3) + 2*dq1*dq2*theta6*cos(2*q2 + q3) + dq1*dq3*theta6*cos(2*q2 + q3) - 2*dq1*dq2*theta7*sin(2*q2 + q3) - dq1*dq3*theta7*sin(2*q2 + q3) + dq1*dq2*theta3*cos(2*q2) + dq1*dq2*theta2*sin(2*q2);
                                                                                                                                                                                                                                                                                                                      dq3^2*theta6*cos(q3) - (dq1^2*theta4*sin(2*q2 + 2*q3))/2 - dq3^2*theta7*sin(q3) - dq1^2*theta6*cos(2*q2 + q3) + dq1^2*theta7*sin(2*q2 + q3) - (dq1^2*theta3*cos(2*q2))/2 - (dq1^2*theta2*sin(2*q2))/2 - (dq1^2*theta5*cos(2*q2 + 2*q3))/2 + 2*dq2*dq3*theta6*cos(q3) - 2*dq2*dq3*theta7*sin(q3);
                                                                                                                                                                                                                                                                                                                                                                                                             (- (theta5*cos(q2 + q3)^2)/2 - theta4*cos(q2 + q3)*sin(q2 + q3) - theta6*cos(q2)*cos(q2 + q3) + (theta5*sin(q2 + q3)^2)/2 + theta7*cos(q2)*sin(q2 + q3))*dq1^2 + (theta7*sin(q3) - theta6*cos(q3))*dq2^2];
    
    %GRAVITY TERMS
    G = [0;
        g0*(theta14*cos(q(2))+theta15*sin(q(2))+theta7/l2*cos(q(2)+q(3))+theta6/l2*sin(q(2)+q(3)) );
        g0*(theta7/l2*cos(q(2)+q(3))+theta6/l2*sin(q(2)+q(3)) )];

    ddq = M \ (u - C - G);
    disp(ddq);


    end

function u = robust_control(q, dq, q_d, dq_d, ddq_d)
    g0=9.8;
    global theta0 l2 rho

    % Gains
    K = diag([100, 100, 100]);        
    lambda= diag([50, 50, 50]);      

    dqs=dq-dq_d;
    qs=q-q_d;
    r = dqs+lambda*qs;

    v = dq_d-lambda*qs;

    a=ddq_d-lambda*dqs;

    %REGRESSOR MATRIX
    y11=a(1);
    y12=a(1)*sin(q(2))^2+(v(1)*dq(2)+dq(1)*v(2))*cos(q(2))*sin(q(2));
    y13=a(1)*cos(q(2))*sin(q(2))+(1/2*v(1)*dq(2)+1/2*dq(1)*v(2))*(cos(q(2))^2-sin(q(2))^2);
    y14=a(1)*sin(q(2)+q(3))^2+(v(1)*dq(2)+dq(1)*v(2))*cos(q(2)+q(3))*sin(q(2)+q(3))+(v(1)*dq(3)+dq(1)*v(3))*cos(q(2)+q(3))*sin(q(2)+q(3));
    y15=a(1)*cos(q(2)+q(3))*sin(q(2)+q(3))+(1/2*v(1)*dq(2)+1/2*dq(1)*v(2))*(cos(q(2)+q(3))^2-sin(q(2)+q(3))^2)+(1/2*v(1)*dq(3)+1/2*dq(1)*v(3))*(cos(q(2)+q(3))^2-sin(q(2)+q(3))^2);
    y16=2*a(1)*cos(q(2))*sin(q(2)+q(3))+(v(1)*dq(2)+dq(1)*v(2))*(cos(q(2))*cos(q(2)+q(3))-sin(q(2))*sin(q(2)+q(3)))+(v(1)*dq(3)+dq(1)*v(3))*cos(q(2))*cos(q(2)+q(3));
    y17=2*a(1)*cos(q(2))*cos(q(2)+q(3))-(v(1)*dq(2)+dq(1)*v(2))*(sin(q(2))*cos(q(2)+q(3))+cos(q(2))*sin(q(2)+q(3)))-(v(1)*dq(3)+dq(1)*v(3))*cos(q(2))*sin(q(2)+q(3));
    y18=a(2)*cos(q(2))-v(2)*dq(2)*sin(q(2));
    y19=a(2)*sin(q(2))+v(2)*dq(2)*cos(q(2));
    y110=(a(2)+a(3))*cos(q(2)+q(3))-dq(2)*v(2)*sin(q(2)+q(3))-dq(3)*v(3)*sin(q(2)+q(3))-(dq(3)*v(2)+v(3)*dq(2))*sin(q(2)+q(3));
    y111=(a(2)+a(3))*sin(q(2)+q(3))+dq(2)*v(2)*cos(q(2)+q(3))+dq(3)*v(3)*cos(q(2)+q(3))+(dq(3)*v(2)+v(3)*dq(2))*cos(q(2)+q(3));
    y22=-dq(1)*v(1)*sin(q(2))*cos(q(2));
    y23=-1/2*dq(1)*v(1)*(cos(q(2))^2-sin(q(2))^2 );
    y24=-dq(1)*v(1)*sin(q(2)+q(3))*cos(q(2)+q(3));
    y25=-1/2*dq(1)*v(1)*(cos(q(2)+q(3))^2-sin(q(2)+q(3))^2 );
    y26=2*a(2)*sin(q(3))+a(3)*sin(q(3))-v(1)*dq(1)*(cos(q(2))*cos(q(2)+q(3))-sin(q(2))*sin(q(2)+q(3)))+(v(2)*dq(3)+dq(2)*v(3))*cos(q(3))+dq(3)*v(3)*cos(q(3))+g0/l2*sin(q(2)+q(3));
    y27=2*a(2)*cos(q(3))+a(3)*cos(q(3))+v(1)*dq(1)*(sin(q(2))*cos(q(2)+q(3))+cos(q(2))*sin(q(2)+q(3)))-(v(2)*dq(3)+dq(2)*v(3))*sin(q(3))-dq(3)*v(3)*sin(q(3))+g0/l2*cos(q(2)+q(3));
    y28=a(1)*cos(q(2));
    y29=a(1)*sin(q(2));
    y210=a(1)*cos(q(2)+q(3));
    y211=a(1)*sin(q(2)+q(3));
    y212=a(2);
    y213=a(3);
    y214=g0*cos(q(2));
    y215=g0*sin(q(2));
    y34=-dq(1)*v(1)*cos(q(2)+q(3))*sin(q(2)+q(3));
    y35=-1/2*dq(1)*v(1)*(cos(q(2)+q(3))^2-sin(q(2)+q(3))^2 );
    y36=a(2)*sin(q(3))-dq(1)*v(1)*cos(q(2))*cos(q(2)+q(3))-dq(2)*v(2)*cos(q(3))+g0/l2*sin(q(2)+q(3));
    y37=a(2)*cos(q(3))+dq(1)*v(1)*cos(q(2))*sin(q(2)+q(3))+dq(2)*v(2)*sin(q(3))+g0/l2*cos(q(2)+q(3));
    y310=a(1)*cos(q(2)+q(3));
    y311=a(1)*sin(q(2)+q(3));
    y313=a(2)+a(3);

    Y=[y11,y12,y13,y14,y15,y16,y17,y18,y19,y110,y111,0,0,0,0;
        0,y22,y23,y24,y25,y26,y27,y28,y29,y210,y211,y212,y213,y214,y215;
        0,0,0,y34,y35,y36,y37,0,0,y310,y311,0,y313,0,0];

    %Robust controller parameter
    epsilon=0.5;
    nor=norm(transpose(Y)*r);
    
    if nor > epsilon
        delta=-rho*transpose(Y)*r/nor;
    else
        delta=-rho*transpose(Y)*r/epsilon;

    end
    
    u=Y*(theta0+delta)-K*r;
    disp('u_');
    disp(u);
    
end


function u=feedbacklinearization(q, dq, q_d, dq_d, ddq_d)
    global l2
    g0=9.8;
    Kp=diag([100,100,100]);
    Kd=diag([100,100,100]);
    
    %ideal conditions from the dynamic model
    global theta_min
    theta1=theta_min(1);
    theta2=theta_min(2);
    theta3=theta_min(3);
    theta4=theta_min(4);
    theta5=theta_min(5);
    theta6=theta_min(6);
    theta7=theta_min(7);
    theta8=theta_min(8);
    theta9=theta_min(9);
    theta10=theta_min(10);
    theta11=theta_min(11);
    theta12=theta_min(12);
    theta13=theta_min(13);
    theta14=theta_min(14);
    theta15=theta_min(15);

    

    %INERTIA MATRIX
   
    M = [theta1+theta2*sin(q(2))^2+theta3*sin(q(2))*cos(q(2))+theta4*sin(q(2)+q(3))^2+theta5*sin(q(2)+q(3))*cos(q(2)+q(3))+2*theta6*cos(q(2))*sin(q(2)+q(3))+2*theta7*cos(q(2))*cos(q(2)+q(3)), theta8*cos(q(2))+theta9*sin(q(2))+theta10*cos(q(2)+q(3))+theta11*sin(q(2)+q(3)), theta10*cos(q(2)+q(3))+theta11*sin(q(2)+q(3));
         theta8*cos(q(2))+theta9*sin(q(2))+theta10*cos(q(2)+q(3))+theta11*sin(q(2)+q(3)), theta12+2*theta7*cos(q(3))+2*theta6*sin(q(3)), theta13+theta7*cos(q(3))+theta6*sin(q(3));
         theta10*cos(q(2)+q(3))+theta11*sin(q(2)+q(3)), theta13+theta7*cos(q(3))+theta6*sin(q(3)), theta13];


    q2=q(2);
    q3=q(3);
    dq1=dq(1);
    dq2=dq(2);
    dq3=dq(3);
    %CORIOLIS AND CENTRIFUGAL TERMS
    C=[dq2^2*theta11*cos(q2 + q3) + dq3^2*theta11*cos(q2 + q3) - dq2^2*theta10*sin(q2 + q3) - dq3^2*theta10*sin(q2 + q3) + dq2^2*theta9*cos(q2) - dq2^2*theta8*sin(q2) + dq1*dq2*theta5*cos(2*q2 + 2*q3) + dq1*dq3*theta5*cos(2*q2 + 2*q3) + dq1*dq2*theta4*sin(2*q2 + 2*q3) + dq1*dq3*theta4*sin(2*q2 + 2*q3) + 2*dq2*dq3*theta11*cos(q2 + q3) - 2*dq2*dq3*theta10*sin(q2 + q3) + dq1*dq3*theta6*cos(q3) - dq1*dq3*theta7*sin(q3) + 2*dq1*dq2*theta6*cos(2*q2 + q3) + dq1*dq3*theta6*cos(2*q2 + q3) - 2*dq1*dq2*theta7*sin(2*q2 + q3) - dq1*dq3*theta7*sin(2*q2 + q3) + dq1*dq2*theta3*cos(2*q2) + dq1*dq2*theta2*sin(2*q2);
                                                                                                                                                                                                                                                                                                                      dq3^2*theta6*cos(q3) - (dq1^2*theta4*sin(2*q2 + 2*q3))/2 - dq3^2*theta7*sin(q3) - dq1^2*theta6*cos(2*q2 + q3) + dq1^2*theta7*sin(2*q2 + q3) - (dq1^2*theta3*cos(2*q2))/2 - (dq1^2*theta2*sin(2*q2))/2 - (dq1^2*theta5*cos(2*q2 + 2*q3))/2 + 2*dq2*dq3*theta6*cos(q3) - 2*dq2*dq3*theta7*sin(q3);
                                                                                                                                                                                                                                                                                                                                                                                                             (- (theta5*cos(q2 + q3)^2)/2 - theta4*cos(q2 + q3)*sin(q2 + q3) - theta6*cos(q2)*cos(q2 + q3) + (theta5*sin(q2 + q3)^2)/2 + theta7*cos(q2)*sin(q2 + q3))*dq1^2 + (theta7*sin(q3) - theta6*cos(q3))*dq2^2];
    
    %GRAVITY TERMS
    G = [0;
        g0*(theta14*cos(q(2))+theta15*sin(q(2))+theta7/l2*cos(q(2)+q(3))+theta6/l2*sin(q(2)+q(3)) );
        g0*(theta7/l2*cos(q(2)+q(3))+theta6/l2*sin(q(2)+q(3)) )];
    
    u=M*(ddq_d+Kp*(q_d-q)+Kd*(dq_d-dq))+C+G;
    disp('u=');
    disp(u);
end


function [theta_vector]=compute_parameters(l2,l3,m1,m2,m3,rcx1,rcz1,rcx2,rcy2,rcz2,rcx3,rcy3,rcz3,Iyy1,Ixx2,Iyy2,Ixy2,Ixz2,Iyz2,Izz2,Ixx3,Iyy3,Ixy3,Ixz3,Iyz3,Izz3)
    theta1=m1*(rcx1^2+rcz1^2)+Iyy1+m2*rcz2^2+m2*(rcx2+l2)^2+Iyy2+m3*rcz3^2+m3*(rcx3+l3)^2+Iyy3+m3*l2^2;
    theta2=m2*rcy2^2-m2*(rcx2+l2)^2-Iyy2+Ixx2-m3*l2^2;
    theta3=-2*m2*(rcx2+l2)*rcy2+2*Ixy2;
    theta4=m3*rcy3^2-m3*(rcx3+l3)^2+Ixx3-Iyy3;
    theta5=-2*m3*(rcx3+l3)*rcy3+2*Ixy3;
    theta6=-m3*rcy3*l2;
    theta7=m3*(rcx3+l3)*l2;
    theta8=-m2*rcz2*rcy2+Iyz2;
    theta9=-m2*rcz2*(rcx2+l2)+Ixz2-m3*rcz3*l2;
    theta10=-m3*rcz3*rcy3+Iyz3;
    theta11=-m3*rcz3*(rcx3+l3)+Ixz3;
    theta12=m2*((rcx2+l2)^2+rcy2^2)+Izz2+m3*(l2^2+(rcx3+l3)^2+rcy3^2)+Izz3;
    theta13=m3*((rcx3+l3)^2+rcy3^2)+Izz3;
    theta14=m2*(l2+rcx2)+m3*l2;
    theta15=-m2*rcy2;

    theta_vector=[theta1;theta2;theta3;theta4;theta5;theta6;theta7;theta8;theta9;theta10;theta11;theta12;theta13;theta14;theta15];

end
