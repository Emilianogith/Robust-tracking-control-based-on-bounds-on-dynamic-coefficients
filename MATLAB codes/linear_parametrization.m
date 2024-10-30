syms q1 q2 q3 dq1 dq2 dq3 ddq1 ddq2 ddq3 theta1 theta2 theta3 theta4 theta5 theta6 theta7 theta8 theta9 theta10 theta11 theta12 theta13 real
disp('the given robot inertia matrix') 
q=[q1;q2;q3];
M = [theta1+theta2*sin(q(2))^2+theta3*sin(q(2))*cos(q(2))+theta4*sin(q(2)+q(3))^2+theta5*sin(q(2)+q(3))*cos(q(2)+q(3))+2*theta6*cos(q(2))*sin(q(2)+q(3))+2*theta7*cos(q(2))*cos(q(2)+q(3)), theta8*cos(q(2))+theta9*sin(q(2))+theta10*cos(q(2)+q(3))+theta11*sin(q(2)+q(3)), theta10*cos(q(2)+q(3))+theta11*sin(q(2)+q(3));
         theta8*cos(q(2))+theta9*sin(q(2))+theta10*cos(q(2)+q(3))+theta11*sin(q(2)+q(3)), theta12+2*theta7*cos(q(3))+2*theta6*sin(q(3)), theta13+theta7*cos(q(3))+theta6*sin(q(3));
         theta10*cos(q(2)+q(3))+theta11*sin(q(2)+q(3)), theta13+theta7*cos(q(3))+theta6*sin(q(3)), theta13];


disp('Christoffel matrices')
M1=M(:,1);
C1=(1/2)*(jacobian(M1,q)+jacobian(M1,q)'-diff(M,q1))
M2=M(:,2);
C2=(1/2)*(jacobian(M2,q)+jacobian(M2,q)'-diff(M,q2))
M3=M(:,3);
C3=(1/2)*(jacobian(M3,q)+jacobian(M3,q)'-diff(M,q3))
disp('robot centrifugal and Coriolis terms')
dq=[dq1;dq2;dq3];

c1=simplify(dq'*C1*dq);
c2=simplify(dq'*C2*dq);
c3=simplify(dq'*C3*dq);
c=[c1;c2;c3]
disp('time derivative of the inertia matrix')
dM=diff(M,q1)*dq1+diff(M,q2)*dq2+diff(M,q3)*dq3


disp('skew-symmetric factorization of velocity terms')
S1=dq'*C1;
S2=dq'*C2;
S3=dq'*C3;
S=[S1;S2;S3]
disp('check skew-symmetry of N=dM-2*S') 
N=simplify(dM-2*S)
NplusNT=simplify(N+N')


%% OUTPUT LINEAR PARAMETRIZATION

syms theta14 theta15 l2 g0

G = [0;
    g0*(theta14*cos(q(2))+theta15*sin(q(2))+theta7/l2*cos(q(2)+q(3))+theta6/l2*sin(q(2)+q(3)) );
    g0*(theta7/l2*cos(q(2)+q(3))+theta6/l2*sin(q(2)+q(3)) )];

disp('regressor Y in linear parametrization Y(q,dq,ddq)*theta=tau')
ddq=[ddq1;ddq2;ddq3];
tau=M*ddq+c+G;
theta=[theta1;theta2;theta3;theta4;theta5;theta6;theta7;theta8;theta9;theta10;theta11;theta12;theta13;theta14;theta15];
Y=simplify(jacobian(tau,theta))

