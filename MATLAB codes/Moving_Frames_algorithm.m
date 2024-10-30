clear all
clc
close all

% Define symbolic variables

syms alpha d a theta



% number of joints 

N=3;       %<-----MODIFY

% Insert DH table of parameters

DHTABLE = [  pi/2       0           sym('l1')     sym('q1');
             0          sym('l2')   0             sym('q2');
             0          sym('l3')   0             sym('q3');
             % 0       sym('l4')       0           sym('q4');
             % pi/2         0               0               sym('q5');
             % 0            0              sym('d6')        sym('q6')
];



TDH = [ cos(theta) -sin(theta)*cos(alpha)  sin(theta)*sin(alpha) a*cos(theta);
        sin(theta)  cos(theta)*cos(alpha) -cos(theta)*sin(alpha) a*sin(theta);
          0             sin(alpha)             cos(alpha)            d;
          0               0                      0                   1];

A = cell(1,N);
E = cell(1,N);


for i = 1:N
    alpha = DHTABLE(i,1);
    a = DHTABLE(i,2);
    d = DHTABLE(i,3);
    theta = DHTABLE(i,4);
    A{i} = subs(TDH);
    R{i}=A{i}(1:3,1:3);
    % disp(['A',num2str(i),'='])
    % disp([A{i}])

    %computing the homogeneous transformation matrix
    if i==1
           E{i}=A{i};
    else
        E{i} = E{i-1}*A{i};
    end
    disp(E{i})
end


% RECURSIVE ALGORITHM

%Initialization
sigma=[0;
       0;
       0]; %0 for revolute 1 for prismatic

v0=[0;0;0];
w0=[0;0;0];
 
syms qd1 qd2 qd3 m1 m2 m3      %<-----MODIFY-------------------------------
qdot=[qd1;
      qd2;
      qd3];
m=[m1;
   m2;
   m3];

syms Ix1 Ix2 Ix3 Iy1 Iy2 Iy3 Iz1 Iz2 Iz3 Ixy1 Ixz1 Iyz1 Ixy2 Ixz2 Iyz2 Ixy3 Ixz3 Iyz3
I1=[Ix1,Ixy1,Ixz1;
    Ixy1,Iy1,Iyz1;
    Ixz1,Iyz1,Iz1];
I2=[Ix2,Ixy2,Ixz2;
    Ixy2,Iy2,Iyz2;
    Ixz2,Iyz2,Iz2];
I3=[Ix3,Ixy3,Ixz3;
    Ixy3,Iy3,Iyz3;
    Ixz3,Iyz3,Iz3];

I = {I1, I2, I3};

%Position of CoM in Frame i
rc1=[sym('rcx1');
     sym('rcy1');
     sym('rcz1')];
rc2=[sym('rcx2');
     sym('rcy2');
     sym('rcz2')];
rc3=[sym('rcx3');
     sym('rcy3');
     sym('rcz3')];

rc=horzcat(rc1,rc2, rc3);
%--------------------------------------------------------------------------
syms g0
g=[0;0;-g0];     %<------definition of g vector according with Frame 0
U=0;
k=0;
for i = 1:N

    %computing velocities w{i}, v{i} in Frame i
    if i==1
        w{i} = transpose(R{i})*( w0+(1-sigma(i))*qdot(i)*[0;0;1] );
        v{i} = simplify(transpose(R{i})*( v0+sigma(i)*qdot(i)*[0;0;1]) +(cross(w{i}, transpose(R{i})*A{i}(1:3,4))));
    else 
        w{i} = simplify(transpose(R{i})*( w(i-1)+(1-sigma(i))*qdot(i)*[0;0;1] ));
        v{i} = simplify(transpose(R{i})*( v{i-1}+sigma(i)*qdot(i)*[0;0;1]) + (cross(w{i}, transpose(R{i})*A{i}(1:3,4))));
    end
    disp(['w',num2str(i),'='])
    disp(w{i})
    disp(['v',num2str(i),'='])
    disp(v{i})

    %computing vc{i} in Frame i
    vc{i}= simplify(v{i} + cross(w{i},rc(:,i)));

    disp(['vc',num2str(i),'='])
    disp(vc{i});

    %computing the kinetic energy
    assume([vc{i} w{i}],"real")
    rotational=simplify(1/2*transpose(w{i})*I(i)*w{i}, 'IgnoreAnalyticConstraints',true);
    translational=simplify(combine(1/2*m(i)*transpose(vc{i})*vc{i}, 'IgnoreAnalyticConstraints',true));
    T{i}=simplify(rotational+translational);
    % T{i}=simplify(1/2*m(i)*norm(vc{i})^2 + 1/2* transpose(w{i})* I(i)*w{i} , 'IgnoreAnalyticConstraints',true);
     T{i}=simplify(collect(T{i},qdot), steps=100);


    %computing pc{i} in Frame 0
    pc{i}=simplify(E{i}(1:3,4)+E{i}(1:3,1:3)*rc(:,i));
    dpc{i}=simplify(E{i}(1:3,1:3)*vc{i});

    disp(['pc',num2str(i),' in Frame 0 ='])
    disp(pc{i});

    disp(['dpc',num2str(i),' in Frame 0 ='])
    disp(dpc{i});


    disp(['T',num2str(i),'='])  
    disp(T{i});


    %Potential energy
    U=simplify(U-m(i)*transpose(g)*pc{i});


    k=simplify(k+T{i}, steps=100);
end
disp('Total kinetic energy:')
%disp(k)
k=simplify(collect(k, qd1), steps=100);
disp(k)


%% POTENTIAL ENERGY
syms q1 q2 q3
disp('gravity vector:')
g_vec=transpose(simplify(jacobian(U,[q1,q2,q3])));    %<----- modify the differentiation variables by hand
disp(g_vec)



