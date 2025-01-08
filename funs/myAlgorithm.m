function [G0, alpha0] = myAlgorithm(X,num_N,num_C,num_V, p, beta,mm)
sX = [num_N, num_C, num_V];
randSeed = 1;seed=1;rng(randSeed(seed));
for v = 1:num_V
    m{v}=fix(mm*size(X{v},1));
%     m{4}=6;
%         m{v}=mm;
% m{1}=2*mm;m{2}=5*mm;m{3}=5*mm;m{4}=2*mm;m{5}=2*mm;
end
for v = 1:num_V 
    L0{v} = zeros(num_N, num_C);             % tensor c 迭代的中间变量
    Y10{v} = zeros(num_N, num_C);            % Lagrange multipliers
    Y20{v} = zeros(num_N, num_C);
    C0{v} = zeros(num_N, num_C);             % tensor
    z0 = zeros(1, num_V);            % 求alpha的中间变量  
    A0{v} = zeros(num_N, num_C);
    nn{v} = zeros(num_N, num_N);
    vv{v} = zeros(num_C, num_C);
    num_A = size(X{v}, 1);
    W0{v} = orth(rand(num_A, m{v}));
    G0{v} =orth(rand(num_N, num_C));
    G0{v}(G0{v}<0) = 0;
    H0{v} =orth(rand(m{v}, num_C));
end
alpha0 = repmat(1 / num_V, [1,num_V]);        
betaf = ones(num_V, 1);                       

Isconverg = 0;
iter = 0;
epson = 1e-3;   
eta = 1.1;
mu0 = 10e-5;max_mu = 10e15;
rho0 = 10e-5;max_rho = 10e15;
%% 迭代过程
while(Isconverg == 0)  
     %% update H{v} 
for v = 1:num_V 
    H1{v} = W0{v}' * X{v} * G0{v};
    H0{v}=H1{v};
end

    %% update G{v}  
    for v = 1:num_V     
        F{v} = 2*X{v}'*W0{v}*H0{v}/alpha0(v) + mu0*C0{v} - Y10{v} + rho0*A0{v} - Y20{v};
        [nn{v}, ~, vv{v}] = svd(F{v}, 'econ');
        G1{v} = nn{v} * vv{v}';  
        G0{v} =G1{v} ; 
    end

    %% update A{v}
    for v = 1:num_V
        A1{v} = G0{v} + Y20{v} ./ rho0;
        A1{v}(A1{v} < 0) = 0; % 非负
        A0{v} =A1{v} ;
    end
    
    %% update tensor C{v}
    for v = 1:num_V
        L1{v} = G0{v} + Y10{v} ./ mu0;
        L0{v}=L1{v};
    end
    L_tensor = cat(3, L0{:,:});                     %沿着第三个维度将所有 L0{v} 拼接起来
    L_vector = L_tensor(:);                         %将 L_tensor 转换成一个列向量 L_vector 
    [myj, ~] = wshrinkObj_weight_lp(L_vector, beta*betaf./mu0, sX, 0, 3, p);  %得到一个新的列向量 myj ，其中 beta*betaf./mu0 代表张量标准型前的系数，sX 是张量的大小，0 表示不进行权重调整，3 表示使用张量的 shatten 范数，p 表示 shatten 范数的 p 次方。
    C_tensor = reshape(myj, sX);                    %将 myj 按照 sX=[N, C, V] 的大小进行重构
    
    for v = 1:num_V
        C1{v} = C_tensor(:,:,v);                    %将其拆分成 num_V 个二维张量 C1{v} ，并将其赋值给 C0{v} 
        C0{v}=C1{v};
    end

    %% update alpha{v}
    for v = 1:num_V
        z1(v) = norm( W0{v}'*X{v} - H0{v}*G0{v}', 'fro');
        z0(v)=z1(v);
    end
    alpha1 = z0 ./ sum(z0);
    alpha0=alpha1;
    %% update W{v}
    for v = 1:num_V
        for num=1:20
            E{v} =1./alpha0(v)*((100000*eye(size(X{v}*X{v}'))-X{v}*X{v}')*W0{v}+X{v}*G0{v}*H0{v}');
            [nn{v}, ~, vv{v}] = svd(E{v}, 'econ');
            W1{v} = nn{v} * vv{v}' ;    
            chaW=norm(W1{v}-W0{v},inf);
            W0{v}=W1{v};  
            if chaW<1e-2
                break;
            end
            num=num+1;
        end
    end
    %% update Y1 Y2 mu rho
    for v = 1:num_V
        Y11{v} = Y10{v} + mu0 * (G0{v} - C0{v});
        Y21{v} = Y20{v} + rho0 * (G0{v} - A0{v});
        Y10{v}=Y11{v};
        Y20{v}=Y21{v};
    end
    mu1 = min(eta * mu0, max_mu);       %确保新的 mu1 变量不会超过上限值 max_mu
    rho1 = min(eta * rho0, max_rho);
    mu0 =mu1 ;
    rho0=rho1;                                           

    %% 迭代收敛
Isconverg = 1;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    for v = 1:num_V                    
        if (norm(G0{v} - C0{v},inf)>epson)
            history.norm_G_C = norm(G0{v} - C0{v},inf);
            Isconverg = 0;
        end
        if (norm(G0{v} - A0{v},inf)>epson)
            history.norm_G_A = norm(G0{v} - A0{v},inf);
            Isconverg = 0;
        end 
    end
    if iter > 1000
        Isconverg = 1;
    end
    iter = iter + 1;     
end
end