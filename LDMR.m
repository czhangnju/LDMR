function [Xn] = LDMR(y, A, D, trls, Xinit, Einit, alpha, beta, imgsize)

mu = 1;
eps_abs = 1e-3;
eps_rel = 1e-3;
Max_Iter = 100;
p = imgsize(1);
q = imgsize(2);
n = size(A,2);
Xn = Xinit;
En = Einit;
Zn = zeros(prod(imgsize),1);

classnum = length(D);
WA = [];
for i=1:length(trls)
    lb = trls(i);
    WA = [WA D(lb)*A(:,i)];
end
M1 = zeros(n, n);
M2 = zeros(n, n);
nc = 0;
for i=1:classnum
   pos = find(trls==i);
   M1(pos,pos) = D(i)*A(:,pos)'*A(:,pos);
   M2(pos,pos) = (D(i)*A(:,pos))'*(D(i)*A(:,pos));
   nc  = nc + length(pos);
end
M = pinv(alpha/mu*M1+beta/mu*(WA'*WA)+A'*A-beta/mu*M2)*A';

for iter = 1:Max_Iter  
    Zo = Zn;
    Xo = Xn;
    Eo = En;
   
    % update E
    m1 = reshape( A*Xo - y + Zo/mu, imgsize);   
    [AU,SU,VU] = svd(m1,'econ');   
    SU = diag(SU);    
    SVP = length(find(SU>1/mu));  
    if SVP >= 1  
        SU = SU(1:SVP)-1/mu;      
    else
        SVP = 1;  
        SU = 0;  
    end 
    En = AU(:,1:SVP)*diag(SU)*VU(:,1:SVP)';
    En = En(:);

    % update X
    g = y + En - Zo/mu;
    Xn = M*g; 
       
    % update Z
    Zn = Zo + mu*(A*Xn - En - y);
    
%     obj(iter) = sum(svd(En,'econ'));
%     for kk=1:classnum
%         pos = find(trls ==kk);
%         Ai = A(:,pos);
%         xi = Xn(pos);
%         obj(iter) = obj(iter) + alpha/2*D(kk)*norm(Ai*xi,2)^2; 
%         for jj=1:classnum 
%             pos = find(trls ==jj);
%             Aj = A(:,pos);
%             xj = Xn(pos);
%             obj(iter) = obj(iter) + beta/2*(D(kk)*Ai*xi)'*(D(jj)*Aj*xj);
%         end
%     end
    
    % check the convergence condition
    eps_pri = sqrt(p*q)*eps_abs + eps_rel*max( max(norm(A*Xn,2),norm(En,2)), norm(y,2));
    eps_dual = sqrt(n)*eps_abs+eps_rel*norm(A'*Zn, 2);
    r = A*Xn - y - En;
    s = mu*A'*(En-Eo);
    convergence = (norm(r,2)<eps_pri) && (norm(s,2)<eps_dual);
    if convergence
        break;
    end  
    
    
end

end


