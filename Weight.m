function [D,Res] = Weight(y, trdat, X, trls, delta)
%
classnum = numel(unique(trls));
Res = zeros(classnum,1);
for i=1:classnum
    pos = find(trls == i);
    Xi  = X(pos);
    e = norm(y - trdat(:,pos)*Xi, 2);
    Res(i) = e;
end

%D = (Res - min(Res))/(max(Res)-min(Res));
D = exp(Res/max(Res)/delta);

%D = exp(Res/delta);
%{
Res = zeros(size(trdat,2),1);
for i=1:size(trdat,2)
    Res(i) = norm(y-trdat(:,i),2);
end
D = exp(Res/max(Res));
%}

end

