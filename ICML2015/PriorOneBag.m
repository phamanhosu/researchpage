function p=PriorOneBag(X,W)
% pro=exp(W'*X);
pro=ExpSubstractOneBag(W,X);
sumpro=sum(pro);
one_=ones(1,length(sumpro));
invsumpro=one_./sumpro;
p=pro*diag(invsumpro);
end



function p=ExpSubstractOneBag(W,X)
wx_in=W'*X;
[wx_max,wx_out]=SubstractWX(wx_in);
p=exp(wx_out);
end