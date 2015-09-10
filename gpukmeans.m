% gpukmeans use one GPU to compute.
function [label, center, D] = gpukmeans(X, k, start)

[n, dim]=size(X);

g=gpuArray(X);
if (sum(size(start)~=[k, dim])==0), center=start; 
else  center=X(randperm(n,k),:); end

kk=gpuArray(single(repmat(1:k,n,1)));
idxes=gpuArray(single(1:k));

last = 0;label=1;
it=0; maxit=100;
while any(label ~= last) && it<maxit
    last = label;

    D=sqdist(center,g);
    [val,label] = min(D,[],2); % assign samples to the nearest centers

    N=sum(bsxfun(@eq,label,kk));
    missCluster=idxes(N==0);
    if ~isempty(missCluster)
        % fprintf('dropped:%d on %d\n',length(missCluster),it);
        [~,idx] = sort(val,1,'descend');
        lbl=label(idx);
        for i=1:length(missCluster);
            nc=find(N(lbl)>1,1);
            N(lbl(nc))=N(lbl(nc))-1;
            N(missCluster(i))=N(missCluster(i))+1;
            lbl(nc)=missCluster(i);
            label(idx(nc))=missCluster(i);
        end
    end
    
    [lbl,id]=sort(label);
    dif=[diff(lbl)>0; true];
    tmp=(cumsum(g(id,:)));
    center=tmp(dif,:);
    center=[center(1,:); diff(center)];
    center=bsxfun(@rdivide,center,diff([0; find(dif)]));
    it=it+1;
end


center=gather(center);
label=gather(label);
D=gather(D);

end

function d=sqdist(a,b)

    d = bsxfun(@plus, -2*b*a', sum(a.*a,2)');
    d = bsxfun(@plus, d, sum(b.*b,2));

end
