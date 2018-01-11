function pi = boltzmann_pol(model,Q)
pi = zeros(4, model.stateCount);

for a = 1:4,
    for s = 1:model.stateCount,
        pi(a,s) = exp(Q(s,a))/sum(exp(Q(s,:)));
    end
end

end