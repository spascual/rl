function [v, pi] = policyIteration(model, maxit)
v = zeros(model.stateCount, 1);     % initialize the value function
pi = ones(model.stateCount, 1);
policy_stable = false;
j = 0;
V_val = []
while policy_stable == false,
    for i = 1:maxit,    % Evaluate policy pi
        v_ = zeros(model.stateCount, 1); % initialize new value function
        Q_val = bellman_update(model, v);       
        for s = 1:model.stateCount,
            v_(s) = Q_val(s,pi(s));          % VALUE FUNCTION UPDATES(4)
        end
        stop_criterion = norm(v-v_,1);
        v = v_; 
        if stop_criterion < 0.01,
            display(i) %exit early
            break;
           
        end
    end 
    Q_val_ = bellman_update(model, v);      % POLICY (GREEDY) IMPROVEMENT (6)
    [temp, pi_] = max(Q_val_,[],2);
    if pi == pi_
        policy_stable = true;
        display(j)
    end
    pi = pi_;
    j = j+1;
end
    for i = 1:maxit,    % Evaluate policy pi
        v_ = zeros(model.stateCount, 1); % initialize new value function
        Q_val = bellman_update(model, v);       
        for s = 1:model.stateCount,
            v_(s) = Q_val(s,pi(s));          % VALUE FUNCTION UPDATES(4)
        end
        stop_criterion = norm(v-v_,1);
        v = v_; 
        if stop_criterion < 0.01,               %exit early
            break;
        end
    end 
end
