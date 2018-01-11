function [v, pi, diff] = valueIteration(model, maxit)
% initialize the value function
v = zeros(model.stateCount, 1);
diff = [];
    for i = 1:maxit,
        % Initialize the policy and the new value function
        pi = ones(model.stateCount, 1);
        v_ = zeros(model.stateCount, 1);
        % Update Q-value using 
        Q_val = 0; 
        for s_ = 1:model.stateCount,
            term = reshape(model.P(:,s_,:),model.stateCount,4).*(model.R(:,:) + model.gamma*v(s_));
            Q_val = Q_val + term;
        end
        v_ =  max(Q_val,[],2);
        stop_criterion = norm(v-v_,1);
        diff = [diff, stop_criterion];
        v = v_;
        % exit early
        if stop_criterion < 0.000001,
            display('exit early');
            display(i);
            break;
        end
    end
[v, pi] = max(Q_val,[],2);
end

