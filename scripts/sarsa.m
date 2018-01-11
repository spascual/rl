function [v, pi, rewards] = sarsa(model, maxit, maxeps)
% initialize the value function
Q = zeros(model.stateCount, 4);
rewards = []; 
alpha = 0.15; 
for i = 1:maxeps,
    % Every time we reset the episode, start at the given startState
    % Initialize action arbitrarily 
    s = model.startState;
    % Choose action a_ from Q (eps-greely)
    [~,pi] = max(Q,[],2);
    a = greedy(pi(s),0.1);
    tot_reward = 0;
    for j = 1:maxit,
        % compute acummulated reward 
        tot_reward = tot_reward + model.R(s,a);
        % Sample s_ from p(s_|s,a)
        p = 0;
        r = rand;
        for s_ = 1:model.stateCount,
            p = p + model.P(s, s_, a);
            if r <= p,
                break;
            end
        end
        % Choose action a_ from s_ using policy pi_ derived Q (eps-greely)
        [~,pi_] = max(Q,[],2);
        a_ = greedy(pi_(s_),0.1);
        % Update Q-value for current state and action
        Q(s,a) = Q(s,a) + alpha*(model.R(s,a) + model.gamma*Q(s_, a_) - Q(s, a));
%         if model.R(s,:) == -10,
%             %break
%         elseif  s_ == model.goalState,
%             Q(s_,:) = 0;
%             tot_reward = tot_reward + model.R(s_,a_);
%             break
                    % Update new state and action
        s = s_;
        a = a_;
        if  s == model.goalState,
            Q(s,:) = 0;
            tot_reward = tot_reward + model.R(s,a);
            break
        end
    end
    tot_reward;
    rewards = [rewards, tot_reward];
    %display(Q(48,4));
end

% Evaluate value function and policy from Q
[v, pi] = max(Q,[],2);
window_size = 25;
coef = ones(1, window_size)/window_size;
avg_rewards = filter(coef, 1, rewards);
rewards = avg_rewards;
end
