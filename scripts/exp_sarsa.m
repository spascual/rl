function [v, pi_, rewards] = exp_sarsa(model, maxit, maxeps)
% initialize the value function
Q = zeros(model.stateCount, 4);
rewards = []; 
alpha = 0.3; 
for i = 1:maxeps,
    % Every time we reset the episode, start at the given startState
    % Initialize action arbitrarily 
    s = model.startState;
    tot_reward = 0;
    for j = 1:maxit,
        % Choose action a_ from Q (eps-greely)
        pi_ = boltzmann_pol(model, Q);
        p = 0;
        r = rand;
        for a = 1:4,
            p = p + pi_(a,s);
            if r <= p,
                break;
            end
        end
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
        % Boltzmann policy for expectation
        Q_exp = Q(s_,:)*pi_(:,s_);
        % Update Q-value function
        Q(s,a) = Q(s,a) + alpha*(model.R(s,a) + model.gamma*Q_exp - Q(s, a));
        s = s_;    
        if s == model.goalState,
            Q(s,:) = 0; %This invalidates previous update but we need to have taken action first
            tot_reward = tot_reward + model.R(s,1);
            break % No need to add reward
        end
    end
    %Store acumulated reward in episode
    rewards = [rewards, tot_reward];
end
% Evaluate value function and policy from Q
[v, pi] = max(Q,[],2);
window_size = 25;
coef = ones(1, window_size)/window_size;
avg_rewards = filter(coef, 1, rewards);
rewards = avg_rewards;
end