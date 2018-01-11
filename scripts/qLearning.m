function [v, pi, rewards] = qLearning(model, maxit, maxeps)
% initialize the value function
Q = zeros(model.stateCount, 4);
rewards = [];
alpha = 0.15; 
for i = 1:maxeps,
    % Every time we reset the episode, start at the given startState
    % Initialize action arbitrarily 
    s = model.startState;
    tot_reward = 0;
    episode = [];
    for j = 1:maxit,
        % Choose action a from s using  policy derived from Q (eps-greely)
        [~,pi] = max(Q,[],2); 
        a = greedy(pi(s),0.01);
%         for a = 1:4,
%             p = p + model.pi(a);
%             if r <= p,
%                 break;
%             end
%         end
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
        % Update Q-value for current state and action
        Q_max = max(Q,[],2);
        Q(s,a) = Q(s,a) + alpha*(model.R(s,a) + model.gamma*Q_max(s_) - Q(s, a));
        %Stop episode when goal state reached
%         if model.R(s,:) == -10,
%             %break
%         elseif s == model.goalState,
%             Q(s,:) = 0; %This invalidates previous update but we need to have taken action first
%             break % No need to add reward
        s = s_; 
        if s == model.goalState,
            Q(s,:) = 0; %This invalidates previous update but we need to have taken action first
            tot_reward = tot_reward + model.R(s,1);
            break % No need to add reward
        end
    end
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

