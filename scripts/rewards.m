clear all
cliffworld; 
%smallworld;
[v,pi, rewards] = sarsa(model, 50 , 1000);
[v2,pi2, rewards2] = qLearning(model,50, 1000);
%[v3, pi3,rewards3] = exp_sarsa(model, 50, 700);
figure(2);clf
plot(rewards(40:1000))
hold on 
plot(rewards2(40:1000))
title('SARSA vs Qlearning, eps=0.1')
legend('SARSA, alpha=.15', 'Q-learning, alpha=.15');
%ylim([-20, 1])
plot(rewards3(25:700),'k')
hold off
plotVP(v3,pi3, paramSet)


%         pi = boltzmann_pol(model, Q);
%         q = 0;
%         t = rand;
%         for a = 1:4,
%             q = q + pi(a,s);
%             if t <= q,
%                 break;
%             end
%         end
[v2,pi2, rewards2] = qLearning(model, 20, 100);
plot(rewards2,'r')

[v,pi, rewards] = sarsa(model, 50 , 10000);
[v2,pi2, rewards2] = qLearning(model,50, 10000);

plotVP(v2,pi2, paramSet)

[v,pi, rewards] = sarsa(model, 200 , 10000);
plotVP(v,pi, paramSet)