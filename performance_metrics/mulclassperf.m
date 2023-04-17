% Performance metrics
function [ACG,ACA,MCC,PRE,SEN,SPE,F1s,BAC,C] = mulclassperf(Y,Yp,nc)
% Confusion matrix
C = zeros(nc);
for i = 1:nc
    for j = 1:nc
        C(j,i) = sum((Y==i)&(Yp==j));
    end
end
C = C + eps; 
n  = sum(C(:));     % Number of samples
tp = zeros(1,nc);   % True positives
tn = zeros(1,nc);   % True negatives
fp = zeros(1,nc);   % False negatives
fn = zeros(1,nc);   % False positives
for i = 1:nc
    tp(i) = C(i,i);
    fp(i) = sum(C(i,:))-C(i,i);
    fn(i) = sum(C(:,i))-C(i,i);
    tn(i) = n-(tp(i)+fp(i)+fn(i));
end
% Indices
ACG = sum(tp)/n;                          % Global accuracy
ACA = sum((tp+tn)./(tp+fp+fn+tn+eps))/nc; % Average Accuracy
PRE = sum(tp./(tp+fp+eps))/nc;            % Precision
SEN = sum(tp./(tp+fn+eps))/nc;            % Sensitivity 
SPE = sum(tn./(tn+fp+eps))/nc;            % Specificiy
F1s = (2*PRE*SEN)/(PRE+SEN);              % F1 score
BAC = 0.5*(SEN+SPE);                      % Balanced accuracy
% Multiclass MCC
Skl_num  = 0;
Skl_den1 = 0;
Skl_den2 = 0;
Ct  = C';
for k = 1:nc
    for l = 1:nc
        Skl_num  = Skl_num  + sum(C(k,:).*(C(:,l)'));
        Skl_den1 = Skl_den1 + sum(C(k,:).*(Ct(:,l)'));
        Skl_den2 = Skl_den2 + sum(Ct(k,:).*(C(:,l)'));
    end
end
MCC = (n*trace(C) - Skl_num)/(sqrt(n*n - Skl_den1)*sqrt(n*n - Skl_den2)+eps);