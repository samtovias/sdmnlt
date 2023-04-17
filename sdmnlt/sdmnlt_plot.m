% Plot SDMN-LT on synthetic datasets
% Learning smooth dendrite morphological neurons for pattern classification using linkage
% ... trees and evolutionary-based hyperparameter tuning (SDMN-LT)
% Evolutionary algorithm: Micro Genetic Algorithm (micro-GA, mga)
function sdmnlt_plot(X,Y,params,dendrite)
NC = numel(cat(1,dendrite.number));
if size(dendrite(1).C,2) == 2
    % Grid 
    mn = -1.1; mx = +1.1; s = 200;
    [x1,x2] = meshgrid(linspace(mn,mx,s),linspace(mn,mx,s));
    X2 = [x1(:) x2(:)];
    % Prediction 
    Yp = sdmnlt_predict(X2,params,dendrite);
    Yp = reshape(Yp,s,s);
    Ypp = sdmnlt_predict(X,params,dendrite);
    % Performance metrics
    [ACC,~,MCC,~,~,~,F1s,~,~] = mulclassperf(Y,Ypp',NC);
    % Figure 
    f = figure('color',[1 1 1]);
    f.Position = [293 250 908 420]; 
    % Plot dendrites 
    subplot(1,2,1);
    h = gscatter(X(:,1),X(:,2),Y); 
    hold on;
    for i = 1:NC
        for j = 1:dendrite(i).number
            viscircles(dendrite(i).C(j,:),dendrite(i).radii(j),'Color',h(i).Color.*0.75);
        end
    end
    tit1 = sprintf('SDMN-LT with micro-GA\nDendrites');
    title(tit1);
    xlabel('$x_1$','Interpreter','latex','FontSize',15);
    ylabel('$x_2$','Interpreter','latex','FontSize',15);
    legend off;
    axis([-1.1 1.1 -1.1 1.1]); 
    axis square;
    axis xy;
    % Plot decision regions 
    subplot(1,2,2);
    imagesc(x1(:),x2(:),Yp);
    hold on;
    gscatter(X(:,1),X(:,2),Y); 
    tit2 = sprintf('Decision regions\nACC: %.2f | MCC: %.2f | F1-Score: %.2f', ACC, MCC, F1s);
    title(tit2);
    xlabel('$x_1$','Interpreter','latex','FontSize',15);
    ylabel('$x_2$','Interpreter','latex','FontSize',15);
    legend off;
    axis([-1.1 1.1 -1.1 1.1]); 
    axis square;
    axis xy;
    cc = 0.75.*cat(1,h(:).Color);
    colormap(cc);
end 
end 