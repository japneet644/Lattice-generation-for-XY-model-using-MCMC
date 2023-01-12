clear all; close all;
RESULT = zeros(7,9);
RRUvals = [0.3500, 0.4250, 0.5000, 0.5750, 0.6500, 0.6910];
for fgh= 3:8
    Radius = 80;
    numOfLayers = 2;
    plotData = 1;
    numofRUs = 6;
    RRUdist = RRUvals(fgh-2); %0.2 + 0.075*(fgh-1);

    %BSlocations = generateHexGrid(Radius, numOfLayers,plotData);
    [cells,Hexgrid] = generateDAS(Radius,numOfLayers,numofRUs+1,RRUdist,plotData);
    L = 30.5; %46.3 + 33.9*log10(1900) - 13.82*log10(15) - (1.1*log10(1900) - 0.7)*1.65 + (1.56*log10(1900) - 0.8);
    pathloss_parameter = 10^(-L/10);
    noise_sdev = 0.00008;
    noise_power= 10^(-10.9);
    totalBS = size(cells,2); 

    %% Assigning Channel coefficients
    H = zeros(totalBS,totalBS*numofRUs);
    for kc = 1:size(cells,2)
        for ku=2:numofRUs+1
            for kb = 1:size(cells,2)
                cells(kc).RUs(ku,2+kb) = sqrt((pathloss_parameter*10^(0.05*noise_sdev*randn(1)))/(norm(cells(kc).RUs(ku,1:2)-cells(kb).RUs(1,1:2))^3.67));
            end
            H(:,numofRUs*(kc-1)+ku-1) = cells(kc).RUs(ku,3:totalBS+2)';
        end
    end 

    %% frequency reuse
    grid_color = ['b','g','m','g','m','g','m','m','b','g','b','m','b','g','b','m','b','g','b','g','b','g','m','b','m','g','b','g','m','b','m','g','b','g','m','b','m'];
    avgrecieved_power = 0;
    sum_rate_FR3 = 0;
    avginterfer_power=0;
    for ku=2:numofRUs+1
        avgrecieved_power = avgrecieved_power + (norm(cells(1).RUs(ku,3))^2)/numofRUs;
        interfer_power = 0;
        for kb = 2:size(cells,2)
            if grid_color(kb) =='b'
                interfer_power = interfer_power + (norm(cells(1).RUs(ku,kb+2))^2);
            end
        end
        avginterfer_power = avginterfer_power + interfer_power/numofRUs;
        sum_rate_FR3 = sum_rate_FR3 + 1/(3*numofRUs)*log2(1 + (norm(cells(1).RUs(ku,3))^2)/(interfer_power + noise_power));
    end
    RESULT(1,fgh) = 10*log10(avginterfer_power);
    RESULT(2,fgh) = 10*log10(avgrecieved_power);
    RESULT(3,fgh) = sum_rate_FR3
    %% Determine Topology matrix P

    P = zeros(totalBS*numofRUs);
    threshold = 0.1*sqrt(avgrecieved_power);
    for kc = 1:size(cells,2)
        for ku = 2:numofRUs+1
            row = numofRUs*(kc-1)+ku-1; % row number 
            P(row,:) = repelem( cells(kc).RUs(ku,3:totalBS+2) > threshold,numofRUs);
            P(row,numofRUs*(kc-1)+1:numofRUs*(kc-1)+numofRUs) = 1;
            plot([cells(kc).RUs(ku,1),cells(kc).RUs(1,1)],[cells(kc).RUs(ku,2),cells(kc).RUs(1,2)],'k','LineWidth',1.5)
            for kb = 1:totalBS
                if kc ~= kb && P(row, numofRUs*(kb-1)+ku-1)==1
                   plot([cells(kc).RUs(ku,1),cells(kb).RUs(1,1)],[cells(kc).RUs(ku,2),cells(kb).RUs(1,2)],grid_color(kb)) 
                end
            end
        end
    end
    ones(1,114)*P
    P = P>0.5;% each row is the topology of each user
    saveas(gcf,strcat('grid',num2str(fgh),'.png'))
    
    %% Solve WNNM
    % optMap = SolveWNNM(P,'AP');
    % [UMap, sigmaMap, VMap] = svd(optMap); 
    % a1 = sum(sigmaMap>0.1,'all');
    % UMap = UMap(:,1:a1);
    % sigmaMap = sigmaMap(1:a1,1:a1);
    % VMap = VMap(:,1:a1);

    optMap = SolveWNNM(P,'WNNM-ALM');
    [UMap, sigmaMap, VMap] = svd(optMap); 
    a1 = sum(sigmaMap>0.5,'all');
    UMap = UMap(:,1:a1);
    sigmaMap = sigmaMap(1:a1,1:a1);
    VMap = VMap(:,1:a1);
    %[svd(optMap),svd(optMap)]
    %% Sum Rate
    r = a1;
    sigma = sigmaMap;
    U = UMap*sigma;
    V = VMap;

    symbols = randi([0,1],totalBS*numofRUs,1);
    symbols = 2*symbols-1;
    precoded = V'*diag(symbols);
    transmitted_signal = zeros(r,totalBS);
    for ix = 1:totalBS
        transmitted_signal(:,ix) = sum(precoded(:,[numofRUs*(ix-1)+1:numofRUs*ix]),2);
    end
    recieved_signal = transmitted_signal*H + 0.0*randn(r,totalBS*numofRUs);
    decoded_symbols = zeros(totalBS*numofRUs,1);
    interference = zeros(totalBS*numofRUs,1);
    for kc = 1:size(cells,2)
        for ku = 2:numofRUs+1
            ix = numofRUs*(kc-1)+ku-1;
            decoded_symbols(ix) = U(ix,:)*recieved_signal(:,ix)/(cells(kc).RUs(ku,kc+2)); 
        end
    end
    decoded_symbols = decoded_symbols>0;
    decoded_symbols = 2*decoded_symbols-1;
    sum(symbols~=decoded_symbols)

    TIMavgrecieved_power = 0;
    sum_rate_TIM = 0;
    TIMavginterfer_power = 0;
    for ku=2:numofRUs+1
        TIMavgrecieved_power = TIMavgrecieved_power + (norm(cells(1).RUs(ku,3))^2)/numofRUs;
        TIMinterfer_power = 0;
        for kb = 2:size(cells,2)
            if P(ku, numofRUs*(kb-1)+ku-1)==0
                TIMinterfer_power = TIMinterfer_power + (norm(cells(1).RUs(ku,kb+2))^2);
            end
        end
        TIMavginterfer_power = TIMavginterfer_power + TIMinterfer_power/numofRUs;
        sum_rate_TIM = sum_rate_TIM + 1/(r)*log2(1 + (norm(cells(1).RUs(ku,3))^2)/(TIMinterfer_power + noise_power));
    end
    RESULT(4,fgh) = 10*log10(TIMavginterfer_power);
    RESULT(5,fgh) = 10*log10(TIMavgrecieved_power);
    RESULT(6,fgh) = sum_rate_TIM;
    RESULT(7,fgh) = r;
end
save('resultfile.mat','RESULT')