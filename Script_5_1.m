%% SETTING THE SCRIPT
clear
clc

% Remember to change the working directory
cd 'C:\Users\alons\OneDrive\Documentos\Bologna\Clases\Secondo Anno\Advance Time Series Econometrics\First Assignment\Matlab' 

% Setting Latex interpreter
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');  

%% LOADING THE DATA
daily = readtable("NASDAQ Daily.xlsx");
weekly = readtable("NASDAQ Weekly.xlsx"); % This are closing prices of the week

daily = array2timetable(daily.Close, "RowTimes", daily.Date, "VariableNames", "price");
weekly = array2timetable(weekly.Close, "RowTimes", weekly.Date, "VariableNames", "price");

%% PLOT OF PRICES
plot(daily.Time, daily.price, weekly.Time, weekly.price, 'LineWidth', 1.0)
legend({'Daily', 'Weekly'}, 'Location', 'northwest', 'FontSize', 12)
grid on
saveas(gcf, 'Daily and Weekly Levels.png')

%% EVERYTHING IN LOGS
daily.log= log(daily.price);
weekly.log= log(weekly.price);

% Log returns
daily.logreturns = [nan; diff(daily.log)];
weekly.logreturns = [nan; diff(weekly.log)];

% Squared log returns
daily.squared = daily.logreturns.^2;
weekly.squared = weekly.logreturns.^2;

%% PLOT OF LOG LEVELS
plot(daily.Time, daily.log, weekly.Time, weekly.log, 'LineWidth', 1.0)
legend({'Daily', 'Weekly'}, 'Location', 'northwest', 'FontSize', 12)
grid on
saveas(gcf, 'Daily and Weekly Log Levels.png')

%% PLOT OF LOG RETURNS
plot(daily.Time, daily.logreturns, weekly.Time, weekly.logreturns, 'LineWidth', 1.0)
legend({'Daily', 'Weekly'}, 'Location', 'northwest', 'FontSize', 12)
grid on
saveas(gcf, 'Daily and Weekly Log Returns.png')

%% HISTOGRAM OF RETURNS
dailynormalparam = fitdist(daily.logreturns, 'Normal');
weeklynormalparam = fitdist(weekly.logreturns, 'Normal');

% Daily
histfit(daily.logreturns)
legend({'Daily Returns', 'Normal Distribution Fit'}, 'Location', 'northwest', 'FontSize', 10)
grid on
saveas(gcf, 'Daily Returns Histogram.png')

% Weekly
histfit(weekly.logreturns)
legend({'Weekly Returns', 'Normal Distribution Fit'}, 'Location', 'northwest', 'FontSize', 10)
grid on
saveas(gcf, 'Weekly Returns Histogram.png')

%% SQUARED RETURN PLOTS
plot(daily.Time, daily.squared, weekly.Time, weekly.squared, 'LineWidth', 1.0)
legend({'Daily', 'Weekly'}, 'Location', 'northwest', 'FontSize', 12)
grid on
saveas(gcf, 'Daily and Weekly Squared Log Returns.png')

histogram(daily.squared(daily.squared < 0.0001))

%% ARE THE SERIES STATIONARY? DICKEY FULLER TEST
% (*) No test on levels, result of the test is the same one

% Daily data
DFTable1 = adftest(daily, DataVariable = 'log', Model = 'TS', Lags = [0:1:10]); % non-stat
DFTable2 = adftest(daily, DataVariable = 'logreturns', Model = 'AR', Lags = [0:1:10]); % STATIONARY

% Weekly data
DFTable3 = adftest(weekly, DataVariable = 'log', Model = 'TS', Lags = [0:1:10]); % non-stat
DFTable4 = adftest(weekly, DataVariable = 'logreturns', Model = 'AR', Lags = [0:1:10]); % STATIONARY


% Latex Table
DFtex = [];
DFtex = ["Series"; "Daily Prices"; "Daily Returns"; "Weekly Prices"; "Weekly Returns"];

DFtex(1,2) = "P-Value";
DFtex(1,3) = "Null Hypothesis";

DFtex(2,2) = DFTable1.pValue(1);
DFtex(3,2) = DFTable2.pValue(1);
DFtex(4,2) = DFTable3.pValue(1);
DFtex(5,2) = DFTable4.pValue(1);

DFtex(2,3) = "Not Reject";
DFtex(4,3) = DFtex(2,3);
DFtex(3,3) = "Reject";
DFtex(5,3) = DFtex(3,3);

matrix2latex(DFtex, 'DF.tex', 'alignment', 'c');

%% IS THERE SIGNIFICANT SERIAL CORRELATION? THIS NEEDS STATIONARITY
% Ljung-Box test daily data
LBTable1 = lbqtest(daily, DataVariable = 'logreturns', Lags = [1:30]); % correlation
LBTable2 = lbqtest(daily, DataVariable = 'squared', Lags = [1:30]); % correlation

% Ljung-Box test weekly data
LBTable3 = lbqtest(weekly, DataVariable = 'logreturns', Lags = [1:5]); % correlation
LBTable4 = lbqtest(weekly, DataVariable = 'squared', Lags = [1:5]); % correlation


% Latex Tables
DailyLBtex = [];
DailyLBtex = [""; "Returns"; "Returns Squared"]';
DailyLBtex(2,1) = ["Lags"];
DailyLBtex(2,2) = ["P-Value"];
DailyLBtex(2,3) = ["P-Value"];

for lb = 1:10
    DailyLBtex(lb + 2,1) = LBTable1.Lags(lb);
    DailyLBtex(lb + 2,2) = round(LBTable1.pValue(lb),3);
    DailyLBtex(lb + 2,3) = round(LBTable2.pValue(lb),3);
end
matrix2latex(DailyLBtex, 'Daily LB.tex', 'alignment', 'c');


WeeklyLBtex = [];
WeeklyLBtex = [""; "Returns"; "Returns Squared"]';
WeeklyLBtex(2,1) = ["Lags"];
WeeklyLBtex(2,2) = ["P-Value"];
WeeklyLBtex(2,3) = ["P-Value"];

for lb = 1:5
    WeeklyLBtex(lb + 2,1) = LBTable3.Lags(lb);
    WeeklyLBtex(lb + 2,2) = round(LBTable3.pValue(lb),3);
    WeeklyLBtex(lb + 2,3) = round(LBTable4.pValue(lb),3);
end
matrix2latex(WeeklyLBtex, 'Weekly LB.tex', 'alignment', 'c');

%% AUTOCORRELATION PLOTS
dailylogreturns_acf = autocorr(daily.logreturns, NumLags = 300);
dailylogreturns_acf(1,1) = nan;

dailylogsquared_acf = autocorr(daily.squared, NumLags = 300);
dailylogsquared_acf(1,1) = nan;

weeklylogreturns_acf = autocorr(weekly.logreturns, NumLags = 50);
weeklylogreturns_acf(1,1) = nan;

weeklylogsquared_acf = autocorr(weekly.squared, NumLags = 50);
weeklylogsquared_acf(1,1) = nan;

% Daily data autocorrelograms
plot(dailylogreturns_acf, 'LineWidth', 1.0)
axis([0 300 -0.1 0.1])
ylabel('Sample Autocorrelation')
xlabel('Lag')
grid on
saveas(gcf, 'Daily Returns Correlogram.png')

plot(dailylogsquared_acf, 'LineWidth', 1.0)
axis([0 300 0 0.4])
ylabel('Sample Autocorrelation')
xlabel('Lag')
grid on
saveas(gcf, 'Daily Squared Returns Correlogram.png')


% Weekly data autocorrelograms
plot(weeklylogreturns_acf, 'LineWidth', 1.0)
axis([0 50 -0.1 0.1])
ylabel('Sample Autocorrelation')
xlabel('Lag')
grid on
saveas(gcf, 'Weekly Returns Correlogram.png')

plot(weeklylogsquared_acf, 'LineWidth', 1.0)
axis([0 50 0 0.4])
ylabel('Sample Autocorrelation')
xlabel('Lag')
grid on
saveas(gcf, 'Weekly Squared Returns Correlogram.png')


%% GAUSSIANITY? JARQUE-BERA TEST
% Watch out, I think the H0 assumes iid, which we are rejecting on the
% previous section



%% KURTOSIS AND SKEWNESS
kurtdaily = kurtosis(daily.logreturns);
kurtweekly = kurtosis(weekly.logreturns);

skewdaily = skewness(daily.logreturns);
skewweekly = skewness(weekly.logreturns);

%% HILL ESTIMATOR. DAILY DATA

% Right data
dailyordered = daily.logreturns(2:end,1);
dailyordered = sort(dailyordered, 'descend');
dailyordered = array2table(dailyordered, "VariableNames", "right");
aux = height(dailyordered);
dailyordered.position = (1:aux).';
dailyordered.percentage = dailyordered.position/aux;
dailyordered.log = log(dailyordered.percentage);

% Left data
daily_left = daily.logreturns(2:end,1);
daily_left = sort(daily_left, 'ascend');
dailyordered.left = daily_left;
q1 = round(0.05*aux);
iota_right = zeros(q1,1);

% Right Tail
for i = 1:q1
  iota_right(i) = mean(log(dailyordered.right(1:i)/dailyordered.right(i+1)));
end
hill_right = iota_right.^-1;

% Left Tail
iota_left =  zeros(q1,1);
dailyordered.left = -dailyordered.left; % absolute value
for i = 1:q1
  iota_left(i) = mean(log(dailyordered.left(1:i)/dailyordered.left(i+1)));
end
hill_left = iota_left.^-1;


% Graph
plot(hill_left, 'LineWidth', 1.0)
axis([0 q1 0 10])
ylabel('$\theta^{right}$', 'FontSize', 16)
xlabel('Sample Size')
grid on
hold on
plot(hill_right, 'LineWidth', 1.0)
legend({'Left Tail Index', 'Right Tail Index'}, 'Location', 'northeast', 'FontSize', 12)
saveas(gcf, 'Daily Hill.png')
hold off


%% HILL ESTIMATOR. WEEKLY DATA

% Right data
weeklyordered = weekly.logreturns(2:end,1);
weeklyordered = sort(weeklyordered, 'descend');
weeklyordered = array2table(weeklyordered, "VariableNames", "right");
aux = height(weeklyordered);
weeklyordered.position = (1:aux).';
weeklyordered.percentage = weeklyordered.position/aux;
weeklyordered.log = log(weeklyordered.percentage);

% Left data
weekly_left = weekly.logreturns(2:end,1);
weekly_left = sort(weekly_left, 'ascend');
weeklyordered.left = weekly_left;
q1 = round(0.05*aux);
iota_right = zeros(q1,1);

% Right Tail
for i = 1:q1
  iota_right(i) = mean(log(weeklyordered.right(1:i)/weeklyordered.right(i+1)));
end
hill_right = iota_right.^-1;

% Left Tail
iota_left =  zeros(q1,1);
weeklyordered.left = -weeklyordered.left; %absolute value
for i = 1:q1
  iota_left(i) = mean(log(weeklyordered.left(1:i)/weeklyordered.left(i+1)));
end
hill_left = iota_left.^-1;


% Graph
plot(hill_left, 'LineWidth', 1.0)
axis([0 q1 0 10])
ylabel('$\theta$', 'FontSize', 16)
xlabel('Sample Size')
grid on
hold on
plot(hill_right, 'LineWidth', 1.0)
legend({'Left Tail Index', 'Right Tail Index'}, 'Location', 'northeast', 'FontSize', 12)
saveas(gcf, 'Weekly Hill.png')
hold off

%% RANK-SIZE REGRESSION: DAILY DATA
% Estimation grid

grid = [0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01 0.02 0.03 0.04 0.05];
tabledaily = [];
aux = height(dailyordered);
q1 = round(0.05*aux);

for i = grid

    % i% of observations
    q1 = round(i*aux); %
    Y = dailyordered.log(1:q1);
    
    % Right tail
    X_right = log(dailyordered.right(1:q1));
    X_right = [X_right ones(q1,1)];
    alpha_right = X_right\Y; % -2.79
    
    % Left tail
    X_left = log((-1)*dailyordered.left(1:q1));
    X_left = [X_left ones(q1,1)];
    alpha_left = X_left\Y; % -3

    if i < 0.01
        tabledaily(i*1000,1) = i;
        tabledaily(i*1000,2) = q1;
        tabledaily(i*1000,3) = -round(alpha_left(1,1),2);
        tabledaily(i*1000,4) = -round(alpha_right(1,1),2);
    
    elseif i >= 0.01
        tabledaily(i*100 + 9,1) = i;
        tabledaily(i*100 + 9,2) = q1;
        tabledaily(i*100 + 9,3) = -round(alpha_left(1,1),2);
        tabledaily(i*100 + 9,4) = -round(alpha_right(1,1),2);

    end
end
tabledaily(:,5) = tabledaily(:,2);
tabledaily(:,6) = (tabledaily(:,1) - 1)*(-1);


alphatable = array2table(tabledaily, "VariableNames", ["q1 Left" "Observations" "$\theta_{left}$" "$\theta_{right}$" "Observations2" "q1 Right"]);

table2latex(alphatable, 'Daily Tail Index Regression');


%% RANK-SIZE REGRESSION: WEEKLY DATA
% Estimation grid

grid = [0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01 0.02 0.03 0.04 0.05];
tableweekly = [];
aux = height(weeklyordered);
q1 = round(0.05*aux);


for i = grid

    % i% of observations
    q1 = round(i*aux); %
    Y = weeklyordered.log(1:q1);
    
    % Right tail
    X_right = log(weeklyordered.right(1:q1));
    X_right = [X_right ones(q1,1)];
    alpha_right = X_right\Y; % -2.79
    
    % Left tail
    X_left = log((-1)*weeklyordered.left(1:q1));
    X_left = [X_left ones(q1,1)];
    alpha_left = X_left\Y; % -3

    if i < 0.01
        tableweekly(i*1000,1) = i;
        tableweekly(i*1000,2) = q1;
        tableweekly(i*1000,3) = -round(alpha_left(1,1),2);
        tableweekly(i*1000,4) = -round(alpha_right(1,1),2);
    
    elseif i >= 0.01
        tableweekly(i*100 + 9,1) = i;
        tableweekly(i*100 + 9,2) = q1;
        tableweekly(i*100 + 9,3) = -round(alpha_left(1,1),2);
        tableweekly(i*100 + 9,4) = -round(alpha_right(1,1),2);

    end
end
tableweekly(:,5) = tableweekly(:,2);
tableweekly(:,6) = (tableweekly(:,1) - 1)*(-1);


alphatable = array2table(tableweekly, "VariableNames", ["q1 Left" "Observations" "$\theta_{left}$" "$\theta_{right}$" "Observations2" "q1 Right"]);

table2latex(alphatable, 'Weekly Tail Index Regression');

%% DAILY CUSUM TEST, THE FEARED (LORETAN AND PHILLIPS 1994)

% We need consistent residuals, let's use an AR(1) process for now,
% similarly as done in the paper. 

AR = estimate(regARIMA(100,0,0), daily.logreturns(2:end),'Display', 'off');
CUSUMdata = [];
[E, U, V, logL] = infer(AR, daily.logreturns(2:end));

CUSUMdata(:,1) = U;
CUSUMdata(:,2) = CUSUMdata(:,1).^2;
CUSUMdata(:,3) = mean(CUSUMdata(:,2));
CUSUMdata(:,4) = CUSUMdata(:,2) - CUSUMdata(:,3);
s = height(CUSUMdata);

% Now CUSUM test, although very similar since ARs don't work well for
% explaining returns because of the random walk

lags = 3000;
covs = autocorr(CUSUMdata(:,2), 'NumLags', lags)*var(CUSUMdata(:,2));

vu2 = 0;
for j = 1:lags
    vu2 = vu2 + 2*(1-j/(lags + 1))*covs(j+1,1);
end
gamma0 = var(CUSUMdata(:,2));
vu2 = vu2 + gamma0;

r = [0.1:0.1:0.9]';
r(:,2) = round(r*s);
criticalvalues = -[0.67,0.79,0.81,0.80,0.77,0.72,0.63,0.51,0.34]'; % This critical values are obtained from the paper
r(:,3) = criticalvalues;

test = nan(s,1);
for k = r(:,2)
    test(k,1) = r(:,3);
end

Z_stat = [];
for u =  1:s
    Z_stat(u,1) = sum((CUSUMdata(1:u,4)))/sqrt(s*vu2);
end

plot(daily.Time(2:end),Z_stat)
hold on
plot(daily.Time(2:end),test, "o")
grid on
ylabel('CUSUM')
xlabel('Date')
legend({'CUSUM', 'Critical Values for $\theta$ = 2.5'}, 'Location', 'northwest')
hold off
saveas(gcf, 'Daily CUSUM.png')


%% WEEKLY CUSUM TEST, THE FEARED (LORETAN AND PHILLIPS 1994)

% We need consistent residuals, let's use an AR(1) process for now,
% similarly as done in the paper. 

AR = estimate(regARIMA(1,0,0), weekly.logreturns(2:end),'Display', 'off');
CUSUMdata = [];
[E, U, V, logL] = infer(AR, weekly.logreturns(2:end));

CUSUMdata(:,1) = U;
CUSUMdata(:,2) = CUSUMdata(:,1).^2;
CUSUMdata(:,3) = mean(CUSUMdata(:,2));
CUSUMdata(:,4) = CUSUMdata(:,2) - CUSUMdata(:,3);
s = height(CUSUMdata);

% Now CUSUM test, although very similar since ARs don't work well for
% explaining returns because of the random walk

lags = 500;
covs = autocorr(CUSUMdata(:,2), 'NumLags', lags)*var(CUSUMdata(:,2));

vu2 = 0;
for j = 1:lags
    vu2 = vu2 + 2*(1-j/(lags + 1))*covs(j+1,1);
end
gamma0 = var(CUSUMdata(:,2));
vu2 = vu2 + gamma0;

r = [0.1:0.1:0.9]';
r(:,2) = round(r*s);
criticalvalues = -[0.67,0.79,0.81,0.80,0.77,0.72,0.63,0.51,0.34]'; % This critical values are obtained from the paper
r(:,3) = criticalvalues;

test = nan(s,1);
for k = r(:,2)
    test(k,1) = r(:,3);
end

Z_stat = [];
for u =  1:s
    Z_stat(u,1) = sum((CUSUMdata(1:u,4)))/sqrt(s*vu2);
end

plot(weekly.Time(2:end),Z_stat)
hold on
plot(weekly.Time(2:end),test, "o")
grid on
ylabel('CUSUM')
xlabel('Date')
legend({'CUSUM', 'Critical Values for $\theta$ = 2.5'}, 'Location', 'northwest')
hold off
saveas(gcf, 'Weekly CUSUM.png')



%% DAILY: FITTING 25 GARCHs, GJRs AND EGARCHs TO CHOOSE THE BEST 
% ONE OF EACH

% OCCHIO, FOR MATLAB THE NOTATION IS INVERSED FROM OURS

twothirds = round(1:2/3*height(daily))'; % 8705
onethird = round(2/3*height(daily):height(daily))';

dailyforgarch = daily(twothirds,:);
daily_offsample = daily(onethird,:);

rdaily = dailyforgarch.logreturns(2:end);

rdaily_offsample = daily_offsample.logreturns;
daily_offsample_squared = daily_offsample.squared;


luigi_garch_aic = [];
luigi_garch_bic = [];

luigi_gjr_aic = [];
luigi_gjr_bic = [];

luigi_egarch_aic = [];
luigi_egarch_bic = [];

tic
for j = 1:5
    for i = 1:5
        model1 = garch('GARCHLags',i,'ARCHLags',j,'Constant', NaN);
        model2 = gjr('GARCHLags',i,'ARCHLags',j, 'Constant', NaN,'Leverage', NaN);
        model3 = egarch('GARCHLags',i,'ARCHLags',j, 'Leverage', NaN, 'Constant', NaN);
        [estimates1] = estimate(model1,rdaily, 'Display','off');
        [estimates2] = estimate(model2,rdaily, 'Display','off');
        [estimates3] = estimate(model3,rdaily, 'Display','off');
        luigi_garch_aic(i,j) = summarize(estimates1).AIC;
        luigi_gjr_aic(i,j) = summarize(estimates2).AIC;
        luigi_egarch_aic(i,j) = summarize(estimates3).AIC;
        luigi_garch_bic(i,j) = summarize(estimates1).BIC;
        luigi_gjr_bic(i,j) = summarize(estimates2).BIC;
        luigi_egarch_bic(i,j) = summarize(estimates3).BIC;
    end
end
toc

% Minimum information criteria for the GARCH
minimumAIC = min(min(luigi_garch_aic));
[garch_i,arch_i] = find(luigi_garch_aic == minimumAIC);

minimumBIC = min(min(luigi_garch_bic));
[garch_i,arch_i] = find(luigi_garch_bic == minimumBIC)
% For GARCH (3,1) is the minimum for both criteria


% Minimum information criteria for the GJR
minimumAIC = min(min(luigi_gjr_aic));
[garch_i,arch_i] = find(luigi_gjr_aic == minimumAIC);

minimumBIC = min(min(luigi_gjr_bic));
[garch_i,arch_i] = find(luigi_gjr_bic == minimumBIC)
% For GJR (4,4) is the minimum for AIC and (3,1) BIC


% Minimum information criteria for the EGARCH
minimumAIC = min(min(luigi_egarch_aic));
[garch_i,arch_i] = find(luigi_egarch_aic == minimumAIC);

minimumBIC = min(min(luigi_egarch_bic));
[garch_i,arch_i] = find(luigi_egarch_bic == minimumBIC)
% For EGARCH (4,5) is the minimum for AIC and (2,2) for BIC


% Only using the BIC, our 3 candidate models are: 
% GARCH(1,1)
% GJR(1,1)
% EGARCH(1,1)

garchBIC = [luigi_garch_bic(3,1) luigi_gjr_bic(3,1) luigi_egarch_bic(3,1)];
% The smallest is the GJR, is the best in-sample. 



%% DAILY GJR
garchlags = 1;
archlags = 1;

gjrmodel = gjr('GARCHLags',garchlags,'ARCHLags',archlags,'Distribution','Gaussian','Constant',NaN,'Leverage', NaN);
[gjrestimates] = estimate(gjrmodel,rdaily,'Display','off');

gjr_condvar_insample = infer(gjrestimates,rdaily);
gjr_condvol_insample = sqrt(gjr_condvar_insample);

rdaily_offsample = daily_offsample.logreturns;
gjr_condvar_offsample = infer(gjrestimates,rdaily_offsample);
gjr_condvol_offsample = sqrt(gjr_condvar_offsample);

dailygjr_residuals = 100*(daily_offsample_squared - gjr_condvar_offsample);
dailygjr_mse = dailygjr_residuals'*dailygjr_residuals;


%% DAILY: WORKING ON THE GJR(1,1)

gjr_output = summarize(gjrestimates).Table(1:4,:);
gjr_output2 = gjr_output;
gjr_output2{:,:} = round(gjr_output.Variables,2);

table2latex(gjr_output2, 'GJR11.tex');

% Volatility Graphs
plot(dailyforgarch.Time(2:end),rdaily)
hold on
plot(dailyforgarch.Time(2:end),gjr_condvol_insample,'LineWidth', 1.5)
xlabel("Time")
ylabel("Returns and Conditional Volatility")
legend({'Returns', 'Volatility ($\hat{\sigma}_t$)'}, 'Location', 'southwest', 'FontSize', 12)
hold off
grid on
saveas(gcf,"Daily GJR Volatility In-Sample.png")

plot(daily_offsample.Time,rdaily_offsample)
hold on
plot(daily_offsample.Time,gjr_condvol_offsample,'LineWidth', 1.5)
xlabel("Time")
ylabel("Returns and Conditional Volatility")
legend({'Returns', 'Volatility ($\hat{\sigma}_t$)'}, 'Location', 'southwest', 'FontSize', 12)
hold off 
grid on
saveas(gcf,"Daily GJR Volatility Off-Sample.png")

% Conditional Variance Graphs
plot(dailyforgarch.Time(2:end),dailyforgarch.squared(2:end))
hold on
plot(dailyforgarch.Time(2:end),gjr_condvar_insample,'LineWidth', 1.5)
xlabel("Time")
ylabel("Returns and Conditional Variance")
legend({'Returns', 'Variance ($\hat{\sigma}_t^2$)'}, 'Location', 'northwest', 'FontSize', 12)
hold off 
grid on
saveas(gcf,"Daily GJR Variance In-Sample.png")

plot(daily_offsample.Time,daily_offsample_squared)
hold on
plot(daily_offsample.Time,gjr_condvar_offsample,'LineWidth', 1.5)
xlabel("Time")
ylabel("Returns and Conditional Variance")
legend({'Returns', 'Variance ($\hat{\sigma}_t^2$)'}, 'Location', 'northwest', 'FontSize', 12)
hold off 
grid on
saveas(gcf,"Daily GJR Variance Off-Sample.png")


%% GJR(1,1) MISSPECIFICATION ANALYSIS

% Residuals
gjr_residuals = dailyforgarch.logreturns(2:end)./gjr_condvol_insample;

% Plot of the residuals
plot(dailyforgarch.Time(2:end),gjr_residuals)
grid on
xlabel("Time")
ylabel("Standardized Residuals")
saveas(gcf,"Daily STD Residuals.png")

% Histogram of the residuals
histfit(gjr_residuals)
grid on
saveas(gcf,"Daily Histogram STD Residuals.png")

% Jarque Bera Test
[h,p,jbstat,critval] = jbtest(gjr_residuals,0.01); % refuse H0 at 1%
JBTableGJR = [h,p,jbstat,critval];

% Autocorrelogram
autocorr(gjr_residuals, NumLags = 30)
axis([1 30 -0.05 0.25])
title('')
saveas(gcf,"Daily STD Residuals Correlogram.png")

% Ljung-Box Test
resTable = array2table(gjr_residuals);
LB_GJRresiduals = lbqtest(resTable, Lags = [1:30]); % correlation

%% 5-STEP FORECAST.
T = height(dailyforgarch);
n = 8705;
%13060-n=4355

%5steps-ahead

for_ret_5steps=[];

for i=0:4350;
    for_ret_5steps(i+1,1:5)=forecast(gjrestimates,5,daily.logreturns(1:n+i))';
end;
for_vol=sqrt(for_ret_5steps(:,5));

%plot of actual returns vs 1step ahead forecast vs 5 step ahead forecast
plot(daily_offsample.Time(4:end),daily.logreturns(n+5:end))
hold on
plot(daily_offsample.Time(4:end),for_vol, 'color', 'yellow','LineWidth', 1.5)
legend({'Returns','Volatility ($\hat{\sigma}_t$)'}, 'Location', 'northwest', 'FontSize', 12)
xlabel("Time")
ylabel("Returns and Conditional Volatility")
hold off
saveas(gcf, 'forecast comparisons.png')


for_onestep_squared=gjr_condvol_offsample.^2;
plot(daily_offsample.Time(4:end),daily.squared(n+5:end))
hold on
plot(daily_offsample.Time(4:end),for_ret_5steps(:,5),'color', 'yellow','LineWidth', 1.5);
xlabel("Time")
ylabel("Returns and Conditional Variance")
legend({'Returns','Variance ($\hat{\sigma}_t^2$)'}, 'Location', 'northwest', 'FontSize', 12)
hold off
saveas(gcf,'variance forecast comparisons.png')



%% WEEKLY: FITTING 25 GARCHs, GJRs AND EGARCHs TO CHOOSE THE BEST 
% ONE OF EACH

% OCCHIO, FOR MATLAB THE NOTATION IS INVERSED FROM OURS

twothirds = round(1:2/3*height(weekly))'; % 1802
onethird = round(2/3*height(weekly):height(weekly))';

weeklyforgarch = weekly(twothirds,:);
weekly_offsample = weekly(onethird,:);

rweekly = weeklyforgarch.logreturns(2:end);

rweekly_offsample = weekly_offsample.logreturns;
weekly_offsample_squared = weekly_offsample.squared;


luigi_garch_aic = [];
luigi_garch_bic = [];

luigi_gjr_aic = [];
luigi_gjr_bic = [];

luigi_egarch_aic = [];
luigi_egarch_bic = [];

tic
for j = 1:5
    for i = 1:5
        model1 = garch('GARCHLags',i,'ARCHLags',j,'Constant', NaN);
        model2 = gjr('GARCHLags',i,'ARCHLags',j, 'Constant', NaN,'Leverage', NaN);
        model3 = egarch('GARCHLags',i,'ARCHLags',j, 'Leverage', NaN, 'Constant', NaN);
        [estimates1] = estimate(model1,rweekly, 'Display','off');
        [estimates2] = estimate(model2,rweekly, 'Display','off');
        [estimates3] = estimate(model3,rweekly, 'Display','off');
        luigi_garch_aic(i,j) = summarize(estimates1).AIC;
        luigi_gjr_aic(i,j) = summarize(estimates2).AIC;
        luigi_egarch_aic(i,j) = summarize(estimates3).AIC;
        luigi_garch_bic(i,j) = summarize(estimates1).BIC;
        luigi_gjr_bic(i,j) = summarize(estimates2).BIC;
        luigi_egarch_bic(i,j) = summarize(estimates3).BIC;
    end
end
toc

% Minimum information criteria for the GARCH
minimumAIC = min(min(luigi_garch_aic));
[garch_i,arch_i] = find(luigi_garch_aic == minimumAIC);

minimumBIC = min(min(luigi_garch_bic));
[garch_i,arch_i] = find(luigi_garch_bic == minimumBIC)
% For GARCH (3,1) is the minimum for both criteria


% Minimum information criteria for the GJR
minimumAIC = min(min(luigi_gjr_aic));
[garch_i,arch_i] = find(luigi_gjr_aic == minimumAIC);

minimumBIC = min(min(luigi_gjr_bic));
[garch_i,arch_i] = find(luigi_gjr_bic == minimumBIC)
% For GJR (4,4) is the minimum for AIC and (3,1) BIC


% Minimum information criteria for the EGARCH
minimumAIC = min(min(luigi_egarch_aic));
[garch_i,arch_i] = find(luigi_egarch_aic == minimumAIC);

minimumBIC = min(min(luigi_egarch_bic));
[garch_i,arch_i] = find(luigi_egarch_bic == minimumBIC)
% For EGARCH (4,5) is the minimum for AIC and (2,2) for BIC


% Only using the BIC, our 3 candidate models are: 
% GARCH(1,1)
% GJR(1,1)
% EGARCH(1,1)

garchBIC = [luigi_garch_bic(3,1) luigi_gjr_bic(3,1) luigi_egarch_bic(3,1)]
% The smallest is the GARCH, is the best in-sample. 

%% WEEKLY GARCH
garchlags = 1;
archlags = 1;

garchmodel = garch('GARCHLags',garchlags,'ARCHLags',archlags,'Distribution','Gaussian','Constant',NaN);
[garchestimates] = estimate(garchmodel,rweekly,'Display','off');

garch_condvar_insample = infer(garchestimates,rweekly);
garch_condvol_insample = sqrt(garch_condvar_insample);

garch_condvar_offsample = infer(garchestimates,rweekly_offsample);
garch_condvol_offsample = sqrt(garch_condvar_offsample);

weeklygarch_residuals = 100*(weekly_offsample_squared-garch_condvar_offsample);
weeklygarch_mse = weeklygarch_residuals'*weeklygarch_residuals;

%% WEEKLY: WORKING ON THE GARCH(1,1)

garch_output = summarize(garchestimates).Table(1:3,:);
garch_output2 = garch_output;
garch_output2{:,:} = round(garch_output.Variables,2);

table2latex(garch_output2, 'GARCH11.tex');

% Volatility Graphs
plot(weeklyforgarch.Time(2:end),rweekly)
hold on
plot(weeklyforgarch.Time(2:end),garch_condvol_insample,'LineWidth', 1.5)
xlabel("Time")
ylabel("Returns and Conditional Volatility")
legend({'Returns', 'Volatility ($\hat{\sigma}_t$)'}, 'Location', 'southwest', 'FontSize', 12)
hold off
grid on
saveas(gcf,"Weekly GARCH Volatility In-Sample.png")

plot(weekly_offsample.Time,rweekly_offsample)
hold on
plot(weekly_offsample.Time,garch_condvol_offsample,'LineWidth', 1.5)
xlabel("Time")
ylabel("Returns and Conditional Volatility")
legend({'Returns', 'Volatility ($\hat{\sigma}_t$)'}, 'Location', 'southwest', 'FontSize', 12)
hold off 
grid on
saveas(gcf,"Weekly GARCH Volatility Off-Sample.png")

% Conditional Variance Graphs
plot(weeklyforgarch.Time(2:end),weeklyforgarch.squared(2:end))
hold on
plot(weeklyforgarch.Time(2:end),garch_condvar_insample,'LineWidth', 1.5)
xlabel("Time")
ylabel("Returns and Conditional Variance")
legend({'Returns', 'Variance ($\hat{\sigma}_t^2$)'}, 'Location', 'northwest', 'FontSize', 12)
hold off 
grid on
saveas(gcf,"Weekly GARCH Variance In-Sample.png")

plot(weekly_offsample.Time,weekly_offsample_squared)
hold on
plot(weekly_offsample.Time,garch_condvar_offsample,'LineWidth', 1.5)
xlabel("Time")
ylabel("Returns and Conditional Variance")
legend({'Returns', 'Variance ($\hat{\sigma}_t^2$)'}, 'Location', 'northwest', 'FontSize', 12)
hold off 
grid on
saveas(gcf,"Weekly GARCH Variance Off-Sample.png")

%% GARCH(1,1) MISSPECIFICATION ANALYSIS

% Residuals
garch_residuals = weeklyforgarch.logreturns(2:end)./garch_condvol_insample;

% Plot of the residuals
plot(weeklyforgarch.Time(2:end),garch_residuals)
grid on
xlabel("Time")
ylabel("Standardized Residuals")
saveas(gcf,"Weekly STD Residuals.png")

% Histogram of the residuals
histfit(garch_residuals)
grid on
saveas(gcf,"Weekly Histogram STD Residuals.png")

% Jarque Bera Test
[h,p,jbstat,critval] = jbtest(garch_residuals,0.01); % refuse H0 at 1%
JBTableGARCH = [h,p,jbstat,critval];

% Autocorrelogram
autocorr(garch_residuals, NumLags = 30)
axis([1 30 -0.05 0.25])
title('')
saveas(gcf,"Weekly STD Residuals Correlogram.png")

% Ljung-Box Test
resTable = array2table(garch_residuals);
LB_GARCHresiduals = lbqtest(resTable, Lags = [1:30]); % correlation


%% DAILY VALUE-AT-RISK

% Gaussian case
start = find(year(daily.Time)==1973,1);
window = start : height(daily);
tradingdays = 250;

pVaR = [0.05 0.01];
zscore = norminv(pVaR);
normal95 = zeros(length(window),1);
normal99 = zeros(length(window),1);
for t = window
    i = t - start + 1;
    period = t - tradingdays:t-1;
    sigma = std(daily.logreturns(period));
    normal95(i) = -zscore(1)*sigma;
    normal99(i) = -zscore(2)*sigma;
end

plot(daily.Time(window), normal95)
xlabel('Date')
ylabel('VaR(5\%)')
title('VaR(5\%) Estimation with Normal Distribution')


% With GJR estimates
gjr_condvol = cat(1, gjr_condvol_insample, gjr_condvol_offsample);

VaR_gjr95 = zeros(length(window),1);
VaR_gjr99 = zeros(length(window),1);
for t = window
    i = t - start + 1;
    sigma = gjr_condvol(t-1);
    VaR_gjr95(i) = -zscore(1)*sigma;
    VaR_gjr99(i) = -zscore(2)*sigma;

end

plot(daily.Time(window), VaR_gjr95)
xline(daily.Time(length(gjr_condvol_insample)), ':')
xlabel('Date')
ylabel('VaR(5\%)')
title('VaR(5\%) Estimation with GJR estimates')



% Historical distribution
historical95 = zeros(length(window),1);
historical99 = zeros(length(window),1);
for t = window
    i = t - start + 1;
    period = t - tradingdays:t-1;
    X = daily.logreturns(period);
    historical95(i) = -quantile(X,0.05);
    historical99(i) = -quantile(X,0.01);
end

plot(daily.Time(window), historical95)
xlabel('Date')
ylabel('VaR(5\%)')
title('VaR(5\%) Estimation with historical distribution')


% Exponential Weighted Moving Average Method (EWMA)
lambda = 0.94;
sigma2     = zeros(length(daily.logreturns),1);
sigma2(1:2)  = daily.logreturns(2)^2;

for i = 3 : (start-1)
    sigma2(i) = (1-lambda) * daily.logreturns(i)^2 + lambda * sigma2(i-1);
end

EWMA95 = zeros(length(window),1);
EWMA99 = zeros(length(window),1);
for t = window
    k = t - start + 1;
    sigma2(t) = (1-lambda) * daily.logreturns(t)^2 + lambda * sigma2(t-1);
    sigma_ewma = sqrt(sigma2(t));
    EWMA95(k) = -zscore(1)*sigma_ewma;
    EWMA99(k) = -zscore(2)*sigma_ewma;
end

plot(daily.Time(window),EWMA95)
ylabel('VaR')
xlabel('Date')
title('VaR(5\%) Estimation with EWMA Method')


% Visualizing the differences
test_returns = daily.logreturns(window);
test_time   = daily.Time(window);
plot(test_time,[test_returns -normal95 -VaR_gjr95 -historical95 -EWMA95])
ylabel('VaR')
xlabel('Date')
legend({'Returns','Normal', 'GJR', 'Historical','EWMA'},'Location','Best')
title('Comparison of returns and VaR at 95\% for different models')


% Number of violations

dailyvbt = varbacktest(test_returns,[normal95 VaR_gjr95 historical95 EWMA95 normal99 VaR_gjr99 historical99 EWMA99], ...
    'PortfolioID','NASDAQ',...
    'VaRID',{'normal95', 'VaR_gjr95' 'historical95','EWMA95', 'normal99', 'VaR_gjr99' 'historical99','EWMA99'},...
    'VaRLevel',[0.95 0.95 0.95 0.95 0.99 0.99 0.99 0.99]);
summary(dailyvbt)

zoom   = (test_time >= datetime(2000,1,1)) & (test_time <= datetime(2000,6,1));
VaRData   = [-VaR_gjr95(zoom) -historical95(zoom) -EWMA95(zoom)];
VaRFormat = {'-','-','-'};
D = test_time(zoom);
R = test_returns(zoom);
G = VaR_gjr95(zoom);
H = historical95(zoom);
E = EWMA95(zoom);
IndG95    = (R < -G);
IndHS95   = (R < -H);
IndEWMA95 = (R < -E);
bar(D,R,0.5,'FaceColor',[0.7 0.7 0.7]);
hold on
for i = 1 : size(VaRData,2)
    stairs(D-0.5,VaRData(:,i),VaRFormat{i});
end
ylabel('VaR')
xlabel('Date')
legend({'Returns','GJR','Historical','EWMA'},'Location','Best','AutoUpdate','Off')
% title('VaR(5\%) Violations for Different Models')
ax = gca;
ax.ColorOrderIndex = 1;

plot(D(IndG95),-G(IndG95),'o','MarkerSize',8, "MarkerEdgeColor",[0.8500 0.3250 0.0980],'LineWidth',1.5)
plot(D(IndHS95),-H(IndHS95),'diamond','MarkerSize',8, "MarkerEdgeColor",[0.9290 0.6940 0.1250],'LineWidth',1.5)
plot(D(IndEWMA95),-E(IndEWMA95),'^','MarkerSize',8, "MarkerEdgeColor",[0 0.4470 0.7410],'LineWidth',1.5)
xlim([D(1)-1, D(end)+1])
hold off;
saveas(gcf, "Daily VaR.png")


%% WEEKLY VALUE-AT-RISK 

% Gaussian case
start = find(year(weekly.Time)==1973,1);
window = start : height(weekly);
tradingdays = 50;

pVaR = [0.05 0.01];
zscore = norminv(pVaR);
normal95 = zeros(length(window),1);
normal99 = zeros(length(window),1);
for t = window
    i = t - start + 1;
    period = t - tradingdays:t-1;
    sigma = std(weekly.logreturns(period));
    normal95(i) = -zscore(1)*sigma;
    normal99(i) = -zscore(2)*sigma;
end

plot(weekly.Time(window), normal95)
xlabel('Date')
ylabel('VaR(5\%)')
title('VaR(5\%) Estimation with Normal Distribution')


% With GARCH estimates
garch_condvol = cat(1, garch_condvol_insample, garch_condvol_offsample);

VaR_garch95 = zeros(length(window),1);
VaR_garch99 = zeros(length(window),1);

for t = window
    i = t - start + 1;
    sigma = garch_condvol(t-1);
    VaR_garch95(i) = -zscore(1)*sigma;
    VaR_garch99(i) = -zscore(2)*sigma;

end

plot(weekly.Time(window), VaR_garch95)
xline(weekly.Time(length(garch_condvol_insample)), ':')
xlabel('Date')
ylabel('VaR(5\%)')
title('VaR(5\%) Estimation with garch estimates')


% Historical distribution
historical95 = zeros(length(window),1);
historical99 = zeros(length(window),1);
for t = window
    i = t - start + 1;
    period = t - tradingdays:t-1;
    X = weekly.logreturns(period);
    historical95(i) = -quantile(X,0.05);
    historical99(i) = -quantile(X,0.01);
end

plot(weekly.Time(window), historical95)
xlabel('Date')
ylabel('VaR(5\%)')
title('VaR(5\%) Estimation with historical distribution')


% Exponential Weighted Moving Average Method (EWMA)
lambda = 0.94;
sigma2     = zeros(length(weekly.logreturns),1);
sigma2(1:2)  = weekly.logreturns(2)^2;

for i = 3 : (start-1)
    sigma2(i) = (1-lambda) * weekly.logreturns(i)^2 + lambda * sigma2(i-1);
end

EWMA95 = zeros(length(window),1);
EWMA99 = zeros(length(window),1);
for t = window
    k = t - start + 1;
    sigma2(t) = (1-lambda) * weekly.logreturns(t)^2 + lambda * sigma2(t-1);
    sigma_ewma = sqrt(sigma2(t));
    EWMA95(k) = -zscore(1)*sigma_ewma;
    EWMA99(k) = -zscore(2)*sigma_ewma;
end

plot(weekly.Time(window),EWMA95)
ylabel('VaR')
xlabel('Date')
title('VaR(5\%) Estimation with EWMA Method')


% Visualizing the differences
test_returns = weekly.logreturns(window);
test_time   = weekly.Time(window);
plot(test_time,[test_returns -normal95 -VaR_garch95 -historical95 -EWMA95])
ylabel('VaR')
xlabel('Date')
legend({'Returns','Normal', 'GARCH', 'Historical','EWMA'},'Location','Best')
title('Comparison of returns and VaR at 95\% for different models')


% Number of violations
vbtweekly = varbacktest(test_returns,[normal95 VaR_garch95 historical95 EWMA95 normal99 VaR_garch99 historical99 EWMA99], ...
    'PortfolioID','NASDAQ',...
    'VaRID',{'normal95', 'VaR_garch95' 'historical95','EWMA95', 'normal99', 'VaR_garch99' 'historical99','EWMA99'},...
    'VaRLevel',[0.95 0.95 0.95 0.95 0.99 0.99 0.99 0.99]);
summary(vbtweekly)

zoom   = (test_time >= datetime(2000,1,1)) & (test_time <= datetime(2000,12,1));
VaRData   = [-VaR_garch95(zoom) -historical95(zoom) -EWMA95(zoom)];
VaRFormat = {'-','-','-'};
D = test_time(zoom);
R = test_returns(zoom);
G = VaR_garch95(zoom);
H = historical95(zoom);
E = EWMA95(zoom);
IndG95    = (R < -G);
IndHS95   = (R < -H);
IndEWMA95 = (R < -E);
figure;
bar(D,R,0.5,'FaceColor',[0.7 0.7 0.7]);
hold on
for i = 1 : size(VaRData,2)
    stairs(D-0.5,VaRData(:,i),VaRFormat{i});
end
ylabel('VaR')
xlabel('Date')
legend({'Returns','GARCH','Historical','EWMA'},'Location','Best','AutoUpdate','Off')
% title('VaR(5\%) Violations for Different Models')
ax = gca;
ax.ColorOrderIndex = 1;

plot(D(IndG95),-G(IndG95),'o','MarkerSize',8, "MarkerEdgeColor",[0.8500 0.3250 0.0980],'LineWidth',1.5)
plot(D(IndHS95),-H(IndHS95),'diamond','MarkerSize',8, "MarkerEdgeColor",[0.9290 0.6940 0.1250],'LineWidth',1.5)
plot(D(IndEWMA95),-E(IndEWMA95),'^','MarkerSize',8, "MarkerEdgeColor",[0 0.4470 0.7410],'LineWidth',1.5)
xlim([D(1)-1, D(end)+1])
hold off;
saveas(gcf, "Weekly VaR.png")
