clear all
close all
clc

rawData = readtable('train.csv'); %read csv
dataSet = (table2cell(rawData)); %remove table headers

pNumbers = rawData{:,'PassengerId'};
pSurvival = rawData{:,'Survived'};
pClasses = rawData{:,'Pclass'};
pSex = rawData{:,'Sex'};
pAge= rawData{:,'Age'};
pPort = rawData{:,'Embarked'};
pSibSp = rawData{:,'SibSp'};
pParch = rawData{:,'Parch'};


%treat missing ages
missingAges = sum(isnan(pAge));
disp(missingAges + " Missing values found in age.");
if (missingAges/size(dataSet,1)<0.25) %if less than 25% missing use the mean age
    pAge(isnan(pAge)) = mean(pAge,'omitnan');
    disp(missingAges + " Missing ages replaced with average."); %if more than 25% missing, ignore missing values
elseif (missingAges/size(dataSet,1)>0.25) 
     disp(missingAges + " Missing ages ignored.");
end

%treat missing ports
missingPorts = sum(isempty(pPort));
[s,~,j]=unique(pPort);
modePort = s{mode(j)};
disp(missingPorts + " Missing values found in port.");
if (missingPorts/size(dataSet,1)<0.25) %if less than 25% missing use the mean age
    pPort(isempty(pPort)) = cellstr(modePort);
    disp(missingPorts + " Missing ports replaced with mode."); %if more than 25% missing, ignore missing values
elseif (missingPorts/size(dataSet,1)>0.25) 
     disp(missingPorts + " Missing ports ignored.");
end




%categorise the age groups
ageGroups = ordinal(pAge,["'Very Young'","'Young'","'Middle Aged'","'Old'"],[],[0,15,30,65,100]);
age = table (ageGroups);
rawData(:,6) = age;




%combine into 1 table for analysis
usefulData = table(pClasses, pSex, ageGroups, pSurvival);

 %Analysis of the data
 survival = ordinal(pSurvival,{'Survived','Not survived'},[],[0,0.9,1.1]);
 sv= categorical(survival);
 sexStats = grpstats(rawData(:,{'Survived','Sex'}), 'Sex');
 portStats = grpstats(rawData(:,{'Survived','Embarked'}), 'Embarked');
 classStats = grpstats(rawData(:,{'Survived','Pclass'}), 'Pclass');
 ageStats = grpstats(usefulData(:,{'pSurvival','ageGroups'}), 'ageGroups');
 

%visualize data 
%overall surv rate
figure;
 hold on
 noSurvivors = tabulate(sv);
 t = cell2table(noSurvivors,'VariableNames',{'Value','Count','Percent'});
 t.Value = categorical(t.Value);
 b = bar(t.Value,t.Count);
 xlabel('Survival Status');
 ylabel('Number of passengers');
 barXCoords = (b(1).XEndPoints);
 barYCoords = b(1).YEndPoints;
 percentage = string(t.Percent + "%");
 text(barXCoords,barYCoords,percentage,'HorizontalAlignment','center','VerticalAlignment','bottom');
 hold off
 
 %sex surv rate
figure;
 hold on
 t = sexStats;
 t.Sex = categorical(t.Sex);
 b1 = bar(t.Sex,t.mean_Survived);
 xlabel('Sex');
 ylabel('Percentage survived');
 barXCoords = (b1(1).XEndPoints);
 barYCoords = b1(1).YEndPoints;
 percentage = string(t.mean_Survived .* 100+ "%");
 text(barXCoords,barYCoords,percentage,'HorizontalAlignment','center','VerticalAlignment','bottom');
 hold off
 
 %class surv rate
 figure;
 hold on
 t = classStats;
 t.Pclass = categorical(t.Pclass);
 b2 = bar(t.Pclass,t.mean_Survived);
 xlabel('Passenger Class');
 ylabel('Percentage survived');
 barXCoords = (b2(1).XEndPoints);
 barYCoords = b2(1).YEndPoints;
 percentage = string(t.mean_Survived .* 100+ "%");
 text(barXCoords,barYCoords,percentage,'HorizontalAlignment','center','VerticalAlignment','bottom');
 hold off

  %port surv rate
 figure;
 hold on
 t = portStats;
 t.Embarked = categorical(t.Embarked);
 b3 = bar(t.Embarked,t.mean_Survived);
 xlabel('Port Embarked');
 ylabel('Percentage survived');
 barXCoords = (b3(1).XEndPoints);
 barYCoords = b3(1).YEndPoints;
 percentage = string(t.mean_Survived .* 100+ "%");
 text(barXCoords,barYCoords,percentage,'HorizontalAlignment','center','VerticalAlignment','bottom');
 hold off
 

 %age surv rate
 figure;
 hold on
 t = ageStats;
 t.ageGroups = categorical(t.ageGroups);
 b4 = bar(t.ageGroups,t.mean_pSurvival);
 barXCoords = (b4(1).XEndPoints);
 barYCoords = b4(1).YEndPoints;
 xlabel('Age Group');
 ylabel('Percentage survived');
 percentage = string(t.mean_pSurvival .* 100+ "%");
 text(barXCoords,barYCoords,percentage,'HorizontalAlignment','center','VerticalAlignment','bottom');
 hold off

 
 
 %convert selected character data into numerical format
sortedSex = grp2idx(pSex);
ageGroups = grp2idx(ageGroups);
sortedPorts = grp2idx(pPort);
usefulData = table(pClasses, sortedSex, ageGroups, pSurvival);


%Random forest
treeBagger = TreeBagger(100,usefulData(:,1:end-1),usefulData(:,end),'OOBPredictorImportance','On');

figure;
error = oobError(treeBagger);
plot(error); hold on; 
xlabel('Tree Count');
ylabel('Error');

%Attribute importance
figure;
bar(treeBagger.OOBPermutedPredictorDeltaError);
xlabel('Attribute');
ylabel('Importance');
set(gca,'xticklabel',treeBagger.PredictorNames);

[M ,I] = min(error);
disp("Optimum tree count is: " + I);
%random forest with optimal trees
optimalTB = TreeBagger(I,usefulData(:,1:end-1),usefulData(:,end),'OOBPredictorImportance','off','OOBPrediction','on');

%calculate accuracy
Yfit = oobPredict(optimalTB);
error = confusionmat(pSurvival,str2double(Yfit));
error_analysis = trace(error)/sum(error, 'all');
disp(error_analysis*100 + "% match to ground truth table of training file.");


























%load test data

rawData = readtable('testdata_with_groundTruth.csv'); %read csv
dataSet = (table2cell(rawData)); %remove table headers

pNumbers = rawData{:,'PassengerId'};
pSurvival = rawData{:,'Survived'};
pClasses = rawData{:,'Pclass'};
pSex = rawData{:,'Sex'};
pAge= rawData{:,'Age'};
pPort = rawData{:,'Embarked'};
pSibSp = rawData{:,'SibSp'};
pParch = rawData{:,'Parch'};



%treat missing ages
missingAges = sum(isnan(pAge));
disp(missingAges + " Missing values found in age.");
if (missingAges/size(dataSet,1)<0.25) %if less than 25% missing use the mean age
    pAge(isnan(pAge)) = mean(pAge,'omitnan');
    disp(missingAges + " Missing ages replaced with average."); %if more than 25% missing, ignore missing values
elseif (missingAges/size(dataSet,1)>0.25) 
     disp(missingAges + " Missing ages ignored.");
end

%categorise the age groups
ageGroups = ordinal(pAge,["'Very Young'","'Young'","'Middle Aged'","'Old'"],[],[0,15,30,65,100]);
age = table (ageGroups);
rawData(:,6) = age;


 %convert selected character data into numerical format
sortedSex = grp2idx(pSex);
ageGroups = grp2idx(ageGroups);
sortedPorts = grp2idx(pPort);
usefulData = table(pClasses, sortedSex, ageGroups, sortedPorts,pSibSp,pParch,pSurvival);
 
%random forest
prediction = predict(optimalTB,usefulData(:,1:end-1));
error = confusionmat(pSurvival,str2double(prediction));
error_analysis = trace(error)/sum(error, 'all');
disp(error_analysis*100 + "% match to ground truth table of test file.");


