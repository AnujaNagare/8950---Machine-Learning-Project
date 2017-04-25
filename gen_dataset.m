% Generate a dataset with equal number of tuples for
% 0's and 1's as a taget value
%%% @author  Anuja Pradeep Nagare 
%%%          anuja.nagare@uga.edu
%%% @version 25/04/2017

clc
clear all;

% Read values from output csvfile
T = readtable('output.csv');
T.Var1=[];

set1=[];
set2=[];

%read column names of table
col_names = T.Properties.VariableNames;

% Create 2 sets of tuples for 0's and 1's as target value resp.
for i=1:size(T,1)
    if T.y(i) == 1
        set1 = [set1 ; T(i,col_names)];
    elseif T.y(i) == 0
        set2 = [set2 ; T(i,col_names)];
    end
end

% Create a final subset with equal number of tuples for 0's and 1's as target value
finalset = [set1(:,col_names) ;set2(1:size(set1,1),col_names)];

% save as a csv file
writetable(finalset, 'new_dataset.csv');
