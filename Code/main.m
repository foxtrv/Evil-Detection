
function main(USER)

% runs Bag-Of-Words
% run('config_file_1.m');
% do_all('config_file_1');

% runs SIFT
numgood = 0;
numevil = 0;
for i=1:100
    temp = strcat('Good_', int2str(i));
    temp = strcat(temp,'.jpeg');
    numgood = numgood + sift1(temp, USER);
    temp2 = strcat('Evil_', int2str(i));
    temp2 = strcat(temp2,'.jpeg');
    numevil = numevil + sift1(temp2, USER); 
end
fprintf('Number of Features Matched in GOOD Faces With USER Face (using SIFT): %d\n', numgood);
fprintf('Number of Features Matched in EVIL Faces With USER Face (using SIFT): %d\n', numevil);
if numgood > numevil
    t = (numgood - numevil);
    if t > 100
        t = 100;
    end
    fprintf('You are %d%% Good!\n', t);
end
if numgood < numevil
    t = (numevil - numgood);
    if t > 100
        t = 100;
    end
    fprintf('You are %d%% Evil!\n', t);
end

end