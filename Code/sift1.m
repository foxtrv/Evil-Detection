function [RETURNVALUE] = sift1(SAMPLEFACE, USERFACE)
%% Extract features on 1st image

I1 = imread(SAMPLEFACE);
I1 = imresize(I1, [112 92]);
[rows columns numberOfColorChannels] = size(I1);
if numberOfColorChannels > 1
    I1 = single(rgb2gray(I1));
else % it's already gray
    I1 = single(I1);
end

subplot(1,3,2), imshow(I1,[]);

% These parameters limit the number of features detected 
peak_thresh = 5; % increase to limit; default is 0 
edge_thresh = 5; % decrease to limit; default is 10 

[f1,d1] = vl_sift(I1, ...
    'PeakThresh', peak_thresh, ...
    'edgethresh', edge_thresh );
% fprintf('Number of frames (features) detected: %d\n', size(f1,2));
% Show all SIFT features detected
h = vl_plotframe(f1) ; set(h,'color','y','linewidth',2) ;


%% Extract features on 2nd image
I2 = imread(USERFACE);
I2 = imresize(I2, [112 92]);
[rows columns numberOfColorChannels] = size(I2);
if numberOfColorChannels > 1
    I2 = single(rgb2gray(I2));
else % it's already gray
    I2 = single(I2);
end


subplot(1,3,3), imshow(I2,[]);

% These parameters limit the number of features detected 
[f2,d2] = vl_sift(I2, ...
    'PeakThresh', peak_thresh, ...
    'edgethresh', edge_thresh );
% fprintf('Number of frames (features) detected: %d\n', size(f2,2));
% Show all SIFT features detected
h = vl_plotframe(f2) ; set(h,'color','y','linewidth',2) ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Threshold for matching
% Descriptor D1 is matched to a descriptor D2 only if the distance d(D1,D2) multiplied by THRESH is not greater than the distance of D1 to all other % descriptors
% thresh = 2.0; % default = 1.5; increase to limit matches
%thresh = 4.0;
thresh = 1.5;
[matches, scores] = vl_ubcmatch(d1, d2, thresh);

RETURNVALUE = size(matches,2);
% fprintf('Number of matching frames (features): %d\n', size(matches,2));

% Get matching features
indices1 = matches(1,:); 
f1match = f1(:,indices1); 
d1match = d1(:,indices1);

indices2 = matches(2,:);
f2match = f2(:,indices2);
d2match = d2(:,indices2);


%% % Show matches
subplot(1,3,1), imshow([I1,I2], []);
% figure, imshow([I1,I2],[]);

o = size(I1,2) ;
line([f1match(1,:);f2match(1,:)+o], ... 
    [f1match(2,:);f2match(2,:)]) ;
for i=1:size(f1match,2)
    x = f1match(1,i);
    y = f1match(2,i); text(x,y,sprintf('%d',i), 'Color', 'r');
end

for i=1:size(f2match,2)
    x = f2match(1,i);
    y = f2match(2,i); text(x+o,y,sprintf('%d',i), 'Color', 'r');
end

%% Between all pairs of matching features, compute
% orientation difference, scale ratio, and center offset 
allScales = zeros(1,size(matches,2)); % Store computed values 
allAngs = zeros(1,size(matches,2));
allX = zeros(1,size(matches,2)); 
allY = zeros(1,size(matches,2));
for i=1:size(matches, 2)
    scaleRatio = f1match(3,i)/f2match(3,i); 
    dTheta = f1match(4,i) - f2match(4,i);
    % Force dTheta to be between -pi and +pi
    while dTheta > pi
        dTheta = dTheta - 2*pi;
    end
    while dTheta < -pi
        dTheta = dTheta + 2*pi;
    end
    
    allScales(i) = scaleRatio;
    allAngs(i) = dTheta;
    
    x1 = f1match(1,i); % the feature in image 1 
    y1 = f1match(2,i);
    x2 = f2match(1,i); % the feature in image 2 
    y2 = f2match(2,i);
    
    % The "center" of the object in image 1 is located at an offset of
    % (-x1,-y1) relative to the detected feature. We need to scale and rotate % this offset and apply it to the image 2 location.
    offset = [-x1; -y1];
    offset = offset / scaleRatio; % Scale to match image 2 scale
    offset = [cos(dTheta) +sin(dTheta); -sin(dTheta) cos(dTheta)]*offset;
    
    allX(i) = x2 + offset(1);
    allY(i) = y2 + offset(2);
end

%% % Use a coarse Hough space.
% Dimensions are [angle, scale, x, y] % Define bin centers
aBin = -pi:(pi/4):pi;
sBin = 0.5:(2):10;
xBin = 1:(size(I2,2)/5):size(I2,2);
yBin = 1:(size(I2,1)/5):size(I2,1);

H = zeros(length(aBin), length(sBin), length(xBin), length(yBin)); 
for i=1:size(matches, 2)
    a = allAngs(i);
    s = allScales(i); x = allX(i);
    y = allY(i);
    % Find bin that is closest to a,s,x,y
    [~, ia] = min(abs(a-aBin)); 
    [~, is] = min(abs(s-sBin)); 
    [~, ix] = min(abs(x-xBin)); 
    [~, iy] = min(abs(y-yBin));
    H(ia,is,ix,iy) = H(ia,is,ix,iy) + 1;    % Inc accumulator array
end

% Find all bins with 3 or more features
[ap,sp,xp,yp] = ind2sub(size(H), find(H>=3));

%% Get the features corresponding to the largest bin
nFeatures = max(H(:)); % Number of features in largest bin 
[ap,sp,xp,yp] = ind2sub(size(H), find(H == nFeatures));
indices = []; % Make a list of indices 
for i=1:size(matches, 2)
    a = allAngs(i);
    s = allScales(i); x = allX(i);
    y = allY(i);
    % Find bin that is closest to a,s,x,y
    [~, ia] = min(abs(a-aBin)); 
    [~, is] = min(abs(s-sBin)); 
    [~, ix] = min(abs(x-xBin)); 
    [~, iy] = min(abs(y-yBin));
    if ia==ap(1) && is==sp(1) && ix==xp(1) && iy==yp(1) 
        indices = [indices i];
    end
end

%% Show matches to features in largest bin as line segments
o = size(I1,2) ; 
line([f1match(1,indices);f2match(1,indices)+o], ...
    [f1match(2,indices);f2match(2,indices)]) ; 

for i=1:length(indices)
    x = f1match(1,indices(i));
    y = f1match(2,indices(i)); text(x,y,sprintf('%d',indices(i)), 'Color', 'r');
end

for i=1:length(indices)
    x = f2match(1,indices(i));
    y = f2match(2,indices(i)); text(x+o,y,sprintf('%d',indices(i)), 'Color', 'r');
end



facename = strcat('User+', SAMPLEFACE);
fname = '~/Desktop/385 Final Project/Images/Result Images';
saveas(gca, fullfile(fname, facename));

