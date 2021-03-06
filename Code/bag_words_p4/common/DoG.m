function DoG(image_file_names,output_file_names)

  %% Simple interest operator that:
  %%    1. Runs difference of Gaussian operator on image using VL FEAT package.

  %% Inputs: 
  %%      1. image_file_names - cell array of filenames of all images to be processed
  %%      2. output_file_names - cell array of output filenames

  %% Outputs:
  %%      None - it saves the results for each image to the files
  %%      specified in output_file_names.
  %%      Each file holds 4 variables:
  %%          x - x coordinates of points (1 x NumPoints)
  %%          y - y coordinates of points (1 x NumPoints)
  %%          scale - characteristic scale of points (radius, in pixels)  (1 x NumPoints)
  %%          angle - dominant angle of the region around each keypoint  (1 x NumPoints)
  %%          score - Always 1 (1 x NumPoints).
  %%          descriptor - SIFT descriptors of all keypoints (128 x NumPoints)

    
%%% Get total number of images
nImages = length(image_file_names);

%%% Loop over all images
for I = 1:nImages


  % read in ith image
   temp = image_file_names{I};
    ima = imread(temp);
     
  % Convert to single, grayscale image
    if size(ima,3)==3
            ima = rgb2gray(ima);
    end
    %ima = rgb2gray(ima);
    ima = single(ima);
  % Find Interest points using vl_sift. Store features in f and descriptors in d.

  [f,d] = vl_sift(ima);

  % Total number of features from image
  nFeats = size(f,2);
  
  
  
  % From f and d extract x, y, scale, angle and descriptor.

    score = ones(1, nFeats);
    x = f(1,:);
    y = f(2,:);
    scale = f(3,:);
    angle = f(4,:);
    descriptor = d;
    
  fprintf('Image: %d, Number of features detected: %d\n',I,length(x));
  % Save in output file
  save(output_file_names{I},'x','y','scale','score','angle','descriptor');
  
   %%% print out progress every 10 images    
   if (mod(I,10)==0)
      fprintf('%d.',I);
   end
end
