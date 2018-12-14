function image_out = processSkinImage(filename)
%PROCESSSKINIMAGE Segment hand from image
%   image_out = PROCESSSKINIMAGE(filename) 
%   Segment hand by skin color from given image

    % Load image package 
    pkg load image;

    % Initialize
    numrows = 50;
    numcols = 50;
    scale = [numrows numcols];

    % Read the image
    original = imread(filename);
    [M N Z] = size(original);

    % Read the image, and capture the dimensions
    height = size(original,1);
    width = size(original,2);
    
    % Resize the image to 50x50
    image_resized = imresize(original, scale);
    [M N Z] = size(image_resized);

    % Initialize the output image
    image_out = zeros(M,N);

    % Convert the image from RGB to YCbCr
    img_ycbcr = rgb2ycbcr(image_resized);
    Cb = img_ycbcr(:,:,2);
    Cr = img_ycbcr(:,:,3);

    % Get the central color of the image
    % Expected the hand to be in the central of the image
    central_color = img_ycbcr(int32(M/2),int32(N/2),:);
    Cb_Color = central_color(:,:,2);
    Cr_Color = central_color(:,:,3);
    % Set the range
    Cb_Difference = 15;
    Cr_Difference = 10;
 
    % Detect skin pixels
    [r,c,v] = find(Cb>=Cb_Color-Cr_Difference & Cb<=Cb_Color+Cb_Difference & Cr>=Cr_Color-Cr_Difference & Cr<=Cr_Color+Cr_Difference);
    match_count = size(r,1);

    % Mark detected pixels
    for i=1:match_count
        image_out(r(i),c(i)) = 1;
    end

    %imshow(image_out);


end