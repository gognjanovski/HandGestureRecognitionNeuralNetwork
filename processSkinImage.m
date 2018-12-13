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

    % Get the central color of the image
    % Expected the hand to be in the central of the image
    central_color = original(int32(M/2),int32(N/2),:);
    % Convert to YCbCr
    central_color_ycbcr = rgb2ycbcr(central_color);
    Cb_Color = central_color_ycbcr(:,:,2);
    Cr_Color = central_color_ycbcr(:,:,3);
    % Set the range
    Cb_Difference = 15;
    Cr_Difference = 10;

    % Read the image, and capture the dimensions
    height = size(original,1);
    width = size(original,2);

    % Initialize the output image
    image_out = zeros(height,width);

    % Convert the image from RGB to YCbCr
    img_ycbcr = rgb2ycbcr(original);
    Cb = img_ycbcr(:,:,2);
    Cr = img_ycbcr(:,:,3);
 
    % Detect skin pixels
    [r,c,v] = find(Cb>=Cb_Color-Cr_Difference & Cb<=Cb_Color+Cb_Difference & Cr>=Cr_Color-Cr_Difference & Cr<=Cr_Color+Cr_Difference);
    match_count = size(r,1);

    % Mark detected pixels
    for i=1:match_count
        image_out(r(i),c(i)) = 1;
    end

    %imshow(image_out);

    % Resize the image to 50x50
    image_out = imresize(image_out, scale);

end