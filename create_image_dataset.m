pkg load image;

% Remove previous folders
rmdir('dataset_resized', 's');

% Create required directories
mkdir('dataset_resized');
mkdir('dataset_resized/left');
mkdir('dataset_resized/right');
mkdir('dataset_resized/palm');
mkdir('dataset_resized/peace');

label_keys = { 'left', 'right', 'palm', 'peace'};

X = [];
y = [];

% Read all the dataset images
Files=dir('dataset/*/*.jpg');
for k=1:length(Files)
    FileNames = Files(k).name;
    dr = Files(k).folder;
    fileLocation = strcat(dr, '\', FileNames);
    
    % Extract folder name
    path = strsplit(dr, '\');
    folder = path{length(path)};

    % Create the image label depending on the folder name
    % Ex. [1 0 0 0] - left folder
    label = ismember(label_keys, folder);
    y = [y; label];

    % Extract file name
    filename = strsplit(FileNames, '.');
    name = filename{1};
    extension = filename{2};

    % Process image by skin color
    % Returns 50x50 image
    image_out = processSkinImage(fileLocation);

    % Create the features for the image  
    X = [X; image_out(:)'];

    % Write the proceessed image in dataset_resized folder
    output_folder = strcat('dataset_resized', '/', folder, '/', name, '_resized.', extension);
    imwrite(image_out, output_folder);
    
end

% Write the features and labels in file
dlmwrite('x_features', X);
dlmwrite('y_labels', y);