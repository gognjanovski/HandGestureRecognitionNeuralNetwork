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

X_train = [];
y_train = [];
X_test = [];
y_test = [];

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

    % Extract file name
    filename = strsplit(FileNames, '.');
    name = filename{1};
    extension = filename{2};

    % Process image by skin color
    % Returns 50x50 image
    image_out = processSkinImage(fileLocation);

    % Generate random number from 1 to 10
    randNum = ceil(rand() * 10);

    % Split the images in 80%-20% train-test set
    if randNum > 2
        % Create the features for the image  
        X_train = [X_train; image_out(:)'];
        y_train = [y_train; label];
    else 
        X_test = [X_test; image_out(:)'];
        y_test = [y_test; label];
    endif

    % Write the proceessed image in dataset_resized folder
    output_folder = strcat('dataset_resized', '/', folder, '/', name, '_resized.', extension);
    imwrite(image_out, output_folder);
    
end

% Write the train features and labels in file
dlmwrite('x_features_train', X_train);
dlmwrite('y_labels_train', y_train);


% Write the test features and labels in file
dlmwrite('x_features_test', X_test);
dlmwrite('y_labels_test', y_test);