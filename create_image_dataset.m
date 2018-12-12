pkg load image;


rmdir('dataset_resized', 's');
mkdir('dataset_resized');

mkdir('dataset_resized/left');
mkdir('dataset_resized/right');
mkdir('dataset_resized/palm');
mkdir('dataset_resized/peace');

label_keys = { 'left', 'right', 'palm', 'peace'};

X = [];
y = [];

Files=dir('dataset/*/*.jpg');
for k=1:length(Files)
    FileNames = Files(k).name;
    dr = Files(k).folder;
    fileLocation = strcat(dr, '\', FileNames);
    
    path = strsplit(dr, '\');
    folder = path{length(path)};

    label = ismember(label_keys, folder);
    idx = find(label == 1);
    
    y = [y; label];

    filename = strsplit(FileNames, '.');
    name = filename{1};
    extension = filename{2};

    image_out = processSkinImage(fileLocation);

    X = [X; image_out(:)'];

    output_folder = strcat('dataset_resized', '/', folder, '/', name, '_resized.', extension);
    imwrite(image_out, output_folder);
    
end

dlmwrite('x_features', X);
dlmwrite('y_labels', y);