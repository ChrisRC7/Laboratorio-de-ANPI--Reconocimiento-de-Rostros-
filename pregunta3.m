carpetas = length(dir('training')) - 2;
imagenes = length(dir('training/s1')) - 2;
% Assuming you know the number of training samples (N)
N = carpetas * imagenes;  % Assuming each individual has the same number of images

S = [];
fp = 0;

for k = 1:carpetas
    dire = strcat('training/s', num2str(k), '/');
    for j = 1:imagenes
        imgpath = strcat(dire, num2str(j), '.jpg');
        A = imread(imgpath);
        A= im2double(A);
        B = im2double(A);
        f = A(:);
        S = [S f];
    end
end

num_columnas = size(S, 2);

% STEP 2: Compute the mean face f of set S

% Calculate the mean face using (2)
mean_face = mean(S, 2);

% STEP 3: Form matrix A with the computed mean face

% Subtract the mean face from each original face
A = S - mean_face;



% STEP 4: Calculate the SVD of A
[Ur, Sr, Vr] = svdCompact(A);

% Get the rank of matrix A
r = rank(Sr);

% STEP 5: Calculate coordinate vectors for each known individual

% Initialize an array to store the coordinate vectors
coordinate_vectors = zeros(r, N);

for i = 1:N
    xi = Ur(:, 1:r)' * (S(:, i) - mean_face);
    coordinate_vectors(:, i) = xi;
end

for k = 1:carpetas
    newimg = strcat('compare/p', num2str(k), '.jpg');
    % Assuming you have a new input image 'new_face_path'
    new_face_path = newimg;
    new_face = imread(new_face_path);
    new_face = im2double(new_face);
    new_face_vector = new_face(:);

    % Compute the coordinate vector x using (13)
    x = Ur(:, 1:r)' * (new_face_vector - mean_face);

    % Compute the vector projection fp onto the face space using (16)
    fp = Ur(:, 1:r) * x;

    % Compute the distance εf to the face space using (17)
    epsilon_f = norm((new_face_vector - mean_face) - fp);

    % Compute the distance iε to each known individual using (14)
    distances_to_known_individuals = zeros(N, 1);

    for i = 1:N
        distances_to_known_individuals(i) = norm(x - coordinate_vectors(:, i));
    end

    % Identify the known individual associated with the minimum distance
    [~, min_index] = min(distances_to_known_individuals);

    % Display the new face
    subplot(1, 2, 1);
    imshow(new_face);
    title('New Face');
    % Adjust the position of the title
    title_pos = get(get(gca, 'title'), 'position');
    title_pos(2) = -0.2;  % Adjust the vertical position
    set(get(gca, 'title'), 'position', title_pos);

    % Display the identified face using the correct index
    identified_face = reshape(S(:, min_index), size(B, 1), size(B, 2));
    subplot(1, 2, 2);
    imshow(identified_face);
    title('Face Identified');
    % Adjust the position of the title
    title_pos = get(get(gca, 'title'), 'position');
    title_pos(2) = -0.2;  % Adjust the vertical position
    set(get(gca, 'title'), 'position', title_pos);

    % Display the percentage error below the images
    text(-30, 120, ['Error: ', num2str(epsilon_f, '%.2f'), '%'], 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');

    pause(1)
end