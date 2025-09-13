import os
import shutil
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def get_csv_files(data_dir):
    """ Retrieve full file paths for all CSV files in the specified directory """
    return [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]

def recreate_directory(path):
    """ Delete and recreate a directory. """
    if os.path.exists(path):
        shutil.rmtree(path)  # Remove the directory completely
    os.makedirs(path)  # Create the directory anew


def create_k_fold_directories(base_dir, num_folds):
    """ Create directories for k-fold train and test sets. """
    recreate_directory(base_dir)  # Clear the directory if it already exists
    for i in range(num_folds):
        os.makedirs(os.path.join(base_dir, f'train_fold_{i+1}'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, f'train_fold_{i+1}', '__preS'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, f'train_fold_{i+1}', '__preR'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, f'valid_fold_{i+1}'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, f'valid_fold_{i+1}', '__preS'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, f'valid_fold_{i+1}', '__preR'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, f'test_fold_{i+1}'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, f'test_fold_{i+1}', '__preS'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, f'test_fold_{i+1}', '__preR'), exist_ok=True)


def split_and_organize_by_fold(files, base_dir, type, num_folds, args):
    """ Split and organize files into k-fold directories, creating separate train, validation, and test sets within each fold with a 60:20:20 distribution. """
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=args.seed)
    fold_index = 0
    
    for train_val_index, test_index in kf.split(files):
        fold_index += 1
        num_train = int(len(train_val_index) * 0.7)
        train_index = train_val_index[:num_train]
        valid_index = train_val_index[num_train:]

        train_files = [files[i] for i in train_index]
        validation_files = [files[i] for i in valid_index]
        test_files = [files[i] for i in test_index]
        
        for file in train_files:
            shutil.copy(file, os.path.join(base_dir, f'train_fold_{fold_index}', type, os.path.basename(file)))
        for file in validation_files:
            shutil.copy(file, os.path.join(base_dir, f'valid_fold_{fold_index}', type, os.path.basename(file)))
        for file in test_files:
            shutil.copy(file, os.path.join(base_dir, f'test_fold_{fold_index}', type, os.path.basename(file)))


def consolidate_files(susceptible_test_dir, resilient_test_dir, unified_test_dir):
    """ Consolidate files from two directories into a unified directory """
    os.makedirs(unified_test_dir, exist_ok=True)  # Ensure the directory exists
    for file in os.listdir(susceptible_test_dir):
        if file.endswith('.csv'):
            shutil.copy(os.path.join(susceptible_test_dir, file), os.path.join(unified_test_dir, file))
    
    for file in os.listdir(resilient_test_dir):
        if file.endswith('.csv'):
            shutil.copy(os.path.join(resilient_test_dir, file), os.path.join(unified_test_dir, file))


def fit_scaler(source_seq):
    """ Fit a StandardScaler on the training sequences """
    all_data = torch.cat([tensor for tensor in source_seq], dim=0).numpy()
    scaler = StandardScaler()
    scaler.fit(all_data)
    return scaler


def apply_scaler(source_seq, train_scaler):
    """ Apply a fitted scaler to transform sequences """
    scaled_data = []
    for tensor in source_seq:
        scaled_tensor = train_scaler.transform(tensor.numpy())
        scaled_data.append(torch.tensor(scaled_tensor, dtype=torch.float32))
    return scaled_data


def map_filenames_to_labels(susceptible_dir, resilient_dir):
    """ Map each filename in the test directories to their respective labels """
    filename_label_map = {}
    for file in os.listdir(susceptible_dir):
        if file.endswith('.csv'):
            filename_label_map[file] = 0
    for file in os.listdir(resilient_dir):
        if file.endswith('.csv'):
            filename_label_map[file] = 1
    return filename_label_map


def setup_kfold_experiment(args, num_folds=4):
    """
    Setup k-fold cross-validation directory structure and file splitting
    
    Args:
        args: Arguments containing seed and other configuration
        num_folds: Number of folds for cross-validation
        
    Returns:
        tuple: (label_file, save_dataset_dir)
    """
    BASE_DATA_DIR = os.path.join("dataset")

    susceptible_dir = os.path.join(BASE_DATA_DIR, 'preS')
    resilient_dir = os.path.join(BASE_DATA_DIR, 'preR')
    label_file = os.path.join(BASE_DATA_DIR, "AVATAR_SDSBD.xlsx")
    save_dataset_dir = os.path.join(BASE_DATA_DIR, 'fold')

    create_k_fold_directories(save_dataset_dir, num_folds)

    # Get files and split for susceptible and resilient data
    susceptible_files = get_csv_files(susceptible_dir)
    resilient_files = get_csv_files(resilient_dir)

    # Organize susceptible and resilient files into k-folds
    split_and_organize_by_fold(susceptible_files, save_dataset_dir, f'__preS', num_folds, args)
    split_and_organize_by_fold(resilient_files, save_dataset_dir, f'__preR', num_folds, args)

    return label_file, save_dataset_dir


def setup_fold_directories(fold, save_dataset_dir):
    """
    Setup directory paths for a specific fold and create filename label mapping
    
    Args:
        fold: Fold number (1-based)
        save_dataset_dir: Base directory containing fold data
        
    Returns:
        tuple: (fold_dirs, filename_label_map) where fold_dirs is a dictionary 
               containing all directory paths and filename_label_map maps filenames to labels
    """
    train_dir = os.path.join(save_dataset_dir, f'train_fold_{fold}')
    valid_dir = os.path.join(save_dataset_dir, f'valid_fold_{fold}')
    test_dir = os.path.join(save_dataset_dir, f'test_fold_{fold}')

    susceptible_train_dir = os.path.join(train_dir, '__preS')
    resilient_train_dir = os.path.join(train_dir, '__preR')
    unified_train_dir = os.path.join(train_dir, '__train')

    susceptible_valid_dir = os.path.join(valid_dir, '__preS')
    resilient_valid_dir = os.path.join(valid_dir, '__preR')
    unified_valid_dir = os.path.join(valid_dir, '__valid')

    susceptible_test_dir = os.path.join(test_dir, '__preS')
    resilient_test_dir = os.path.join(test_dir, '__preR')
    unified_test_dir = os.path.join(test_dir, '__test')

    consolidate_files(susceptible_train_dir, resilient_train_dir, unified_train_dir)
    consolidate_files(susceptible_valid_dir, resilient_valid_dir, unified_valid_dir)
    consolidate_files(susceptible_test_dir, resilient_test_dir, unified_test_dir)

    __train_dir = os.path.join(train_dir, '__train')
    __valid_dir = os.path.join(valid_dir, '__valid')
    __test_dir = os.path.join(test_dir, '__test')

    fold_dirs = {
        'susceptible_test_dir': susceptible_test_dir,
        'resilient_test_dir': resilient_test_dir,
        'train_dir': __train_dir,
        'valid_dir': __valid_dir,
        'test_dir': __test_dir
    }

    # Create filename to label mapping for this fold
    filename_label_map = map_filenames_to_labels(susceptible_test_dir, resilient_test_dir)

    return fold_dirs, filename_label_map


def prepare_datasets_with_scaling(train_dataset, valid_dataset):
    """
    Prepare datasets with scaling applied
    
    Args:
        train_dataset: Training dataset
        valid_dataset: Validation dataset
        
    Returns:
        tuple: (train_dataset, valid_dataset, train_scaler)
    """
    train_scaler = fit_scaler(train_dataset.source_seq)
    train_dataset.source_seq = apply_scaler(train_dataset.source_seq, train_scaler)
    valid_dataset.source_seq = apply_scaler(valid_dataset.source_seq, train_scaler)
    
    return train_dataset, valid_dataset, train_scaler
