import scipy.stats as stats
from collections import defaultdict
from PIL import Image
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data_level_stats(model, data_list, resize=None):
    """
    param model: The model that is used to produce the baseline statistics
    param data_list: List of tuples, where each tuple comprise an image,
        given as ndarray of shape (H,W,C), and its class
    param device: Device on which to load data
    param resize: Size of each dimension in resized images
    """
    # Get activations and class ids for all data
    activations, clsids = [], []
    for image, label in data_list:
        with torch.no_grad():
            if resize is not None:
                # convert from ndarray to PIL, resize, and convert back to ndarray
                image = Image.fromarray(np.uint8(image * 255))
                image = image.resize(size=(224, 224), resample=Image.BILINEAR)
                image = np.array(image, dtype=np.float32)
                image = image / 255.0
            image = np.expand_dims(image, 0)
            image = torch.tensor(image).to(device)
            h, _ = model(image)  # returns (activation, output)
            h = h.detach().cpu().numpy()
            activations.append(h)
            clsids.append(label)

    activations = np.concatenate(activations)
    clsids = np.array(clsids)

    # Determine typicality of each input using its node activations
    # 1. Start by calculating the mean and standard deviation of node activations of each class
    # 2. For each input, calculate the class to which each node activation most likely belongs
    # 3. Tally the distribution of classes associated with each input's node activations

    # Calculate baseline mean and standard deviation
    mean_activations = []
    std_activations = []
    for c in sorted(set(clsids)):  # iterate over all unique classes in order
        class_activations = activations[
            clsids == c, :
        ]  # select only activations associated with class c
        mean_activations.append(np.mean(class_activations, axis=0, keepdims=True))
        std_activations.append(np.std(class_activations, axis=0, keepdims=True))
    std_activations = np.concatenate(std_activations)
    mean_activations = np.concatenate(mean_activations)

    # Calculate maximum likelihood, assuming uniform class distribution and
    # node activation with truncated normal distribution
    train_labels = []
    train_preds = []  # model predictions
    train_typicality = []  # typicality score of an input wrt its labeled class
    typicality_dist_dict = defaultdict(
        list
    )  # typicality distribution of an input wrt all classes

    for image, label in data_list:
        with torch.no_grad():
            if resize is not None:
                image = Image.fromarray(np.uint8(image * 255))
                image = image.resize(size=(224, 224), resample=Image.BILINEAR)
                image = np.array(image, dtype=np.float32)
                image = image / 255.0
            image = np.expand_dims(image, 0)
            image = torch.tensor(image).to(device)
            h, pred = model(image)
            h = h.detach().cpu().numpy()
            pred = pred.detach().cpu().numpy()
            h = np.squeeze(h)

            # Calculate node activation likelihoods wrt all classes.
            # Likelihood has shape (num_classes, num_nodes), where (i,j) represents
            # the probablity that an input from class i produced the
            # activation at neuron j
            likelihood = stats.norm.pdf(h, mean_activations, std_activations) / (
                1 - stats.norm.cdf(0, mean_activations, std_activations)
            )

            # Max_likelihood has shape (num_nodes,) where element i is the
            # most likely class that produced the activation at node i
            max_likelihood = np.argmax(likelihood, axis=0)

            # Typicality for the input is the percentage of node activations that are
            # most likely produced by the input class (higher the better)
            typicality = (max_likelihood == label).sum() / len(max_likelihood)

            # In addition, calculate the typicality distribution
            typicality_dist = []
            for c in sorted(set(clsids)):
                # calculate percentage of node activations strongly associate with class c
                typicality_class = (max_likelihood == c).sum() / len(max_likelihood)
                typicality_dist.append(typicality_class)
            typicality_dist_dict[label].append(typicality_dist)

            # Save statistics
            train_labels.append(label)
            train_preds.append(np.argmax(pred))
            train_typicality.append(typicality)

    train_labels = np.array(train_labels)
    train_preds = np.array(train_preds)
    train_typicality = np.array(train_typicality)

    # Convert typicality_dist_dict from a dictionary to a confusion matrix
    train_typicality_dist = np.zeros((len(set(clsids)), len(set(clsids))))
    for k, v in typicality_dist_dict.items():
        v = np.array(v)
        train_typicality_dist[k, :] = np.mean(v, axis=0)

    # Calculate typicality distributions for majority and minority subclasses
    class_typicality_match_stats = []  # list of tuples (mean, std)
    class_typicality_mismatch_stats = []
    for c in sorted(set(clsids)):
        class_preds = train_preds[train_labels == c]
        class_typicality = train_typicality[train_labels == c]
        class_typicality_match = class_typicality[class_preds == c]
        class_typicality_mismatch = class_typicality[class_preds != c]

        # Save typicality stats and account for special cases        
        if len(class_typicality_match) == 0:
            match_mean = 0
            match_std = 1e-10 # avoid dividing by zero
        elif len(class_typicality_match) == 1:
            match_mean = class_typicality_match[0]
            match_std = 1e-10
        else:
            match_mean = class_typicality_match.mean()
            match_std = class_typicality_match.std()
        
        if len(class_typicality_mismatch) == 0:
            mismatch_mean = 0
            mismatch_std = 1e-10
        elif len(class_typicality_mismatch) == 1:
            mismatch_mean = class_typicality_mismatch[0]
            mismatch_std = 1e-10
        else:
            mismatch_mean = class_typicality_mismatch.mean()
            mismatch_std = class_typicality_mismatch.std()

        class_typicality_match_stats.append((match_mean, match_std))
        class_typicality_mismatch_stats.append((mismatch_mean, mismatch_std))

    return (
        train_typicality_dist,
        class_typicality_match_stats,
        class_typicality_mismatch_stats,
        mean_activations,
        std_activations,
    )


def get_per_example_stats(
    model,
    data_list,
    mean_activations,
    std_activations,
    class_typicality_match_stats,
    class_typicality_mismatch_stats,
    resize=None,
):
    """
    param model: The model that is used to produce the baseline statistics
    param data_list: List of tuples, where each tuple comprise an image,
        given as ndarray of shape (H,W,C), and its classid
    param mean_activations: List of ndarrays, where each ndarray is the mean activations for a class
    param std_activations: List of ndarrays, where each ndarray is the std activations for a class
    param class_typicality_match_stats: List of tuples, where each tuple (mean, std) describes
        the distribution over typicality values for a class when the model prediction matches the true label
    param class_typicality_mismatch_stats: List of tuples, where each tuple (mean, std) describe
        the distribution over typicality values for a class when the model prediction does not match the true
        label
    param resize: Size of each dimension in resized images
    """
    typicality_output = []
    majority_minority_output = []

    for image, label in data_list:
        with torch.no_grad():
            if resize is not None:
                image = Image.fromarray(np.uint8(image * 255))
                image = image.resize(size=(224, 224), resample=Image.BILINEAR)
                image = np.array(image, dtype=np.float32)
                image = image / 255.0
            image = np.expand_dims(image, 0)
            image = torch.tensor(image).to(device)
            h, _ = model(image)
            h = h.detach().cpu().numpy()
            h = np.squeeze(h)

            # Calculate node activation likelihoods wrt all classes.
            # Likelihood has shape (num_classes, num_nodes), where (i,j) represents
            # the probablity that an input from class i produced the
            # activation at neuron j
            likelihood = stats.norm.pdf(h, mean_activations, std_activations) / (
                1 - stats.norm.cdf(0, mean_activations, std_activations)
            )

            # Max_likelihood has shape (num_nodes,) where element i is the
            # most likely class that produced the activation at node i
            max_likelihood = np.argmax(likelihood, axis=0)

            # Typicality for the input is the percentage of node activations that are
            # most likely produced by the input class (higher the better)
            typicality = (max_likelihood == label).sum() / len(max_likelihood)
            typicality_output.append(typicality)

            # Calculate majority/minority
            match_mean, match_std = class_typicality_match_stats[label]
            mismatch_mean, mismatch_std = class_typicality_mismatch_stats[label]
            match_likelihood = stats.norm.pdf(typicality, match_mean, match_std) / (
                1 - stats.norm.cdf(0, match_mean, match_std)
            )
            mismatch_likelihood = stats.norm.pdf(
                typicality, mismatch_mean, mismatch_std
            ) / (1 - stats.norm.cdf(0, mismatch_mean, mismatch_std))

            if match_likelihood >= mismatch_likelihood:
                majority_minority_output.append("majority")
            else:
                majority_minority_output.append("minority")

    return typicality_output, majority_minority_output


def demo():
    """
    Example code demonstrating how the above two functions can be used along with the
    resnet18_bean_regularization model. DELETE when integration is complete.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate a pretrained model
    from resnet_bean_regularization import get_model

    weights_path = "./models/BEAN-2-abs-dist-alpha-100.pt"
    model_kwargs = {
        "data_means": [0.39382024, 0.4159701, 0.40887499],
        "data_stds": [0.18931773, 0.18901625, 0.19651154],
        "num_classes": 10,
    }
    model = get_model(weights_path, **model_kwargs)
    print("Instantiated model")

    # Get all training data in the same format as Armory
    # i.e., images with channel last and normalized to [0,1]
    import numpy as np
    from PIL import Image
    import glob

    root_dir = "resisc10/train"
    image_dirs = glob.glob(root_dir + "/*")
    image_dirs.sort()

    complete_data_list = []
    for c, d in enumerate(image_dirs):
        images = glob.glob(d + "/*.jpg")
        images.sort()
        for image in images:
            im = Image.open(image)
            im = np.array(im, dtype=np.float32)
            im = im / 255.0
            complete_data_list.append((im, c))
    print("Created complete data list")

    # Create a random list of data to act as data filtered by poisoning defense
    num_filtered_data = 100
    filtered_data_idx = np.random.choice(
        len(complete_data_list), size=num_filtered_data, replace=False
    )
    filtered_data_list = [complete_data_list[idx] for idx in filtered_data_idx]
    print("Created filtered data list")

    # Get statistics of the complete dataset
    (
        train_typicality_dist,
        class_typicality_match_stats,
        class_typicality_mismatch_stats,
        mean_activations,
        std_activations,
    ) = get_data_level_stats(model, complete_data_list, device)
    print("Calculated data-level statistics")

    # Get statistics, including majority/minority, of the filtered data
    typicality_output, majority_minority_output = get_per_example_stats(
        model,
        filtered_data_list,
        mean_activations,
        std_activations,
        class_typicality_match_stats,
        class_typicality_mismatch_stats,
        device,
    )
    print("Calculated example-level statistics")
