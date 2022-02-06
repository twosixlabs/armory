import scipy.stats as stats
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from collections import defaultdict
from PIL import Image as PImage
import numpy as np
import torch

def run_silhouette_analysis(X, random_seed = 32, range_n_clusters = [2,3,4,5,6]):
    
    avg_score, silhouette_coeff, cluster_groups, cluster_centers = [], [], [], []
    # run k-means and generate silhouette scores for input activations
    # returns clusters
    #print("plotting activations for input:", X.shape)
    for n_clusters in range_n_clusters:
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_seed)
        cluster_labels = clusterer.fit_predict(X)
        # save cluster centroids
        cluster_centers.append(clusterer.cluster_centers_)
        #collect cluster labels for class disribution
        cluster_groups.append(cluster_labels)
        print("output of K-means:", clusterer)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )
        avg_score.append(silhouette_avg)
        
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        silhouette_coeff.append(sample_silhouette_values)
       
    avg_score = np.array(avg_score)#, dtype=np.float32)
    silhouette_coeff = np.array(silhouette_coeff)#, dtype=np.float32)
    cluster_groups = np.array(cluster_groups)#, dtype=np.int32)
    cluster_centers = np.array(cluster_centers)#, dtype=np.float32)
    
    # print(avg_score)

    return avg_score, silhouette_coeff, cluster_groups, cluster_centers

def get_data_level_stats(model, non_filtered_list, filtered_list, device, n_clusters=None):
    """
    param model: The model that is used to produce the baseline statistics
    param non_filtered_list: List of tuples, where each tuple comprise an image,
        given as ndarray of shape (H,W,C), and its class
    pram filtered_list: List of tuples, where each tuple comprise filtered images,
        given as ndarray of shape (H,W,C), and its class
    param device: Device on which to load data
    
    ### returns sub class statistics of filtered data only
    attribute c_majority: majority cluster label
    attribute c_minority: minority cluster label
    attribute s_majority: majority/minority of each sample from silhouette analysis
    attribute c_labels: cluster label of each sample
    attribute s_values: silhouette value of each sample
    attribute c_avg: threshold value of sub groups has three ranges 
        [-1, 0] is minority: A negative values indicates potential misclassification of the data point
        [0, avg] is majority: Values close to 0 tend to overlap, this range shows typical representation
        [avg, 1] is minority: values in this range are considered outliers 
    
    """
    activations, clsids  = [], []
    for image, label in non_filtered_list:
        with torch.no_grad():
            # convert from ndarray to PIL, resize, and convert back to ndarray
            image = PImage.fromarray(np.uint8(image * 255))
            image = image.resize(size=(224, 224), resample=PImage.BILINEAR)
            image = np.array(image, dtype=np.float32)
            image = image / 255.0
            image = np.expand_dims(image, 0)
            image = torch.tensor(image).to(device)
            h, _ = model(image)  # returns (activation, output)
            h = h.detach().cpu().numpy()
            activations.append(h)
            clsids.append(label)
    
    clsids = np.array(clsids, dtype=int) # shape (num_batches * batch_size, )

    # get activations of filtered data and class labels
    f_clsids = []
    for image, label in filtered_list:
        with torch.no_grad():
            # convert from ndarray to PIL, resize, and convert back to ndarray
            image = PImage.fromarray(np.uint8(image * 255))
            image = image.resize(size=(224, 224), resample=PImage.BILINEAR)
            image = np.array(image, dtype=np.float32)
            image = image / 255.0
            image = np.expand_dims(image, 0)
            image = torch.tensor(image).to(device)
            h, _ = model(image)  # returns (activation, output)
            h = h.detach().cpu().numpy()
            activations.append(h)
            f_clsids.append(label)
    
    # shape (num_batches * batch_size, 512); each row is the activations for one input
    activations = np.concatenate(activations) 
    f_clsids = np.array(f_clsids, dtype=int)
    clsids = np.append(clsids, f_clsids, axis=0)
    
    print(activations.shape, clsids.shape)
    print(activations[clsids == 0,:])
    
    # Determine typicality starting from node activations
    # step 1: search for optimal number of clusters for a given class of activations to perform k-means
    # step 2: derive majority/minorty from: 
    # a) average separation distance of clusters
    # b) silhouette_scores ranging between [-1,1] for each sample
    # majority: cluster_scores =[0,silhouette_avg]
    # minority: cluster_scores = [-1, 0] or cluster_scores[silhouette_avg, 1]

    # default range of clusters
    range_n_clusters = [2,3,4,5,6] if n_clusters is None else [n_clusters]
    
    # compute scores, labels, majority/minority metrics of each input sample
    c_majority = np.zeros(len(set(clsids)), dtype=int)
    c_minority = np.zeros(len(set(clsids)), dtype=int)
    s_majority = np.zeros(len(filtered_list), dtype=bool)
    c_labels = np.zeros(len(filtered_list), dtype=int)
    s_values = np.zeros(len(filtered_list), dtype=float)
    c_avg = np.zeros(len(set(clsids)), dtype=float)
    
    # run analysis per class-wise activations
    for c in sorted(set(clsids)): # iterate over all unique classes in order
        class_activations = activations[
            clsids == c, :
        ]  # select only activations associated with class c 
           # Determine optimal number of clusters from k-means  
           # search on default range [2,3,4,5,6]
        (
            avg_score, 
            silhouette_coeff, 
            cluster_labels, 
            cluster_centers 
        ) = run_silhouette_analysis(class_activations, range_n_clusters=range_n_clusters)
        
        # choose n_clusters with highest avg_score
        n_cluster_ind = np.argmax(avg_score, axis=0)
        n_clusters = range_n_clusters[n_cluster_ind]
        #print("Winning #clusters:",n_clusters)
        
        #select class level labels, values, and centers of winning cluster
        class_cluster_labels = cluster_labels[n_cluster_ind] 
        class_silhouette_values = silhouette_coeff[n_cluster_ind]
        class_silhouette_avg = avg_score[n_cluster_ind]
        class_centers = cluster_centers[n_cluster_ind]
        
        #assign majority/minority subgroups to cluster results
        counts = np.bincount(class_cluster_labels)
        cluster_majority = np.argmax(counts)
        #assign a high value to not choose the same label as minority as well
        counts[cluster_majority] = 1e8 
        cluster_minority = np.argmin(counts)
        #class_majority = max(class_cluster_labels, key=class_cluster_labels.count())
        #class_minority = min(class_cluster_labels, key=class_cluster_labels.count())
        
        # if class_majority == class_minority: # incase of equal sizes, assume two subclasses
        #     class_majority = 0
        #     class_minority = 1
    
        #append only the filtered images 
        N = np.bincount(f_clsids)[c]
        c_majority[c] = cluster_majority
        c_minority[c] = cluster_minority
        # assign binary values: if score >0 and <avg, majority(True), else minority(False)
        s_majority[np.where(f_clsids==c)] =  np.where((class_silhouette_values[-N:]>0) & (class_silhouette_values[-N:]<class_silhouette_avg), 
                                                                  True, False) #True=majority
        c_labels[np.where(f_clsids==c)] = class_cluster_labels[-N:]
        s_values[np.where(f_clsids==c)] = class_silhouette_values[-N:]
        c_avg[c] = class_silhouette_avg   
        
    return (
        c_majority,
        c_minority,
        s_majority,
        c_labels,
        s_values,
        c_avg
    )


def get_majority_subclass_binary_labels(model, non_filtered_list, filtered_list, device):
    """ Return a list of arrays, one per class, each array designating 
        by 1 a majority member of that class, and by 0 a minority

        Requests 2 clusters from get_data_level_stats.
    """

    majority_labels, minority_labels, cluster_labels, _, _ = get_data_level_stats(model, non_filtered_list, filtered_list, device, n_clusters=2)

    # Note that majority label is a list of length N_classes.  
    # The majority label may not be the same in each class.
    # Similarly, cluster_labels is also a list of lists or arrays, one per class.

    binary_labels = []

    for i, majority_label in enumerate(majority_labels):
        binary_labels.append(np.array(cluster_labels[i]) == majority_label)

    return binary_labels


def demo():
    """
    Example code demonstrating how the above two functions can be used along with the
    resnet18_bean_regularization model. DELETE when integration is complete.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate a pretrained model
    from resnet_bean_regularization import get_model

    print('Running on device:', device)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    weights_path = "./models/BEAN-2-abs-dist-alpha-100.pt"
    print(weights_path)
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

    root_dir = "resisc10/test"
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
    
    #remove the elements of filtered_data_list from complete_data_list
    complete_data_list = [i for j, i in enumerate(complete_data_list) if j not in filtered_data_idx]
    
    #print(len(complete_data_list), len(filtered_data_list))
    
    # Get statistics on the training data
    (
        sample_majority,
        sample_minority,
        sample_cluster_labels,
        sample_silhouette_values,
        sample_cluster_avg
    ) = get_data_level_stats(model, complete_data_list, filtered_data_list, device)
    print("Calculated statistics of filtered data")
