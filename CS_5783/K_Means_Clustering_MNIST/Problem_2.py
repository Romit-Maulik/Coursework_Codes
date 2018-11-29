import numpy as np
import matplotlib.pyplot as plt
import gzip as gz

np.random.seed(10)

def load_data():
    # Testing images
    data_file = gz.open(r't10k-images-idx3-ubyte.gz', 'rb')
    images = data_file.read()
    data_file.close()
    images = bytearray(images)[16:]
    images = np.reshape(np.asarray(images), (10000, 784))

    # Testing labels
    data_file = gz.open(r't10k-labels-idx1-ubyte.gz', 'rb')
    labels = data_file.read()
    data_file.close()
    labels = bytearray(labels)[8:]
    labels = np.reshape(np.asarray(labels),(10000,1))

    #Random shuffling
    randomize = np.arange(np.shape(images)[0])
    np.random.shuffle(randomize)
    images = images[randomize,:]
    labels = labels[randomize,:]

    return images, labels

def visualize_image(image,i):
    check_row = np.reshape(image,(28,28))
    plt.figure()
    plt.imshow(check_row)
    plt.savefig(r'Figure_'+str(i)+r'.png')
    plt.show()

def k_means_init(images,num_clusters):

    #First sample chosen at random
    sample = np.random.randint(low=0,high=np.shape(images)[0])
    cluster_vecs = np.reshape(images[sample,:],newshape=(1,784))

    #Find distances
    distances = np.zeros(shape=(10000),dtype='double')
    distances[:] = 1.0e10

    for i in range(num_clusters-1):
        #Calculate distance of all data with newest cluster so far
        distances_temp = np.sum((images[:,:] - cluster_vecs[i,:])**2,axis=1)
        distances = np.minimum(distances,distances_temp)

        #Calculate probabilities
        distances_for_choice = distances/np.sum(distances)

        #Randomly choose new center according to larger distances (i.e. probabilities)
        idx = np.arange(start=0,stop=10000,dtype='int')
        new_sample = np.random.choice(idx,1,p=distances_for_choice)
        new_center = np.reshape(images[new_sample,:],newshape=(1,784))

        #Add to current list of centers
        cluster_vecs = np.concatenate((cluster_vecs,new_center),axis=0)

    return cluster_vecs

def cheat_init(images,labels,num_clusters):
    #Each cluster should be unique

    #First sample chosen at random
    sample = np.random.randint(low=0, high=np.shape(images)[0])
    cluster_vecs = np.reshape(images[sample, :], newshape=(1, 784))
    cluster_labels = labels[sample,:]

    #Randomly choose unique cluster centers
    cluster_num = 1
    while cluster_num != num_clusters:
        sample = np.random.randint(low=0, high=np.shape(images)[0])
        sample_label = labels[sample,:]

        if sample_label != cluster_labels.any():
            sample = np.reshape(images[sample, :], newshape=(1, 784))
            cluster_vecs = np.concatenate((cluster_vecs,sample),axis=0)
            cluster_num = cluster_num + 1

    return cluster_vecs


def k_means_clustering(images,labels,num_clusters):

	#Initializing vectors randomly
    # cluster_vecs = np.random.randint(low=0,high=255,size=(10, 784))

    #Initializing vectors randomly from dataset
    # sample = np.random.randint(low=0, high=np.shape(images)[0],size=10)
    # cluster_vecs = np.reshape(images[sample, :], newshape=(10, 784))

    #Initializing vectors that will be unique centers
    # cluster_vecs = cheat_init(images,labels,num_clusters)

    #Using Kmeans++ for choosing the cluster centers
    cluster_vecs = k_means_init(images,num_clusters)


    cluster_vecs = cluster_vecs.astype(int)
    #Initializing cluster distance array
    cluster_distance = np.zeros(shape=(10000,num_clusters),dtype='int')
    indices = np.arange(0,10000,dtype='int')

    #K-Means clustering iterations
    for iteration in range(30):
        for cluster in range(num_clusters):
            center_vector = cluster_vecs[cluster,:]
            cluster_distance[indices,cluster] = np.sum((images[indices,:]-center_vector[:])**2,axis=1)

        cluster_labels = np.argmin(cluster_distance,axis=1)

        for cluster in range(num_clusters):
            mask = cluster_labels == cluster
            if (np.shape(cluster_labels[mask])[0]>0):
                cluster_vecs[cluster,:] = np.sum(images[mask,:],axis=0)//np.shape(cluster_labels[mask])[0]


        of = 0.0
        for cluster in range(num_clusters):
            mask = cluster_labels == cluster
            of = of + np.sum(cluster_vecs[cluster,:]-images[mask,:]**2)

        print('Objective function value:', of)

    mask = cluster_labels == 0
    sub_images = images[mask,:]
    visualize_image(sub_images[0,:],0)

    mask = cluster_labels == 1
    sub_images = images[mask,:]
    visualize_image(sub_images[0,:],1)

    mask = cluster_labels == 2
    sub_images = images[mask,:]
    visualize_image(sub_images[0,:],2)

    exit()

    # Visualize centers
    for i in range(num_clusters):
        cluster = cluster_vecs[i,:]
        visualize_image(cluster,i)


images, labels = load_data()
k_means_clustering(images,labels,3)




