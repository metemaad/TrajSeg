import random

def randomize_array(ar):
    """
    Randomizes the contents of an array
    :param ar: Array to randomize
    :return: A array with the values scrambled
    """
    random.seed(a=1, version=2)
    random_ar = []
    while len(ar)>0:
        index = random.randint(0,len(ar)-1)
        random_ar.append(ar[index])
        del ar[index]
    return random_ar


def prepare_chunks(ar, num_chunks):
    """
    Splits the array into even chunks
    :param ar: The array to split
    :param num_chunks: The number of chunks to divide into
    :return: A list of chunks of values (chunks are lists)
    """
    output = []
    for i in range(0, num_chunks):
        output.append([])
    for i in range(0,len(ar)):
        output[i % num_chunks].append(ar[i])
    print(output)
    return output


def prepare_training_testing(ar,index):
    """
    Gets the training versus testing set designed to be used in conjunction with split_array
    :param ar: The array to get the values
    :param index: The index for testing
    :return: Tuple (large sample, small sample)
    """
    smaller_sample = ar[index]
    larger_sample = []
    for i in range(0,len(ar)):
        if i != index:
            larger_sample+=ar[i]
    return larger_sample, smaller_sample
