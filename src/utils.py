import matplotlib.pyplot as plt

def load_data(data_dir):
    # Load data from directory and return as arrays or data generators
    pass

def plot_sample_images(data):
    images, labels = next(data)
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i])
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    plt.show()
