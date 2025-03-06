from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np
import wandb

# Initialize wandb
wandb.login()
wandb.init(project="da6401_assignment_1")

# Load the dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Define class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Plot one sample image for each class
plt.figure(figsize=(10, 10))
for i in range(10):
    # Find the first image of each class
    idx = np.where(y_train == i)[0][0]
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[idx], cmap=plt.cm.binary)
    plt.xlabel(class_names[i])
    
# Log the entire figure to wandb
wandb.log({"class_samples": wandb.Image(plt)})

# You can also log individual images
for i in range(10):
    idx = np.where(y_train == i)[0][0]
    wandb.log({
        f"class_{i}_{class_names[i]}": wandb.Image(
            x_train[idx], 
            caption=class_names[i]
        )
    })

plt.show()