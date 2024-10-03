import os

from matplotlib import pyplot as plt

save_path = "./Results/loss_curve/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
def Draw_loss_curve(Epochs, Mean_Loss,run_time):
    epochs = list(range(1, Epochs+1))
    Mean_Loss = Mean_Loss
    plt.plot(epochs, Mean_Loss, marker='o', linestyle='-', color='blue', label='Mean Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, f'{run_time}loss_curve.png'))
    plt.show()
    plt.close()
if __name__ == '__main__':
    Mean_Loss = [0.3411,0.1233,0.1081,0.0880,0.0827,1.6295, 1.5153, 1.4983, 1.4904, 1.4876]
    Draw_loss_curve(10, Mean_Loss=Mean_Loss,run_time=1)