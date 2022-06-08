# from datasets.HAM import HAM_Dataset
from trainer.VGG_trainer import VGG_Trainer
from trainer.ResNet_trainer import ResNet_Trainer
from util import *
import click


# lr = 1e-3
# epochs = 100
# batch_size = 32
# weight_decay = 1e-6
@click.command()
@click.argument('network', type=click.Choice(['VGG', 'Resnet']))
@click.option('--lr', type=float, default=1e-4)
@click.option('--epochs', type=int, default=100)
@click.option('--batch_size', type=int, default=32)
@click.option('--weight_decay', type=float, default=1e-6)

def main(network, lr, epochs, batch_size, weight_decay):
    train_loader, test_loader = generate_loader(batch_size=batch_size, test_size=0.2)
    if network == 'VGG':
        trainer = VGG_Trainer(train_set=train_loader, test_set=test_loader, lr = lr, epochs=epochs,
                              batch_size=batch_size, weight_decay=weight_decay)
    elif network == 'Resnet':
        trainer = ResNet_Trainer(train_set=train_loader, test_set=test_loader, lr = lr, epochs=epochs,
                            batch_size=batch_size, weight_decay=weight_decay)

    trainer.train()
    trainer.test()

if __name__ == '__main__':
    main()