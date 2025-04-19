import os
import torch
import numpy as np
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from data_loader_medmnist_insight import get_loader
from medmnist import INFO, Evaluator
import wandb
import torch.nn.functional as F

from attentionmixer import VisionTransformer # original vit from TIMM components


wandb.require("core")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class Solver(object):
    def __init__(self, args):
        self.args = args
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        # Initialize W&B
        wandb.init(project="4 heads 196 patch 196 emb dim", #PnemoniaMnist
                   name=f"(breastmnist) AttnMixer", config={ #"pneumonia-vit_tiny-16-imsize-224-dim-swap"
            "learning_rate": self.args.lr,
            "epochs": self.args.epochs,
            "batch_size": self.args.batch_size,
            "dataset": "(BreastMnist)",
            "model_architecture": "(breastmnist) AttnMixer",
        })
############################################## READ DATA
        # Get data loaders for medmnist dataset
        self.train_loader, self.val_loader, self.test_loader, self.task, self.n_cl = get_loader(args)
        data_flag = args.dataset  # e.g., 'pathmnist'
        self.class_labels = INFO[data_flag]['label']
        print("#################################")
        print(self.class_labels)
        print("##################################")

 ############################################## CREATE MODEL
        self.model = VisionTransformer(
        img_size=224, patch_size=16, in_chans=3, num_classes=self.n_cl,
        embed_dim=196, depth=12, num_heads=4, mlp_ratio=4., qkv_bias=True,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0, norm_layer=nn.LayerNorm # default 0., 0., 0.1
    )
        print(self.model)

        # Move model to GPU if available
        if self.args.is_cuda:

            
            # Move the model to GPU 1
            self.model = self.model.to(device)

        # Display total parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params}")

        # Option to load pretrained model
        if self.args.load_model:
            print("Using pretrained model")
            self.model.load_state_dict(torch.load(os.path.join(self.args.model_path, 'xxxx')))

        # Define loss function
        if self.task == "multi-label, binary-class":
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        # Arrays to record training progression
        self.train_losses     = []
        self.test_losses      = []
        self.train_accuracies = []
        self.test_accuracies  = []

    def test_dataset(self, loader):
        # Set model to evaluation mode
        self.model.eval()

        all_labels = []
        all_logits = []
        all_pred = []

        for (x, y) in loader:
            if self.args.is_cuda:
                #x = x.cuda()
                x= x.to(device)

            with torch.no_grad():
                #logits, embeddings = self.model(x, return_embeddings=True)
                logits = self.model(x)

            if self.task == 'multi-label, binary-class':
                y = y.to(torch.float32)
                pred = (logits > 0.5).float()
            else:
                y = y.squeeze().long()
                pred = torch.argmax(logits, dim=1).float()

            all_labels.append(y)
            all_logits.append(logits.cpu())
            all_pred.append(pred.cpu())

        all_labels = torch.cat(all_labels).to(device)
        all_logits = torch.cat(all_logits).to(device)
        all_pred = torch.cat(all_pred).to(device)

        all_labels_np = all_labels.cpu().numpy()
        all_pred_np = all_pred.cpu().numpy()

        cm = confusion_matrix(y_true=all_labels_np, y_pred=all_pred_np, labels=list(range(self.n_cl)))

        loss = self.loss_fn(all_logits, all_labels).item()
        acc = accuracy_score(y_true=all_labels.cpu(), y_pred=all_pred.cpu())

        return acc, cm, loss, all_labels_np, all_pred_np

    def test(self, train=True):
        if train:
            acc, cm, loss, _, _ = self.test_dataset(self.train_loader)
            print(f"Train acc: {acc:.2%}\tTrain loss: {loss:.4f}\nTrain Confusion Matrix:")

        acc, cm, loss, all_labels_np, all_pred_np = self.test_dataset(self.val_loader) # originally test_loader
        print(f"Test acc: {acc:.2%}\t Test loss: {loss:.4f}\n Test Confusion Matrix:")

        
        # Log test metrics to W&B
        class_names = [self.class_labels[str(i)] for i in range(self.n_cl)]
        wandb.log({
            "test_accuracy": acc,
            "test_loss": loss,
            "test_confusion_matrix": wandb.plot.confusion_matrix(
                y_true=all_labels_np,
                preds=all_pred_np,
                class_names=class_names)
        })
                    # === Restore the best model ===


        return acc, loss


    def train(self):
        iters_per_epoch = len(self.train_loader)
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
       
        best_acc = 0

        for epoch in range(self.args.epochs):
            self.model.train()

            train_epoch_loss = []
            train_epoch_accuracy = []

            for i, (x, y) in enumerate(self.train_loader):
                if self.args.is_cuda:
                    #x, y = x.cuda(), y.cuda()
                    x, y = x.to(device), y.to(device)

                logits = self.model(x)

                if self.task == 'multi-label, binary-class':
                    y = y.to(torch.float32)
                    loss = self.loss_fn(logits, y)
                else:
                    y = y.squeeze().long()
                    loss = self.loss_fn(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_pred = logits.max(1)[1]
                batch_accuracy = (y == batch_pred).float().mean()
                train_epoch_loss.append(loss.item())
                train_epoch_accuracy.append(batch_accuracy.item())

                if i % 50 == 0 or i == (iters_per_epoch - 1):
                    print(f'Ep: {epoch+1}/{self.args.epochs}\tIt: {i+1}/{iters_per_epoch}\tbatch_loss: {loss:.4f}\tbatch_accuracy: {batch_accuracy:.2%}')

            test_acc, test_loss = self.test(train=((epoch + 1) % 25 == 0))
            old_best_acc = best_acc
            best_acc = max(test_acc, best_acc)
            print(f"Best test acc: {best_acc:.2%}\n")

            # Log training metrics to W&B
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": sum(train_epoch_loss) / iters_per_epoch,
                "train_accuracy": sum(train_epoch_accuracy) / iters_per_epoch,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "best_test_accuracy": best_acc
            })

            #torch.save(self.model.state_dict(), os.path.join(self.args.model_path, "pneumonia-vit_custom-16-imsize-224-dim-swap.pt"))
            torch.save(self.model.state_dict(), os.path.join(self.args.model_path, "breast_attnmixr.pt"))

                        #saving best model
            if  best_acc > old_best_acc:
                best_acc = test_acc
                torch.save(self.model.state_dict(), os.path.join(self.args.model_path, "best_breast_attnmixer.pt"))
                print(f"✅ Saved new best model at epoch {epoch+1} with test acc: {test_acc:.2%}")


            self.train_losses.append(sum(train_epoch_loss) / iters_per_epoch)
            self.test_losses.append(test_loss)
            self.train_accuracies.append(sum(train_epoch_accuracy) / iters_per_epoch)
            self.test_accuracies.append(test_acc)

        best_model_path = os.path.join(self.args.model_path, "best_breast_attnmixer.pt")
        print(f"\n🔁 Loading best model from: {best_model_path}")
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()

            # === Final test on test set ===
        print("\n🏁 Final evaluation on test set using the best model:")
        final_test_acc, final_test_loss = self.test(train=False)
        print(f"✅ Final Test Accuracy: {final_test_acc:.2%}, Loss: {final_test_loss:.4f}")



    def validate(self, log_images=False, batch_idx=0):
        # Set the model to evaluation mode
        self.model.eval()

        all_labels = []
        all_logits = []
        all_pred = []

        val_loss = 0.0
        correct = 0

        # Create a W&B Table with appropriate columns
        table = wandb.Table(columns=["image", "pred", "target"] + [f"score_{i}" for i in range(self.n_cl)])

        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader): # originally val_loader
                if self.args.is_cuda:
                    # x = x.cuda()
                    # y = y.cuda()
                    x = x.to(device)
                    y = y.to(device)

                #logits, embeddings = self.model(x, return_embeddings=True)
                logits = self.model(x)
                
                # Ensure target tensor is 1D for cross entropy loss
                if self.task == 'multi-label, binary-class':
                    y = y.to(torch.float32)
                    pred = (logits > 0.5).float()  # Apply threshold for predictions
                    loss = self.loss_fn(logits, y)
                else:
                    y = y.squeeze().long()  # Ensure y is 1D
                    pred = torch.argmax(logits, dim=1)
                    loss = self.loss_fn(logits, y)
                
                val_loss += loss.item() * y.size(0)
                correct += (pred == y).sum().item()

                # Aggregate for final accuracy computation
                all_labels.append(y.cpu())
                all_logits.append(logits.cpu())
                all_pred.append(pred.cpu())

                # Log one batch of images, predictions, and labels to W&B table
                if log_images and i == batch_idx:
                    for img, pred, targ, prob in zip(x.cpu(), pred.cpu(), y.cpu(), F.softmax(logits, dim=1).cpu()):
                        pred_label = self.class_labels.get(str(pred.item()), 'Unknown')
                        targ_label = self.class_labels.get(str(targ.item()), 'Unknown')
                        
                        # Diagnostic print statements
                        print(f"Predicted: {pred.item()}, Target: {targ.item()}")
                        print(f"Pred Label: {pred_label}, Target Label: {targ_label}")
                        
                        table.add_data(wandb.Image(img), pred_label, targ_label, *prob.numpy())

        # Calculate validation metrics
        all_labels = torch.cat(all_labels).numpy()
        all_pred = torch.cat(all_pred).numpy()

        val_loss /= len(self.test_loader.dataset) # originally val_loader
        accuracy = correct / len(self.test_loader.dataset) # originally val_loader


        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_pred, labels=list(range(self.n_cl)))

        # Log validation metrics and table to W&B
        #class_names = [self.class_labels.get(i, 'Unknown') for i in range(self.n_cl)]
        class_names = [self.class_labels[str(i)] for i in range(self.n_cl)]
        wandb.log({
            "val_accuracy": accuracy,
            "val_loss": val_loss,
            "predictions_table": table,
            "val_confusion_matrix": wandb.plot.confusion_matrix(
                y_true=all_labels,
                preds=all_pred,
                class_names=class_names
            )
        })

        print(f"Val acc: {accuracy:.2%}\tVal loss: {val_loss:.4f}\n")
        print("Val Confusion Matrix:")
        print(cm)
        wandb.finish()


    def plot_graphs(self):
        plt.plot(self.train_losses, color='b', label='Train')
        plt.plot(self.test_losses, color='r', label='Test')

        plt.ylabel('Loss', fontsize=18)
        plt.yticks(fontsize=16)
        plt.xlabel('Epoch', fontsize=18)
        plt.xticks(fontsize=16)
        plt.legend(fontsize=15, frameon=False)

        plt.savefig(os.path.join(self.args.output_path, 'graph_loss.png'), bbox_inches='tight')
        plt.close('all')

        plt.plot(self.train_accuracies, color='b', label='Train')
        plt.plot(self.test_accuracies, color='r', label='Test')

        plt.ylabel('Accuracy', fontsize=18)
        plt.yticks(fontsize=16)
        plt.xlabel('Epoch', fontsize=18)
        plt.xticks(fontsize=16)
        plt.legend(fontsize=15, frameon=False)

        plt.savefig(os.path.join(self.args.output_path, 'graph_accuracy.png'), bbox_inches='tight')


        plt.close('all')

        # Log plots to W&B
        wandb.log({"Loss Graph": wandb.Image(os.path.join(self.args.output_path, 'graph_loss.png'))})
        wandb.log({"Accuracy Graph": wandb.Image(os.path.join(self.args.output_path, 'graph_accuracy.png'))})

