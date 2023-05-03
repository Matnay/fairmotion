# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

from fairmotion.tasks.motion_prediction import generate, utils, test
from fairmotion.utils import utils as fairmotion_utils


logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def set_seeds():
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def critic_loss(discriminator,inputs,real,fake,lambda_=10,alpha=0.01):
    '''
     
    ONLY FOR TCN GAN 


    Gradient Penalty Loss function for discriminator for stable training of GAN
    alpha,lambda_: relative importance of each of the loss contribution (lambda_: GP loss & alpha: L2 Loss)

    Returns: Gradient Penalty Loss + L2 regularization loss for discriminator

    Inspired from WGAN-GP : https://arxiv.org/abs/1704.00028
    '''
    epsilon = torch.rand(1)
    x_cap = epsilon*torch.cat((inputs,real),1) + (1-epsilon)*torch.cat((inputs,fake),1)
    out = discriminator(x_cap)
    gradients = torch.autograd.grad(outputs=out,inputs=x_cap,\
                                    grad_outputs=torch.ones_like(out), create_graph=True,retain_graph=True)[0]
    gp_loss = torch.mean((torch.norm(gradients,dim=(-1,-2)) - 1)**2)

    norm = 0
    for name,params in discriminator.named_parameters():
        if name.endswith('weight'):
            norm += torch.sum(params)**2
    norm = torch.sqrt(norm)
    return lambda_*gp_loss + alpha*norm


def train(args):
    if args.wandb:
        ## Wandb setup for logging purposes
        wandb.login()
        # Wandb initialize
        run = wandb.init(project=None, ## Name of the project 
                config = {"representation":"rotmat",
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "optimizer":args.optimizer,
                "batch size":args.batch_size,
                "scheduler":"Pytorch Default Params",
                "Number of Residual Blocks":args.num_layers,
                "hidden dimensions":args.hidden_dim,
                "device":args.device,
                "save model frequency": args.save_model_frequency,
                "attention":args.attention},
                name=None # Name of the session
                )
    fairmotion_utils.create_dir_if_absent(args.save_model_path)
    logging.info(args._get_kwargs())
    utils.log_config(args.save_model_path, args)

    set_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device if args.device else device
    logging.info(f"Using device: {device}")

    logging.info("Preparing dataset...")
    dataset, mean, std = utils.prepare_dataset(
        *[
            os.path.join(args.preprocessed_path, f"{split}.pkl")
            for split in ["train", "test", "validation"]
        ],
        batch_size=args.batch_size,
        device=device,
        shuffle=args.shuffle,
    )
    # Loss per epoch is the average loss per sequence
    num_training_sequences = len(dataset["train"]) * args.batch_size

    # number of predictions per time step = num_joints * angle representation
    # shape is (batch_size, seq_len, num_predictions)
    _, tgt_len, num_predictions = next(iter(dataset["train"]))[1].shape


########################################################################
#             Model Loader #
    if args.architecture == "tcn_gan":
        (gen_model,dis_model) = utils.prepare_model(
            input_dim=num_predictions,
            hidden_dim=args.hidden_dim,
            device=device,
            num_layers=args.num_layers,
            architecture=args.architecture,
            attention = args.attention,
            att_heads = args.attention_heads
        )
    model = utils.prepare_model(
        input_dim=num_predictions,
        hidden_dim=args.hidden_dim,
        device=device,
        num_layers=args.num_layers,
        architecture=args.architecture,
    )
########################################################################
    criterion = nn.MSELoss()
    torch.autograd.set_detect_anomaly(True)
########################################################################
# Model Evaluation before training

    if args.architecture == "tcn_gan":
        gen_model.init_weights()
        dis_model.init_weights()
    
        G_training_losses, G_val_losses, D_training_losses = [], [], []
        epoch_loss = 0
        for iterations, (src_seqs, tgt_seqs) in enumerate(dataset["train"]):
            gen_model.eval()
            src_seqs, tgt_seqs = src_seqs.to(device), tgt_seqs.to(device)
            outputs = gen_model(src_seqs)
            loss = criterion(outputs, tgt_seqs)
            epoch_loss += loss.item()
        epoch_loss = epoch_loss / num_training_sequences
        val_loss = generate.eval(
            gen_model, criterion, dataset["validation"], args.batch_size, device,type="tcn"
        )

    if args.architecture == "tcn":
        model.init_weights()
        training_losses, val_losses = [], []

        epoch_loss = 0
        for iterations, (src_seqs, tgt_seqs) in enumerate(dataset["train"]):
            model.eval()
            src_seqs, tgt_seqs = src_seqs.to(device), tgt_seqs.to(device)
            outputs = model(src_seqs)
            loss = criterion(outputs, tgt_seqs)
            epoch_loss += loss.item()
        epoch_loss = epoch_loss / num_training_sequences
        val_loss = generate.eval(
            model, criterion, dataset["validation"], args.batch_size, device,type="tcn"
        )

    else:
        model.init_weights()
        training_losses, val_losses = [], []
        epoch_loss = 0
        for iterations, (src_seqs, tgt_seqs) in enumerate(dataset["train"]):
            model.eval()
            src_seqs, tgt_seqs = src_seqs.to(device), tgt_seqs.to(device)
            outputs = model(src_seqs, tgt_seqs, teacher_forcing_ratio=1,)
            loss = criterion(outputs, tgt_seqs)
            epoch_loss += loss.item()
        epoch_loss = epoch_loss / num_training_sequences
        val_loss = generate.eval(
            model, criterion, dataset["validation"], args.batch_size, device,
        )
    logging.info(
        "Before training: "
        f"Training loss {epoch_loss} | "
        f"Validation loss {val_loss}"
    )
#######################################################################################################################################
## Loss function for TCN GAN 

    if args.architecture == "tcn":
        gen_opt = utils.prepare_optimizer(gen_model, args.optimizer, args.lr)
        dis_opt = utils.prepare_optimizer(dis_model, args.optimizer, args.lr)

        logging.info("Training a TCN GAN for Time Series / Motion Prediction...")

        for epoch in range(args.epochs):
            g_epoch_loss,d_epoch_loss = 0,0
   
            logging.info(
                f"Running epoch {epoch} | "
                f"GANS training"
            )

            for iterations, (src_seqs, tgt_seqs) in enumerate(dataset["train"]):
                src_seqs, tgt_seqs = src_seqs.to(device), tgt_seqs.to(device)
                loader = iter(dataset["train"])
                if args.n_crit_iterations < args.batch_size:
                    raise "Reduce the number of n_crit_iterations"
                d_loss_total = 0
                for _ in range(args.n_crit_iterations):
                    ## Train the Discriminator
                    dis_model.zero_grad()
                    inputs,real = next(loader)
                    inputs,real = inputs.to(device),real.to(device)
                    latent_z    = torch.randn((inputs.shape[0],216,128)).double()
                    fake        = gen_model(inputs.transpose(2,1),latent_z)
                    
                    d_loss_real = dis_model(torch.cat((inputs,real),1)).mean()             
                    d_loss_fake = dis_model(torch.cat((inputs,fake.detach()),1)).mean()
                    d_loss_gp   = critic_loss(dis_model,inputs,real,fake)
                    d_loss      = d_loss_fake-d_loss_real+d_loss_gp

                    d_loss_total += d_loss
                    d_loss.backward()
                    dis_opt.step()
                    
                d_epoch_loss += d_loss_total/args.n_crit_iterations

                ## Train the Generator
                gen_model.zero_grad()
                x_real,_      = src_seqs, tgt_seqs
                latent_z      = torch.randn((x_real.shape[0],216,128)).double()
                y_fake        = gen_model(x_real.transpose(2,1),latent_z)
                g_loss_gan    = -dis_model(torch.cat((x_real,y_fake),1)).mean()
                
                    
                if iterations==0:
                    g_cons = (torch.norm(y_fake,dim=(-1,-2)).sum())**0.5
                    g_prev = y_fake
                else:g_cons = (torch.norm(g_prev-y_fake,dim=(-1,-2)).sum())**0.5 
                    
                g_loss_total =  g_loss_gan + g_cons
                g_epoch_loss += g_loss_total

                g_loss_total.backward()
                g_prev = y_fake.detach()
                gen_opt.step()

            G_training_losses.append(g_epoch_loss/args.batch_size)
            D_training_losses.append(d_epoch_loss/args.batch_size)
            logging.info(f'Generator Loss:{g_epoch_loss/args.batch_size}     | Epoch: {iterations+1}')
            logging.info(f'Discriminator Loss:{d_epoch_loss/args.batch_size} | Epoch: {iterations+1}')

               
            val_loss = generate.eval(
                gen_model, criterion, dataset["validation"], args.batch_size, device,type="tcn_gan"
            )
            G_val_losses.append(val_loss)
            gen_opt.epoch_step(val_loss=val_loss)
            logging.info(
                f"Generator Val loss: {val_loss}"
            )
            
            if epoch % args.save_model_frequency == 0:
                _, rep = os.path.split(args.preprocessed_path.strip("/"))
                _, mae = test.test_model(
                    model=gen_model,
                    dataset=dataset["validation"],
                    rep=rep,
                    device=device,
                    mean=mean,
                    std=std,
                    max_len=tgt_len,
                )
                logging.info(f"Validation MAE: {mae}")
                torch.save(
                    model.state_dict(), f"{args.save_model_path}/{epoch}.model"
                )
                if len(G_val_losses) == 0 or val_loss <= min(G_val_losses):
                    torch.save(
                        model.state_dict(), f"{args.save_model_path}/best.model")
            #gen_opt.scheduler.step(val_loss)
            wandb.log({"Mean Angle Error": mae, "Generator training loss": g_epoch_loss, "Discriminator training loss":d_epoch_loss,\
                       "Generator Validation loss":val_loss})

        run.finish()
        return G_training_losses, D_training_losses
#######################################################################################################################################
    
#######################################################################################################################################
##  Loss function for TCN 
    if args.architecture == "tcn":
        opt = utils.prepare_optimizer(model, args.optimizer, args.lr)
        logging.info("Training a Temporal Convolutional Network for Time Series / Motion Prediction...")

        for epoch in range(args.epochs):
            epoch_loss = 0
            model.train()

            logging.info(
                f"Running epoch {epoch} | "
                f"Dilation Rate is set as 2"
            )
            for iterations, (src_seqs, tgt_seqs) in enumerate(dataset["train"]):
                opt.optimizer.zero_grad()
                src_seqs, tgt_seqs = src_seqs.to(device), tgt_seqs.to(device)
                outputs = model(src_seqs)
                outputs = outputs.double()
                loss = criterion(
                    outputs,
                    utils.prepare_tgt_seqs(args.architecture, src_seqs, tgt_seqs),
                )
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            epoch_loss = epoch_loss / num_training_sequences
            training_losses.append(epoch_loss)
            val_loss = generate.eval(
                model, criterion, dataset["validation"], args.batch_size, device,type="tcn"
            )
            val_losses.append(val_loss)
            opt.epoch_step(val_loss=val_loss)
            logging.info(
                f"Training loss {epoch_loss} | "
                f"Validation loss {val_loss} | "
                f"Iterations {iterations + 1}"
            )
            
            if epoch % args.save_model_frequency == 0:
                _, rep = os.path.split(args.preprocessed_path.strip("/"))
                _, mae = test.test_model(
                    model=model,
                    dataset=dataset["validation"],
                    rep=rep,
                    device=device,
                    mean=mean,
                    std=std,
                    max_len=tgt_len,
                )
                logging.info(f"Validation MAE: {mae}")
                torch.save(
                    model.state_dict(), f"{args.save_model_path}/{epoch}.model"
                )
                if len(val_losses) == 0 or val_loss <= min(val_losses):
                    torch.save(
                        model.state_dict(), f"{args.save_model_path}/best.model")
            opt.scheduler.step(val_loss)
            if args.wandb:wandb.log({"Mean Angle Error": mae, "training loss": epoch_loss, "validation loss":val_loss})
        if args.wandb: run.finish()
        return training_losses, val_losses


    else:
        opt = utils.prepare_optimizer(model, args.optimizer, args.lr)
        logging.info("Training model...")

        for epoch in range(args.epochs):
            epoch_loss = 0
            model.train()
            teacher_forcing_ratio = np.clip(
                (1 - 2 * epoch / args.epochs), a_min=0, a_max=1,
            )
            logging.info(
                f"Running epoch {epoch} | "
                f"teacher_forcing_ratio={teacher_forcing_ratio}"
            )
            for iterations, (src_seqs, tgt_seqs) in enumerate(dataset["train"]):
                opt.optimizer.zero_grad()
                src_seqs, tgt_seqs = src_seqs.to(device), tgt_seqs.to(device)
                outputs = model(
                    src_seqs, tgt_seqs, teacher_forcing_ratio=teacher_forcing_ratio
                )
                outputs = outputs.double()
                loss = criterion(
                    outputs,
                    utils.prepare_tgt_seqs(args.architecture, src_seqs, tgt_seqs),
                )
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            epoch_loss = epoch_loss / num_training_sequences
            training_losses.append(epoch_loss)
            val_loss = generate.eval(
                model, criterion, dataset["validation"], args.batch_size, device,
            )
            val_losses.append(val_loss)
            opt.epoch_step(val_loss=val_loss)
            logging.info(
                f"Training loss {epoch_loss} | "
                f"Validation loss {val_loss} | "
                f"Iterations {iterations + 1}"
            )
            if epoch % args.save_model_frequency == 0:
                _, rep = os.path.split(args.preprocessed_path.strip("/"))
                _, mae = test.test_model(
                    model=model,
                    dataset=dataset["validation"],
                    rep=rep,
                    device=device,
                    mean=mean,
                    std=std,
                    max_len=tgt_len,
                )
                logging.info(f"Validation MAE: {mae}")
                torch.save(
                    model.state_dict(), f"{args.save_model_path}/{epoch}.model"
                )
                if len(val_losses) == 0 or val_loss <= min(val_losses):
                    torch.save(
                        model.state_dict(), f"{args.save_model_path}/best.model")
            opt.scheduler.step(val_loss)
            if args.wandb:wandb.log({"Mean Angle Error": mae, "training loss": epoch_loss, "validation loss":val_loss})
        return training_losses, val_losses

def plot_curves(args, training_losses, val_losses):
    plt.plot(range(len(training_losses)), training_losses)
    plt.plot(range(len(val_losses)), val_losses)
    plt.ylabel("MSE Loss")
    plt.xlabel("Epoch")
    plt.savefig(f"{args.save_model_path}/loss.svg", format="svg")


def main(args):
    train_losses, val_losses = train(args)
    plot_curves(args, train_losses, val_losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequence to sequence motion prediction training"
    )
    parser.add_argument(
        "--preprocessed-path",
        type=str,
        help="Path to folder with pickled " "files",
        required=True,
    )
    parser.add_argument(
        "--batch-size", type=int, help="Batch size for training", default=64
    )
    parser.add_argument(
        "--shuffle", action='store_true',
        help="Use this option to enable shuffling",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        help="Hidden size of LSTM units in encoder/decoder",
        default=1024,
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        help="Number of layers of LSTM/Transformer in encoder/decoder",
        default=1,
    )
    parser.add_argument(
        "--save-model-path",
        type=str,
        help="Path to store saved models",
        required=True,
    )
    parser.add_argument(
        "--save-model-frequency",
        type=int,
        help="Frequency (in terms of number of epochs) at which model is "
        "saved",
        default=1,
    )
    parser.add_argument(
        "--epochs", type=int, help="Number of training epochs", default=200
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Training device",
        default=None,
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--architecture",
        type=str,
        help="Seq2Seq archtiecture to be used",
        default="seq2seq",
        choices=[
            "seq2seq",
            "tied_seq2seq",
            "transformer",
            "transformer_encoder",
            "rnn",
            "tcn",    ## Newly Added
            "tcn_gan" ## Newly Added
        ],
    )
    ####################################################################################################
    ## Newly Added parsers
    parser.add_argument(
        "--wandb", action='store_true', help="Weights And Biases API callback for real time data logging", default=False,
    )
    parser.add_argument(
        "--attention", action='store_true', help="Want Attention?", default=False,
    )
    parser.add_argument(
        "--attention_heads",type=int, help="Number of attention heads", default=1,
    )
    parser.add_argument(
        "--attention_activation",type=str, help="Attention activation", default='softmax',
    )
    parser.add_argument(
        "--gan_activation",type=str, help="TCN GAN activation at the terminal layer", default='sigmoid',
    )
    ####################################################################################################



    parser.add_argument(
        "--lr", type=float, help="Learning rate", default=None,
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        help="Torch optimizer",
        default="sgd",
        choices=["adam", "sgd", "noamopt"],
    )
    args = parser.parse_args()
    main(args)
